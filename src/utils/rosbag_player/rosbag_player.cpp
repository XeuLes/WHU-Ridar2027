#include <cv_bridge/cv_bridge.h>
#include <cstdlib>  // 用于 getenv
#include <filesystem>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>  // 用于安全解析点云字段
#include <vision_interface/msg/match_info.hpp>

// -------------------------- LVX2 格式定义（严格按 Livox
// 规范）--------------------------
#pragma pack(push, 1)  // 强制1字节对齐，避免结构体成员偏移错误
// LVX2 文件头（参考 Livox 官方格式文档）
struct Lvx2FileHeader {
    char signature[8];  // 固定标识："LVX2.0\0"（6字符+2字节结束符）
    uint64_t version;  // 版本号：固定 0x0000000100000000（1.0版本）
    uint64_t header_size;  // 文件头总大小：固定 sizeof(Lvx2FileHeader)
    uint64_t data_size;     // 数据区总大小（后续计算更新）
    uint64_t frame_count;   // 总帧数（后续统计更新）
    uint64_t device_count;  // 设备数量：默认1（单Livox设备）
    uint8_t device_info[1024];  // 设备信息（简化：填充设备型号标识）
};

// LVX2 单帧头（每帧点云的元信息）
struct Lvx2FrameHeader {
    uint64_t frame_index;     // 帧序号（从0开始递增）
    uint64_t timestamp_type;  // 时间戳类型：2=ROS时间（1=UTC时间）
    uint64_t timestamp;  // 时间戳（微秒级，与点云采集时间同步）
    uint32_t point_count;   // 本帧点数量
    uint8_t  device_index;  // 设备索引：0（单设备场景）
    uint8_t  reserved[3];   // 保留字段：填充0
};

// LVX2 单点数据（Livox 标准点格式：x/y/z为米，intensity为0-255）
struct Lvx2Point {
    float   x;          // X坐标（单位：米）
    float   y;          // Y坐标（单位：米）
    float   z;          // Z坐标（单位：米）
    uint8_t intensity;  // 强度（0-255，Livox 雷达输出范围）
    uint8_t tag;  // 点类型标签：0=普通点，1=边缘点（默认0）
    uint16_t line;  // 激光线号：Livox 单线束填0（多线束按实际填写）
};
#pragma pack(pop)  // 恢复默认对齐方式

// -------------------------- LVX2
// 写入工具类（封装文件操作）--------------------------
namespace fs = std::filesystem;
class Lvx2Writer {
public:
    // 构造函数：创建LVX2文件并初始化头
    Lvx2Writer(const std::string& save_path)
        : file_(save_path, std::ios::binary | std::ios::trunc)
    {
        if (!file_.is_open()) {
            throw std::runtime_error("LVX2文件创建失败：" + save_path);
        }

        // 1. 初始化文件头
        memset(&file_header_, 0, sizeof(Lvx2FileHeader));
        memcpy(file_header_.signature, "LVX2.0\0", 8);      // 固定标识
        file_header_.version = 0x0000000100000000;          // 版本号
        file_header_.header_size = sizeof(Lvx2FileHeader);  // 头大小
        file_header_.device_count = 1;                      // 单设备
        // 填充设备信息（示例：标记为 Livox Mid-40，可根据实际设备修改）
        std::string dev_info = "Livox Mid-40; ROS2 PointCloud2 Export";
        memcpy(file_header_.device_info, dev_info.c_str(),
               std::min(dev_info.size(), 1024ul));

        // 2. 先写入临时文件头（后续析构时更新真实数据）
        file_.write(reinterpret_cast<const char*>(&file_header_),
                    sizeof(Lvx2FileHeader));
        RCLCPP_INFO(rclcpp::get_logger("Lvx2Writer"), "LVX2文件已创建：%s",
                    save_path.c_str());
    }

    // 析构函数：更新文件头并关闭文件（确保文件完整性）
    ~Lvx2Writer()
    {
        if (file_.is_open()) {
            // 1. 计算数据区大小（总文件大小 - 头大小）
            file_.seekp(0, std::ios::end);
            uint64_t total_file_size = file_.tellp();
            file_header_.data_size =
                total_file_size - file_header_.header_size;
            // 2. 更新文件头的帧数和数据大小
            file_.seekp(0);  // 回到文件开头
            file_.write(reinterpret_cast<const char*>(&file_header_),
                        sizeof(Lvx2FileHeader));
            // 3. 关闭文件
            file_.close();
            RCLCPP_INFO(rclcpp::get_logger("Lvx2Writer"),
                        "LVX2文件保存完成：总帧数=%lu，数据大小=%lu字节",
                        file_header_.frame_count, file_header_.data_size);
        }
    }

    // 写入一帧点云到LVX2文件（核心接口）
    void writePointCloudFrame(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud)
    {
        if (!file_.is_open() || !cloud)
            return;

        // 1. 验证点云是否包含 x/y/z/intensity 字段（避免解析错误）
        if (!hasRequiredFields(*cloud)) {
            RCLCPP_WARN(rclcpp::get_logger("Lvx2Writer"),
                        "点云缺少必要字段（x/y/z/intensity），跳过该帧");
            return;
        }

        // 2. 准备帧头
        Lvx2FrameHeader frame_header;
        memset(&frame_header, 0, sizeof(Lvx2FrameHeader));
        frame_header.frame_index = file_header_.frame_count;  // 帧序号递增
        frame_header.timestamp_type = 2;  // ROS时间戳类型
        // 转换ROS时间戳为微秒（sec*1e6 + nsec/1e3）
        frame_header.timestamp =
            static_cast<uint64_t>(cloud->header.stamp.sec * 1e6 +
                                  cloud->header.stamp.nanosec / 1e3);
        frame_header.point_count =
            cloud->width * cloud->height;  // 本帧总点数
        frame_header.device_index = 0;     // 单设备索引

        // 3. 写入帧头
        file_.write(reinterpret_cast<const char*>(&frame_header),
                    sizeof(Lvx2FrameHeader));

        // 4.
        // 解析PointCloud2并写入点数据（使用安全迭代器，避免字段偏移错误）
        sensor_msgs::PointCloud2ConstIterator<float>   iter_x(*cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float>   iter_y(*cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float>   iter_z(*cloud, "z");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_intensity(
            *cloud, "intensity");

        Lvx2Point lvx_point;
        for (size_t i = 0; i < frame_header.point_count;
             ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            lvx_point.x = *iter_x;
            lvx_point.y = *iter_y;
            lvx_point.z = *iter_z;
            lvx_point.intensity =
                *iter_intensity;  // 假设强度已在0-255范围（Livox输出默认满足）
            lvx_point.tag = 0;   // 普通点标签
            lvx_point.line = 0;  // 单线束设备填0

            file_.write(reinterpret_cast<const char*>(&lvx_point),
                        sizeof(Lvx2Point));
        }

        // 5. 帧数递增
        file_header_.frame_count++;
    }

private:
    // 检查点云是否包含 x/y/z/intensity 必要字段
    bool hasRequiredFields(const sensor_msgs::msg::PointCloud2& cloud)
    {
        std::unordered_set<std::string> required_fields = {"x", "y", "z",
                                                           "intensity"};
        for (const auto& field : cloud.fields) {
            required_fields.erase(field.name);
        }
        return required_fields.empty();  // 为空表示所有必要字段都存在
    }

    std::ofstream  file_;         // LVX2文件流
    Lvx2FileHeader file_header_;  // LVX2文件头（实时更新）
};

// -------------------------- 信号处理：Ctrl+C 优雅退出
// --------------------------
void on_exit([[maybe_unused]] int sig)
{
    RCLCPP_INFO(rclcpp::get_logger("RosbagPlayer"),
                "收到退出信号（Ctrl+C）");
    rclcpp::shutdown();
    exit(0);
}

// -------------------------- 核心节点：Rosbag播放+LVX2保存
// --------------------------
class RosbagPlayer : public rclcpp::Node {
public:
    RosbagPlayer(const rclcpp::NodeOptions& options)
        : Node("rosbag_player_node", options)
    {
        // 1. 声明并获取ROS参数（rosbag文件路径）
        this->declare_parameter<std::string>("rosbag_file", "");
        this->get_parameter("rosbag_file", rosbag_file_);
        if (rosbag_file_.empty()) {
            RCLCPP_ERROR(this->get_logger(),
                         "请通过 'rosbag_file' 参数指定rosbag路径！");
            rclcpp::shutdown();
            return;
        }

        // 2.
        // 初始化LVX2写入器（保存到用户主目录，文件名：rosbag名称_lvx2.lvx2）
        initLvx2Writer();

        // 3. 创建消息发布者（保持原功能）
        pointcloud_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera_image", rclcpp::SensorDataQoS());
        match_info_pub_ =
            this->create_publisher<vision_interface::msg::MatchInfo>(
                "/match_info", 10);

        // 4. 注册退出信号处理
        signal(SIGINT, on_exit);

        // 5. 尝试打开rosbag文件
        try {
            rosbag_reader_.open(rosbag_file_);
            RCLCPP_INFO(this->get_logger(), "成功打开rosbag文件：%s",
                        rosbag_file_.c_str());
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "打开rosbag失败：%s",
                         e.what());
            rclcpp::shutdown();
            return;
        }

        // 6. 启动rosbag播放线程（避免阻塞主线程）
        play_thread_ =
            std::make_shared<std::thread>(&RosbagPlayer::playRosbag, this);
    }

    // 析构函数：等待线程结束并释放资源
    ~RosbagPlayer()
    {
        if (play_thread_ && play_thread_->joinable()) {
            play_thread_->join();
        }
        lvx2_writer_.reset();  // 触发LVX2文件头更新和文件关闭
        RCLCPP_INFO(this->get_logger(), "节点资源已释放");
    }

private:
    // 初始化LVX2写入器：生成主目录下的保存路径
    void initLvx2Writer()
    {
        // 获取用户主目录（兼容Linux/macOS）
        const char* home_dir = getenv("HOME");
        if (!home_dir) {
            RCLCPP_ERROR(this->get_logger(),
                         "无法获取用户主目录，LVX2保存失败");
            return;
        }

        // 生成LVX2文件名：从rosbag路径提取名称（如 "test.db3" →
        // "test_lvx2.lvx2"）
        fs::path    rosbag_path(rosbag_file_);
        std::string lvx2_filename =
            rosbag_path.stem().string() + "_lvx2.lvx2";
        std::string lvx2_save_path =
            std::string(home_dir) + "/" + lvx2_filename;

        // 创建LVX2写入器
        try {
            lvx2_writer_ = std::make_unique<Lvx2Writer>(lvx2_save_path);
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "LVX2写入器初始化失败：%s",
                         e.what());
        }
    }

    // 核心逻辑：播放rosbag并处理消息
    void playRosbag()
    {
        while (rclcpp::ok() && rosbag_reader_.has_next()) {
            // 记录处理开始时间（控制播放速率）
            auto start_time = std::chrono::high_resolution_clock::now();

            // 读取rosbag中的下一条消息
            auto bag_msg = rosbag_reader_.read_next();
            if (!bag_msg)
                continue;

            // --------------------------
            // 处理点云消息（核心：保存为LVX2）--------------------------
            if (bag_msg->topic_name == "/livox/lidar") {
                // 反序列化PointCloud2消息
                auto cloud_msg =
                    std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2>
                                          serializer;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_msg->serialized_data);
                serializer.deserialize_message(&serialized_msg,
                                               cloud_msg.get());

                // 更新点云时间戳为当前ROS时间（保持与播放同步）
                cloud_msg->header.stamp = this->now();
                // 发布点云（保持原功能）
                pointcloud_pub_->publish(*cloud_msg);
                // 保存点云到LVX2文件（若写入器初始化成功）
                if (lvx2_writer_) {
                    lvx2_writer_->writePointCloudFrame(cloud_msg);
                }
            }

            // --------------------------
            // 处理压缩图像消息（保持原功能）--------------------------
            else if (bag_msg->topic_name == "/compressed_image") {
                auto compressed_img_msg =
                    std::make_shared<sensor_msgs::msg::CompressedImage>();
                rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
                                          serializer;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_msg->serialized_data);
                serializer.deserialize_message(&serialized_msg,
                                               compressed_img_msg.get());

                // 解码压缩图像为OpenCV格式，再转为普通Image消息
                try {
                    cv::Mat img = cv::imdecode(compressed_img_msg->data,
                                               cv::IMREAD_COLOR);
                    auto    img_msg = cv_bridge::CvImage(
                                          std_msgs::msg::Header(), "bgr8", img)
                                       .toImageMsg();
                    img_msg->header.stamp = this->now();
                    image_pub_->publish(*img_msg);
                }
                catch (const cv_bridge::Exception& e) {
                    RCLCPP_WARN(this->get_logger(), "图像解码失败：%s",
                                e.what());
                }
            }

            // --------------------------
            // 处理匹配信息消息（保持原功能）--------------------------
            else if (bag_msg->topic_name == "/match_info") {
                auto match_info_msg =
                    std::make_shared<vision_interface::msg::MatchInfo>();
                rclcpp::Serialization<vision_interface::msg::MatchInfo>
                                          serializer;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_msg->serialized_data);
                serializer.deserialize_message(&serialized_msg,
                                               match_info_msg.get());

                match_info_pub_->publish(*match_info_msg);
            }

            // --------------------------
            // 控制播放速率（约10Hz，与原逻辑一致）--------------------------
            auto end_time = std::chrono::high_resolution_clock::now();
            auto process_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count();
            if (process_duration < 100 &&
                process_duration > 1) {  // 若处理耗时<100ms，休眠补全
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100 - process_duration));
            }
        }

        // rosbag播放完毕
        RCLCPP_INFO(this->get_logger(), "rosbag播放完成！");
        rclcpp::shutdown();
    }

    // 成员变量
    std::string                  rosbag_file_;    // rosbag文件路径
    rosbag2_cpp::Reader          rosbag_reader_;  // rosbag读取器
    std::shared_ptr<std::thread> play_thread_;    // 播放线程

    // 消息发布者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
                                                          pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_pub_;

    std::unique_ptr<Lvx2Writer>
        lvx2_writer_;  // LVX2写入器（智能指针自动管理）
};

// 注册节点为ROS 2组件（允许通过ComposableNode加载）
RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)
