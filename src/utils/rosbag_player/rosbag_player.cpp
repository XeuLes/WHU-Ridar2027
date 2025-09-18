#include <arpa/inet.h>
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage/storage_options.hpp>  // 补充StorageOptions头文件
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <vision_interface/msg/match_info.hpp>

// -------------------------- 字节序转换工具函数 --------------------------
inline uint64_t le_to_be64(uint64_t le_val)
{
    return htobe64(le_val);
}
inline uint32_t le_to_be32(uint32_t le_val)
{
    return htobe32(le_val);
}

// -------------------------- LVX2 格式定义 --------------------------
#pragma pack(push, 1)
struct Lvx2FileHeader {
    char     signature[8];       // "LVX2.0" + 2字节'\0'
    uint64_t version;            // 大端：0x0000000100000000
    uint64_t header_size;        // 大端：sizeof(Lvx2FileHeader)
    uint64_t data_size;          // 大端：数据区总大小
    uint64_t frame_count;        // 大端：总帧数
    uint64_t device_count;       // 大端：1（单设备）
    uint8_t  device_info[1024];  // 设备信息
};

struct LivoxDeviceInfo {
    char    model[32];      // 设备型号："Livox Mid-40"
    char    sn[64];         // 序列号："UNKNOWN_SN"
    char    firmware[32];   // 固件版本："v2.1.0"
    uint8_t reserved[896];  // 填充0
};

struct Lvx2FrameHeader {
    uint64_t frame_index;     // 大端：帧序号
    uint64_t timestamp_type;  // 大端：2（ROS时间）
    uint64_t timestamp;       // 大端：微秒时间戳
    uint32_t point_count;     // 大端：本帧点数
    uint8_t  device_index;    // 0（单设备）
    uint8_t  reserved[3];     // 填充0
};

struct Lvx2Point {
    float    x;          // 小端：X坐标（米）
    float    y;          // 小端：Y坐标（米）
    float    z;          // 小端：Z坐标（米）
    uint8_t  intensity;  // 强度（0-255）
    uint8_t  tag;        // 0（普通点）
    uint16_t line;       // 小端：0（单线束）
};
#pragma pack(pop)

// -------------------------- LVX2
// 写入工具类（修复fpos运算）--------------------------
namespace fs = std::filesystem;
class Lvx2Writer {
public:
    Lvx2Writer(const std::string& save_path)
        : file_(save_path, std::ios::binary | std::ios::trunc),
          frame_count_(0)
    {
        if (!file_.is_open()) {
            throw std::runtime_error("LVX2文件创建失败：" + save_path);
        }

        Lvx2FileHeader header{};
        memcpy(header.signature, "LVX2.0", 6);
        header.signature[6] = '\0';
        header.signature[7] = '\0';
        header.version = le_to_be64(0x0000000100000000);
        header.header_size = le_to_be64(sizeof(Lvx2FileHeader));
        header.device_count = le_to_be64(1);
        header.data_size = le_to_be64(0);
        header.frame_count = le_to_be64(0);

        LivoxDeviceInfo dev_info{};
        memcpy(dev_info.model, "Livox Mid-40", sizeof("Livox Mid-40") - 1);
        memcpy(dev_info.sn, "UNKNOWN_SN", sizeof("UNKNOWN_SN") - 1);
        memcpy(dev_info.firmware, "v2.1.0", sizeof("v2.1.0") - 1);
        memcpy(header.device_info, &dev_info, sizeof(LivoxDeviceInfo));

        file_.write(reinterpret_cast<const char*>(&header),
                    sizeof(Lvx2FileHeader));
        RCLCPP_INFO(rclcpp::get_logger("Lvx2Writer"),
                    "LVX2文件创建成功：%s", save_path.c_str());
    }

    ~Lvx2Writer()
    {
        if (!file_.is_open())
            return;

        // 【修复1】将std::fpos转为streamoff（避免类型不匹配）
        const std::streamoff file_end =
            file_.tellp();  // 用streamoff接收文件指针位置
        const uint64_t data_size =
            static_cast<uint64_t>(file_end - sizeof(Lvx2FileHeader));

        // 更新data_size（定位到data_size字段偏移）
        file_.seekp(offsetof(Lvx2FileHeader, data_size));
        const uint64_t data_size_be = le_to_be64(data_size);
        file_.write(reinterpret_cast<const char*>(&data_size_be),
                    sizeof(data_size_be));

        // 更新frame_count
        file_.seekp(offsetof(Lvx2FileHeader, frame_count));
        const uint64_t frame_count_be = le_to_be64(frame_count_);
        file_.write(reinterpret_cast<const char*>(&frame_count_be),
                    sizeof(frame_count_be));

        file_.close();
        RCLCPP_INFO(rclcpp::get_logger("Lvx2Writer"),
                    "LVX2文件保存完成：总帧数=%lu，数据区大小=%lu字节",
                    frame_count_, data_size);
    }

    void write_frame(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud)
    {
        if (!file_.is_open() || !cloud)
            return;
        if (!check_cloud_fields(*cloud)) {
            RCLCPP_WARN(rclcpp::get_logger("Lvx2Writer"),
                        "点云缺少x/y/z/intensity字段，跳过该帧");
            return;
        }

        Lvx2FrameHeader frame_header{};
        frame_header.frame_index = le_to_be64(frame_count_);
        frame_header.timestamp_type = le_to_be64(2);
        const uint64_t timestamp_us = cloud->header.stamp.sec * 1000000ULL +
                                      cloud->header.stamp.nanosec / 1000ULL;
        frame_header.timestamp = le_to_be64(timestamp_us);
        const uint32_t point_count = cloud->width * cloud->height;
        frame_header.point_count = le_to_be32(point_count);
        frame_header.device_index = 0;

        file_.write(reinterpret_cast<const char*>(&frame_header),
                    sizeof(Lvx2FrameHeader));

        sensor_msgs::PointCloud2ConstIterator<float>   iter_x(*cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float>   iter_y(*cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float>   iter_z(*cloud, "z");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_int(
            *cloud, "intensity");

        Lvx2Point lvx_point{};
        for (size_t i = 0; i < point_count;
             ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_int) {
            lvx_point.x = *iter_x;
            lvx_point.y = *iter_y;
            lvx_point.z = *iter_z;
            lvx_point.intensity = *iter_int;
            lvx_point.tag = 0;
            lvx_point.line = 0;
            file_.write(reinterpret_cast<const char*>(&lvx_point),
                        sizeof(Lvx2Point));
        }

        frame_count_++;
    }

private:
    bool check_cloud_fields(const sensor_msgs::msg::PointCloud2& cloud)
    {
        std::unordered_set<std::string> required = {"x", "y", "z",
                                                    "intensity"};
        for (const auto& field : cloud.fields) {
            required.erase(field.name);
        }
        return required.empty();
    }

    std::ofstream file_;         // LVX2文件流
    uint64_t      frame_count_;  // 已写入帧数
};

// -------------------------- 信号处理 --------------------------
void sigint_handler([[maybe_unused]] int sig)
{
    RCLCPP_INFO(rclcpp::get_logger("RosbagPlayer"),
                "收到退出信号（Ctrl+C）");
    rclcpp::shutdown();
    exit(EXIT_SUCCESS);
}

// --------------------------
// 核心节点（修复rosbag2读取器）--------------------------
class RosbagPlayer : public rclcpp::Node {
public:
    explicit RosbagPlayer(const rclcpp::NodeOptions& options)
        : Node("rosbag_player", options),
          // 【修复2】使用SequentialReader类型（而非基类Reader）
          rosbag_reader_(
              std::make_unique<rosbag2_cpp::readers::SequentialReader>())
    {
        // 1. 获取rosbag路径参数
        declare_parameter("rosbag_file", "");
        const std::string rosbag_path =
            get_parameter("rosbag_file").as_string();
        if (rosbag_path.empty()) {
            RCLCPP_ERROR(get_logger(), "请指定rosbag路径：--ros-args -p "
                                       "rosbag_file:=/path/to/bag.db3");
            rclcpp::shutdown();
            return;
        }

        // 2. 初始化LVX2写入器
        init_lvx2_writer(rosbag_path);

        // 3. 创建发布者
        pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10);
        pub_image_ = create_publisher<sensor_msgs::msg::Image>(
            "camera_image", rclcpp::SensorDataQoS());
        pub_match_ = create_publisher<vision_interface::msg::MatchInfo>(
            "/match_info", 10);

        // 4. 注册退出信号
        signal(SIGINT, sigint_handler);

        // 5. 打开rosbag
        if (!open_rosbag(rosbag_path)) {
            rclcpp::shutdown();
            return;
        }

        // 6. 启动播放线程
        play_thread_ =
            std::make_unique<std::thread>(&RosbagPlayer::play_rosbag, this);
    }

    ~RosbagPlayer()
    {
        if (play_thread_ && play_thread_->joinable()) {
            play_thread_->join();
        }
        lvx2_writer_.reset();
        RCLCPP_INFO(get_logger(), "节点资源已释放");
    }

private:
    void init_lvx2_writer(const std::string& rosbag_path)
    {
        const char* home_dir = getenv("HOME");
        if (!home_dir) {
            RCLCPP_ERROR(get_logger(), "无法获取用户主目录，LVX2保存失败");
            return;
        }

        const fs::path    bag_path(rosbag_path);
        const std::string lvx2_name =
            bag_path.stem().string() + "_lvx2.lvx2";
        const std::string lvx2_path =
            std::string(home_dir) + "/" + lvx2_name;

        try {
            lvx2_writer_ = std::make_unique<Lvx2Writer>(lvx2_path);
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "LVX2写入器初始化失败：%s",
                         e.what());
        }
    }

    // 【修复3】使用rosbag2_storage::StorageOptions（而非rosbag2_cpp）
    bool open_rosbag(const std::string& rosbag_path)
    {
        try {
            rosbag2_storage::StorageOptions storage_opts;
            storage_opts.uri = rosbag_path;
            storage_opts.storage_id = "sqlite3";  // 明确指定存储插件

            // 添加 ConverterOptions
            rosbag2_cpp::ConverterOptions converter_opts;
            converter_opts.input_serialization_format = "cdr";
            converter_opts.output_serialization_format = "cdr";

            rosbag_reader_->open(storage_opts, converter_opts);
            RCLCPP_INFO(get_logger(), "成功打开rosbag：%s",
                        rosbag_path.c_str());
            return true;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "打开rosbag失败：%s", e.what());
            return false;
        }
    }

    void play_rosbag()
    {
        while (rclcpp::ok() && rosbag_reader_->has_next()) {
            const auto start_time =
                std::chrono::high_resolution_clock::now();

            const auto bag_msg = rosbag_reader_->read_next();
            if (!bag_msg)
                continue;
            const std::string& topic = bag_msg->topic_name;

            // 处理点云
            if (topic == "/livox/lidar") {
                auto cloud =
                    std::make_shared<sensor_msgs::msg::PointCloud2>();
                deserialize_msg(*bag_msg, *cloud);
                cloud->header.stamp = now();
                pub_cloud_->publish(*cloud);
                if (lvx2_writer_) {
                    lvx2_writer_->write_frame(cloud);
                }
            }

            // 处理压缩图像
            else if (topic == "/compressed_image") {
                auto compressed_img =
                    std::make_shared<sensor_msgs::msg::CompressedImage>();
                deserialize_msg(*bag_msg, *compressed_img);

                try {
                    const cv::Mat img = cv::imdecode(compressed_img->data,
                                                     cv::IMREAD_COLOR);
                    if (img.empty()) {
                        RCLCPP_WARN(get_logger(), "压缩图像解码为空");
                        continue;
                    }
                    auto img_msg = cv_bridge::CvImage(
                                       std_msgs::msg::Header(), "bgr8", img)
                                       .toImageMsg();
                    img_msg->header.stamp = now();
                    pub_image_->publish(*img_msg);
                }
                catch (const cv_bridge::Exception& e) {
                    RCLCPP_WARN(get_logger(), "图像解码失败：%s", e.what());
                }
            }

            // 处理匹配信息
            else if (topic == "/match_info") {
                auto match_info =
                    std::make_shared<vision_interface::msg::MatchInfo>();
                deserialize_msg(*bag_msg, *match_info);
                pub_match_->publish(*match_info);
            }

            // 控制播放速率（10Hz）
            const auto process_dur =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time)
                    .count();
            if (process_dur < 100 && process_dur > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100 - process_dur));
            }
        }

        RCLCPP_INFO(get_logger(), "rosbag播放完成！");
        rclcpp::shutdown();
    }

    template <typename MsgT>
    void
    deserialize_msg(const rosbag2_storage::SerializedBagMessage& bag_msg,
                    MsgT&                                        msg)
    {
        rclcpp::Serialization<MsgT> serializer;
        rclcpp::SerializedMessage serialized_msg(*bag_msg.serialized_data);
        serializer.deserialize_message(&serialized_msg, &msg);
    }

    // 成员变量
    std::unique_ptr<rosbag2_cpp::readers::SequentialReader>
        rosbag_reader_;  // 【修复2】具体类型而非基类
    std::unique_ptr<std::thread> play_thread_;
    std::unique_ptr<Lvx2Writer>  lvx2_writer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr       pub_image_;
    rclcpp::Publisher<vision_interface::msg::MatchInfo>::SharedPtr
        pub_match_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)
