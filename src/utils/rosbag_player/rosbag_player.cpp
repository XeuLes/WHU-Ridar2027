#include <cv_bridge/cv_bridge.h>  // 用于OpenCV图像与ROS图像消息的转换
#include <thread>                 // 用于多线程操作
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>  // OpenCV库，用于图像处理
#include <rclcpp/rclcpp.hpp>   // ROS2 C++核心库
#include <rclcpp_components/register_node_macro.hpp>  // 用于注册ROS2组件节点
#include <rosbag2_cpp/reader.hpp>  // rosbag2读取器，用于读取bag文件
#include <rosbag2_cpp/readers/sequential_reader.hpp>  // 顺序读取器实现
#include <rosbag2_storage/serialized_bag_message.hpp>  // 序列化bag消息类型
#include <sensor_msgs/msg/compressed_image.hpp>  // 压缩图像消息类型
#include <sensor_msgs/msg/detail/image__struct.hpp>  // 图像消息结构定义
#include <sensor_msgs/msg/image.hpp>                 // 图像消息类型
#include <sensor_msgs/msg/point_cloud2.hpp>          // 点云消息类型
#include <vision_interface/msg/match_info.hpp>  // 自定义匹配信息消息类型

using namespace std::chrono_literals;  // 简化时间单位使用（如100ms）

/**
 * @brief 信号处理函数，用于捕获Ctrl+C退出信号
 * @param sig 信号值（未使用）
 */
void on_exit([[maybe_unused]] int sig)
{
    RCUTILS_LOG_INFO("Exit by Ctrl+C");  // 打印退出信息
    rclcpp::shutdown();                  // 关闭ROS2节点
    exit(0);                             // 退出程序
}

/**
 * @brief
 * RosbagPlayer类，继承自rclcpp::Node，用于读取并播放rosbag文件中的消息
 */
class RosbagPlayer : public rclcpp::Node {
public:
    /**
     * @brief 构造函数，初始化节点并设置参数、发布者和线程
     * @param options 节点选项
     */
    RosbagPlayer(const rclcpp::NodeOptions& options)
        : Node("rosbag_player_node",
               options)  // 初始化节点，名称为"rosbag_player_node"
    {
        // 声明并获取"rosbag_file"参数（bag文件路径），默认值为空
        this->declare_parameter<std::string>("rosbag_file", "");
        this->get_parameter("rosbag_file", rosbag_file);
        this->declare_parameter<std::string>("lidar_input_topic", "/livox/lidar");
        this->declare_parameter<std::string>("image_input_topic", "/camera0/image");
        this->declare_parameter<std::string>("compressed_image_input_topic",
                                            "/compressed_image");
        this->get_parameter("lidar_input_topic", lidar_input_topic_);
        this->get_parameter("image_input_topic", image_input_topic_);
        this->get_parameter("compressed_image_input_topic",
                            compressed_image_input_topic_);
        this->declare_parameter<bool>("static_image_only", false);
        this->declare_parameter<double>("static_image_publish_rate", 5.0);
        this->get_parameter("static_image_only", static_image_only_);
        this->get_parameter("static_image_publish_rate",
                            static_image_publish_rate_);

        // 创建点云发布者，话题为"/livox/lidar"，队列长度10
        pointcloud_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10);

        // 创建图像发布者，话题为"camera_image"，使用传感器数据QoS
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera_image", rclcpp::SensorDataQoS());

        // 创建匹配信息发布者，话题为"/match_info"，队列长度10
        match_info_publisher_ =
            this->create_publisher<vision_interface::msg::MatchInfo>(
                "/match_info", 10);

        // 注册信号处理函数，捕获Ctrl+C信号
        signal(SIGINT, on_exit);

        // 打开指定的rosbag文件
        reader_.open(rosbag_file);

        // 启动线程处理bag播放逻辑
        if (static_image_only_) {
            processing_thread_ = std::make_shared<std::thread>(
                &RosbagPlayer::publish_static_image_from_bag, this);
        } else {
            processing_thread_ =
                std::make_shared<std::thread>(&RosbagPlayer::play_bag, this);
        }
    }

    /**
     * @brief 析构函数，确保线程正确退出
     */
    ~RosbagPlayer()
    {
        // 若线程存在且可join，则等待线程结束
        if (processing_thread_ && processing_thread_->joinable()) {
            processing_thread_->join();
        }
    }

private:
    bool read_next_image(sensor_msgs::msg::Image& output_msg)
    {
        while (rclcpp::ok() && reader_.has_next()) {
            auto bag_message = reader_.read_next();
            if (bag_message->topic_name == image_input_topic_) {
                auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  image_msg.get());
                output_msg = *image_msg;
                return true;
            }

            if (bag_message->topic_name != compressed_image_input_topic_) {
                continue;
            }

            auto image_msg =
                std::make_shared<sensor_msgs::msg::CompressedImage>();
            rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
                serialization;
            rclcpp::SerializedMessage serialized_msg(
                *bag_message->serialized_data);
            serialization.deserialize_message(&serialized_msg,
                                              image_msg.get());

            auto img = cv::imdecode(image_msg->data, cv::IMREAD_COLOR);
            if (img.empty()) {
                RCLCPP_WARN(this->get_logger(),
                            "Decoded compressed image is empty, skipping.");
                continue;
            }

            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                           .toImageMsg();
            output_msg = *msg;
            return true;
        }

        return false;
    }

    void publish_static_image_from_bag()
    {
        sensor_msgs::msg::Image static_image;
        if (!read_next_image(static_image)) {
            RCLCPP_ERROR(this->get_logger(),
                         "No valid image found in rosbag: %s "
                         "(image_input_topic=%s, compressed_image_input_topic=%s)",
                         rosbag_file.c_str(), image_input_topic_.c_str(),
                         compressed_image_input_topic_.c_str());
            return;
        }

        const double safe_rate =
            std::max(0.1, static_image_publish_rate_);
        const auto sleep_duration = std::chrono::duration<double>(1.0 / safe_rate);

        RCLCPP_INFO(this->get_logger(),
                    "Static rosbag image mode: publishing first image at %.2f Hz.",
                    safe_rate);

        while (rclcpp::ok()) {
            static_image.header.stamp = this->now();
            image_publisher_->publish(static_image);
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    /**
     * @brief 播放bag文件的核心函数，循环读取并发布消息
     */
    void play_bag()
    {
        // 当ROS2节点正常运行时循环
        while (rclcpp::ok()) {
            // 若bag读取完毕，重新打开bag（实现循环播放）
            if (!reader_.has_next())
                reader_.open(rosbag_file);

            // 记录处理开始时间（用于控制发布速率）
            auto start_time = std::chrono::high_resolution_clock::now();
            // 读取下一条bag消息
            auto bag_message = reader_.read_next();
            // 获取当前ROS时间（用于设置消息时间戳）
            auto ros_time = rclcpp::Clock().now();

            // 根据话题名称判断消息类型并处理
            if (bag_message->topic_name == lidar_input_topic_) {
                // 处理点云消息
                auto pointcloud_msg =
                    std::make_shared<sensor_msgs::msg::PointCloud2>();
                // 创建序列化器
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2>
                    serialization;
                // 包装序列化数据
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                // 反序列化消息
                serialization.deserialize_message(&serialized_msg,
                                                  pointcloud_msg.get());
                // 设置消息时间戳为当前ROS时间
                pointcloud_msg->header.stamp = ros_time;
                // 发布点云消息
                pointcloud_publisher_->publish(*pointcloud_msg);
            } else if (bag_message->topic_name == image_input_topic_) {
                // 处理原始图像消息
                auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  image_msg.get());
                image_msg->header.stamp = ros_time;
                image_publisher_->publish(*image_msg);
            } else if (bag_message->topic_name == compressed_image_input_topic_) {
                // 处理压缩图像消息
                auto image_msg =
                    std::make_shared<sensor_msgs::msg::CompressedImage>();
                // 创建序列化器
                rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
                    serialization;
                // 包装序列化数据
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                // 反序列化消息
                serialization.deserialize_message(&serialized_msg,
                                                  image_msg.get());
                // 解码压缩图像为OpenCV格式
                auto img = cv::imdecode(image_msg->data, cv::IMREAD_COLOR);
                // 转换为ROS图像消息（bgr8格式）
                auto msg =
                    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                        .toImageMsg();
                // 设置时间戳
                msg->header.stamp = ros_time;
                // 发布图像消息
                image_publisher_->publish(*msg);
            } else if (bag_message->topic_name == "/match_info") {
                // 处理匹配信息消息
                auto match_info_msg =
                    std::make_shared<vision_interface::msg::MatchInfo>();
                // 创建序列化器
                rclcpp::Serialization<vision_interface::msg::MatchInfo>
                    serialization;
                // 包装序列化数据
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                // 反序列化消息
                serialization.deserialize_message(&serialized_msg,
                                                  match_info_msg.get());
                // 发布匹配信息消息
                match_info_publisher_->publish(*match_info_msg);
            }

            // 计算当前消息处理耗时
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count();

            // 控制发布速率：若处理耗时小于100ms且大于1ms，则休眠补足到100ms
            if ((duration < 100) && (duration > 1)) {
                std::this_thread::sleep_for(
                    100ms - std::chrono::milliseconds(duration));
            }
        }
        // 当节点退出时打印信息
        RCLCPP_INFO(this->get_logger(), "No more messages in the bag.");
        rclcpp::shutdown();  // 关闭ROS2节点
    }

    // 点云消息发布者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pointcloud_publisher_;
    // 图像消息发布者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    // 匹配信息消息发布者
    rclcpp::Publisher<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_publisher_;
    // rosbag读取器
    rosbag2_cpp::Reader reader_;
    // 处理bag播放的线程
    std::shared_ptr<std::thread> processing_thread_;
    // rosbag文件路径
    std::string rosbag_file;
    std::string lidar_input_topic_{"/livox/lidar"};
    std::string image_input_topic_{"/camera0/image"};
    std::string compressed_image_input_topic_{"/compressed_image"};
    // 只从 rosbag 读取第一帧图像并静态发布，用于启动前 PnP 标定
    bool static_image_only_{false};
    double static_image_publish_rate_{5.0};
};

// 注册节点为ROS2组件，使其可被发现和加载
RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)
