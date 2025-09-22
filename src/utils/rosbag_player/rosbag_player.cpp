#include <cv_bridge/cv_bridge.h>  // 用于OpenCV图像与ROS图像消息的转换
#include <thread>                 // 用于多线程操作
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
        processing_thread_ =
            std::make_shared<std::thread>(&RosbagPlayer::play_bag, this);
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
            if (bag_message->topic_name == "/livox/lidar") {
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
            } else if (bag_message->topic_name == "/compressed_image") {
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
};

// 注册节点为ROS2组件，使其可被发现和加载
RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)
