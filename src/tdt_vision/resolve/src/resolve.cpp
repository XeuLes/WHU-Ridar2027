#include "resolve.h"

// 定义日志输出宏，用于打印信息到标准输出
#define TDT_INFO(msg) std::cout << msg << std::endl

// 雷达数据处理相关的命名空间
namespace tdt_radar {

/**
 * @brief Resolve类，继承自rclcpp::Node，用于处理雷达/视觉数据的解析与转换
 * 功能包括：订阅2D点、匹配信息、检测结果，转换为3D坐标，发布点云和处理后的检测结果
 */
Resolve::Resolve(const rclcpp::NodeOptions& node_options)
    : Node("radar_resolve_node",
           node_options)  // 初始化ROS节点，节点名为"radar_resolve_node"
{
    // 创建解析器实例（用于2D到3D坐标转换）
    parser_ = new parser();
    // 加载地图图片（RM2025比赛地图）
    minimap = cv::imread("config/RM2025.png");

    // 创建订阅者，订阅"camera_point2D"话题（2D点数据），回调函数为callback
    point_sub = this->create_subscription<geometry_msgs::msg::Vector3>(
        "camera_point2D", rclcpp::SensorDataQoS(),  // 使用传感器数据QoS配置
        std::bind(&Resolve::callback, this, std::placeholders::_1));

    // 创建发布者，发布"camera_point3D"话题（3D点云数据）
    pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera_point3D", rclcpp::SensorDataQoS());

    // 创建订阅者，订阅"match_info"话题（比赛信息，如标记点、机器人血量等）
    match_info_sub =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "match_info", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::MatchInfoCallback, this,
                      std::placeholders::_1));

    // 创建订阅者，订阅"detect_result"话题（检测结果，如红蓝方机器人2D坐标）
    detect_sub =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "detect_result", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::DetectCallback, this,
                      std::placeholders::_1));

    // 创建发布者，发布"resolve_result"话题（解析后的3D检测结果）
    pub_radar = this->create_publisher<vision_interface::msg::DetectResult>(
        "resolve_result", rclcpp::SensorDataQoS());

    // 打印节点初始化成功信息
    TDT_INFO("Load radar resolve node success!");
}

/**
 * @brief 处理比赛信息回调函数
 * 存储接收到的标记点、机器人血量和比赛时间
 * @param msg 订阅到的MatchInfo消息
 */
void Resolve::MatchInfoCallback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    // 存储6个标记点信息
    for (int i = 0; i < 6; i++) {
        markers[i] = msg->marks[i];
    }
    // 存储16个机器人的血量信息
    for (int i = 0; i < 16; i++) {
        robot_hp[i] = msg->robot_hp[i];
    }
    // 存储比赛时间
    match_time = msg->match_time;
}

/**
 * @brief 处理2D点坐标回调函数
 * 将2D点转换为3D坐标，生成点云并发布
 * @param msg 订阅到的Vector3消息（包含2D点x、y坐标）
 */
void Resolve::callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    // 从消息中提取2D点坐标
    cv::Point2f point;
    point.x = msg->x;
    point.y = msg->y;

    // 通过解析器将2D点转换为3D坐标
    auto center_point = parser_->parse(point);

    // 创建点云对象，用于存储3D点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ send_point;

    // 填充3D点坐标（y坐标做转换，可能是坐标系调整）
    send_point.x = center_point.x;
    send_point.y = 15 - center_point.y;  // 坐标系y轴方向调整
    std::cout << center_point << std::endl;  // 打印转换后的坐标（调试用）
    send_point.z = 1;  // z坐标固定为1（可能在平面地图上）

    // 将点添加到点云
    cloud->points.push_back(send_point);

    // 转换点云为ROS消息并发布
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "rm_frame";  // 设置坐标系ID
    pub->publish(output);
}

/**
 * @brief 处理检测结果回调函数
 * 解析红蓝双方机器人的2D坐标为3D坐标，在地图上绘制标记，生成带颜色的点云，发布处理结果
 * @param msg 订阅到的DetectResult消息（包含红蓝方机器人2D坐标）
 */
void Resolve::DetectCallback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    // 克隆地图图片（避免修改原图），用于绘制机器人位置
    cv::Mat              Map_clone = minimap.clone();
    std::vector<map_car> cars;  // 存储地图上的车辆信息（未使用）
    vision_interface::msg::DetectResult send_data;  // 待发布的解析结果
    // 创建带颜色的点云（RGBA）
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGBA>);

    // 处理红蓝双方各6个机器人的坐标
    for (int i = 0; i < 6; i++) {
        // 处理蓝方机器人
        cv::Point2f blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];

        // 若坐标有效（x和y不同时为0）
        if (blue_point.x * blue_point.y) {
            // 2D转3D坐标
            auto center_point = parser_->parse(blue_point);
            center_point.y = 15 + center_point.y;  // 蓝方y坐标调整

            // 存储解析后的3D坐标到发布消息
            send_data.blue_x[i] = center_point.x;
            send_data.blue_y[i] = center_point.y;

            // 创建带颜色的点（蓝色）
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 0;
            send_point.g = 0;
            send_point.b = 255;  // 蓝色
            send_point.a = 255;  // 不透明
            cloud->points.push_back(send_point);

            // 在地图上绘制蓝方机器人标记（圆形+编号）
            cv::circle(
                Map_clone,
                // 地图上的像素坐标（地图尺寸映射）
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(200, 0, 0), -1);  // 蓝色圆形，半径20，填充
            cv::putText(
                Map_clone, std::to_string(i + 1),  // 机器人编号（1-6）
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255),
                2);  // 白色文字
        }

        // 处理红方机器人（逻辑同蓝方，颜色不同）
        cv::Point2f red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];

        if (red_point.x * red_point.y) {
            auto center_point = parser_->parse(red_point);
            center_point.y = 15 + center_point.y;  // 红方y坐标调整

            send_data.red_x[i] = center_point.x;
            send_data.red_y[i] = center_point.y;

            // 创建带颜色的点（红色）
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 255;  // 红色
            send_point.g = 0;
            send_point.b = 0;
            send_point.a = 255;
            cloud->points.push_back(send_point);

            // 在地图上绘制红方机器人标记（圆形+编号）
            cv::circle(
                Map_clone,
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(0, 0, 200), -1);  // 红色圆形
            cv::putText(
                Map_clone, std::to_string(i + 1),
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
    }

    // 设置发布消息的时间戳（与输入消息一致）
    send_data.header.stamp = msg->header.stamp;
    // 发布解析后的检测结果
    pub_radar->publish(send_data);

    // 转换带颜色的点云为ROS消息并发布
    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "rm_frame";      // 坐标系ID
    cloud_msg.header.stamp = msg->header.stamp;  // 时间戳与输入一致
    pub->publish(cloud_msg);
}
}  // namespace tdt_radar

// 注册ROS 2组件，使节点可被发现和加载
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Resolve)
