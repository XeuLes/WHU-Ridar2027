#include "kalman_filter.h"  // 卡尔曼滤波器类的头文件
#include <pcl_conversions/pcl_conversions.h>  // PCL与ROS消息格式转换的头文件
#include "filter_plus.h"  // 扩展滤波器类的头文件

namespace tdt_radar {  // 自定义命名空间，避免命名冲突

// 卡尔曼滤波器节点类，继承自rclcpp::Node
class KalmanFilter : public rclcpp::Node {
public:
    // 构造函数，初始化节点名称和选项
    KalmanFilter(const rclcpp::NodeOptions& node_options)
        : rclcpp::Node("kalman_filter_node", node_options)
    {
        // 创建订阅者，订阅激光雷达聚类结果话题（PointCloud2类型）
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar_cluster", 10,  // 话题名和队列长度
            std::bind(&KalmanFilter::callback, this,
                      std::placeholders::_1));  // 绑定回调函数

        // 创建发布者，发布卡尔曼滤波后的点云话题
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar_kalman", 10);  // 话题名和队列长度

        // 创建发布者，发布雷达到哨兵的消息
        radar_pub_ =
            this->create_publisher<vision_interface::msg::Radar2Sentry>(
                "/radar2sentry", 10);

        // 创建发布者，发布卡尔曼滤波后的检测结果
        radar_detect_pub_ =
            this->create_publisher<vision_interface::msg::DetectResult>(
                "/kalman_detect", 10);

        // 创建订阅者，订阅解析后的检测结果（传感器数据QoS）
        sub_detect_ =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/resolve_result", rclcpp::SensorDataQoS(),
                std::bind(&KalmanFilter::detect_callback, this,
                          std::placeholders::_1));

        // 创建订阅者，订阅激光雷达检测警告消息
        sub_lidar_ =
            this->create_subscription<vision_interface::msg::RadarWarn>(
                "/lidar_detect", 10,
                std::bind(&KalmanFilter::lidar_callback, this,
                          std::placeholders::_1));

        // 创建订阅者，订阅匹配信息消息
        sub_match_ =
            this->create_subscription<vision_interface::msg::MatchInfo>(
                "/match_info", 10,
                std::bind(&KalmanFilter::match_callback, this,
                          std::placeholders::_1));

        // 打印节点启动日志
        RCLCPP_INFO(this->get_logger(),
                    "Kalman_filter_Node has been started.");
    }

private:
    // 匹配信息回调函数：接收匹配信息并存储
    void KalmanFilter::match_callback(
        const vision_interface::msg::MatchInfo::SharedPtr msg)
    {
        this->match_info = *msg;  // 保存匹配信息
        RCLCPP_INFO(this->get_logger(),
                    "Match_info_callback");  // 打印回调日志
    }

    // 检测结果回调函数：处理检测到的红蓝目标，与卡尔曼滤波器匹配
    void KalmanFilter::detect_callback(
        const vision_interface::msg::DetectResult::SharedPtr msg)
    {
        rclcpp::Time time = msg->header.stamp;  // 获取消息时间戳

        // 遍历6个可能的目标（假设最多6个目标）
        for (int i = 0; i < 6; i++) {
            // 处理红色目标点
            pcl::PointXY red_point;
            red_point.x = msg->red_x[i];
            red_point.y = msg->red_y[i];
            if (red_point.x == 0 ||
                red_point.y == 0)  // 过滤无效点（坐标为0）
                continue;
            // 与所有卡尔曼滤波器进行相机匹配（颜色2代表红色）
            for (auto& kf : KFs) {
                kf.camera_match(time, red_point, 2, i);
            }

            // 处理蓝色目标点
            pcl::PointXY blue_point;
            blue_point.x = msg->blue_x[i];
            blue_point.y = msg->blue_y[i];
            if (blue_point.x == 0 || blue_point.y == 0)  // 过滤无效点
                continue;
            // 与所有卡尔曼滤波器进行相机匹配（颜色0代表蓝色）
            for (auto& kf : KFs) {
                kf.camera_match(time, blue_point, 0, i);
            }
        }
    }

    // 激光雷达检测回调函数：保存激光雷达检测警告信息
    void KalmanFilter::lidar_callback(
        const vision_interface::msg::RadarWarn::SharedPtr msg)
    {
        this->lidar_detect = *msg;  // 存储激光雷达检测结果
    }

    // 主回调函数：处理激光雷达聚类点云，执行卡尔曼滤波的预测与更新
    void KalmanFilter::callback(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        rclcpp::Time time = msg->header.stamp;  // 获取点云时间戳
        auto         now_time = std::chrono::steady_clock::
            now();  // 记录当前时间（用于计算耗时）

        // 转换ROS点云消息到PCL点云（XYZ类型）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        // 提取XY坐标（忽略Z），用于2D卡尔曼滤波
        pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy(
            new pcl::PointCloud<pcl::PointXY>);
        pcl::fromROSMsg(*msg, *cloud);  // ROS->PCL格式转换

        // 提取XY坐标
        for (auto point : cloud->points) {
            pcl::PointXY point_xy;
            point_xy.x = point.x;
            point_xy.y = point.y;
            cloud_xy->points.push_back(point_xy);
        }

        if (cloud_xy->points.size() == 0)  // 无点云数据则直接返回
            return;

        // 对所有卡尔曼滤波器执行预测，并标记为未更新
        for (auto& kf : KFs) {
            kf.update_predict_point();  // 更新预测点
            kf.has_updated = false;     // 标记未更新（等待匹配）
        }

        // 遍历每个点，与卡尔曼滤波器匹配并更新
        for (auto point : cloud_xy->points) {
            std::vector<int> match_kf_indexs;  // 匹配的滤波器索引列表

            // 寻找所有匹配当前点的滤波器
            for (int i = 0; i < this->KFs.size(); i++) {
                if (KFs[i].match(point)) {  // 调用匹配函数
                    match_kf_indexs.push_back(i);
                }
            }

            // 无匹配的滤波器：创建新的卡尔曼滤波器
            if (match_kf_indexs.size() == 0) {
                Kalman_filter_plus kf(point, time);  // 初始化新滤波器
                KFs.push_back(kf);
            }
            // 唯一匹配的滤波器：直接更新
            else if (match_kf_indexs.size() == 1) {
                KFs[match_kf_indexs[0]].update(point, time);
            }
            // 多个匹配的滤波器：选择预测点与当前点距离最近的更新
            else {
                float min_distance = 1000000;  // 初始化最小距离
                int   min_index = 0;
                for (auto index : match_kf_indexs) {
                    // 计算预测点与当前点的距离
                    float distance = KFs[index].Distance(
                        KFs[index].predict_point, point);
                    if (distance < min_distance) {
                        min_distance = distance;
                        min_index = index;
                    }
                }
                KFs[min_index].update(point, time);  // 更新最近的滤波器
            }
        }

        // 生成滤波后的点云（带颜色）
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZRGB>);

        // 遍历所有滤波器，移除长时间未更新的目标，保留有效目标并添加到点云
        for (int i = KFs.size() - 1; i >= 0; i--) {
            // 移除超过1.5秒未更新的滤波器（目标消失）
            if ((KFs[i].last_time) > 1.5) {
                KFs.erase(KFs.begin() + i);
            } else {
                // 构造带颜色的点（Z固定为1.5）
                pcl::PointXYZRGB point;
                point.x = KFs[i].predict_point.x;
                point.y = KFs[i].predict_point.y;
                point.z = 1.5;

                // 根据目标颜色设置点云颜色（0:蓝色, 2:红色,
                // 其他:自定义颜色）
                int color = KFs[i].get_color();
                switch (color) {
                case 0:
                    point.b = 255;  // 蓝色
                    break;
                case 2:
                    point.r = 255;  // 红色
                    break;
                default:
                    point.r = KFs[i].color[0];  // 自定义RGB
                    point.g = KFs[i].color[1];
                    point.b = KFs[i].color[2];
                    break;
                }
                cloud_filtered->points.push_back(point);
            }
        }

        // 发布滤波后的点云
        cloud_filtered->header.frame_id = "rm_frame";  // 设置坐标系
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);  // PCL->ROS格式转换
        output.header.frame_id = "rm_frame";
        output.header.stamp = msg->header.stamp;  // 保持原时间戳
        pub_->publish(output);

        // 计算回调处理耗时（毫秒）
        auto  end_time = std::chrono::steady_clock::now();
        float dur_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  now_time)
                .count();

        // 构造卡尔曼滤波后的检测结果消息
        vision_interface::msg::DetectResult detect_msg;
        for (auto kf : KFs) {
            if (kf.detect_history.size() == 0)  // 无检测历史则跳过
                continue;

            // 填充红蓝目标坐标（根据颜色和编号）
            if (kf.get_color() == 0) {  // 蓝色目标
                int number = kf.get_number();
                detect_msg.blue_x[number] = kf.predict_point.x;
                detect_msg.blue_y[number] = kf.predict_point.y;
            }
            if (kf.get_color() == 2) {  // 红色目标
                int number = kf.get_number();
                detect_msg.red_x[number] = kf.predict_point.x;
                detect_msg.red_y[number] = kf.predict_point.y;
            }
        }

        // 根据自身颜色进行坐标转换（假设0为蓝色方，坐标翻转）
        if (match_info.self_color == 0) {
            for (int i = 0; i < 6; i++) {
                if (detect_msg.blue_x[i] != 0 &&
                    detect_msg.blue_y[i] != 0) {
                    detect_msg.blue_x[i] =
                        28 - detect_msg.blue_x[i];  // X坐标翻转
                    detect_msg.blue_y[i] =
                        15 - detect_msg.blue_y[i];  // Y坐标翻转
                }
                if (detect_msg.red_x[i] != 0 && detect_msg.red_y[i] != 0) {
                    detect_msg.red_x[i] = 28 - detect_msg.red_x[i];
                    detect_msg.red_y[i] = 15 - detect_msg.red_y[i];
                }
            }
        }

        // 发布卡尔曼滤波后的检测结果
        radar_detect_pub_->publish(detect_msg);

        // 构造雷达到哨兵的消息（敌方目标坐标）
        vision_interface::msg::Radar2Sentry radar_msg;
        if (match_info.self_color == 0) {  // 蓝色方：敌方为红色
            for (int i = 0; i < 6; i++) {
                radar_msg.radar_enemy_x[i] = detect_msg.red_x[i];
                radar_msg.radar_enemy_y[i] = detect_msg.red_y[i];
            }
            radar_pub_->publish(radar_msg);  // 发布敌方坐标
        }
    }

    // 成员变量定义（原代码中省略，此处补充注释说明）
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        sub_;  // 激光雷达聚类订阅者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pub_;  // 滤波后点云发布者
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar_pub_;  // 雷达到哨兵发布者
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        radar_detect_pub_;  // 滤波检测结果发布者
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        sub_detect_;  // 解析结果订阅者
    rclcpp::Subscription<vision_interface::msg::RadarWarn>::SharedPtr
        sub_lidar_;  // 激光雷达检测订阅者
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
                                     sub_match_;  // 匹配信息订阅者
    vision_interface::msg::MatchInfo match_info;  // 存储匹配信息
    vision_interface::msg::RadarWarn lidar_detect;  // 存储激光雷达检测结果
    std::vector<Kalman_filter_plus> KFs;  // 卡尔曼滤波器实例集合
};

}  // namespace tdt_radar

// 注册ROS 2组件，使节点可被发现和加载
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::KalmanFilter)
