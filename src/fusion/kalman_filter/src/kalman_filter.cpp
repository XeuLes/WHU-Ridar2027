// 卡尔曼滤波节点实现文件
// 该文件实现了一个基于卡尔曼滤波的多目标跟踪系统，用于处理雷达点云数据

#include "kalman_filter.h"                    // 卡尔曼滤波器头文件
#include <pcl_conversions/pcl_conversions.h>  // PCL点云与ROS消息转换库
#include "filter_plus.h"                      // 扩展滤波器头文件

namespace tdt_radar {  // 命名空间定义

int KalmanFilter::effective_self_color() const
{
    if (use_referee_identity_color_ && has_referee_identity_color_) {
        return referee_identity_color_;
    }
    if (use_match_info_color_) {
        return normalized_self_color(match_info.self_color);
    }
    return normalized_self_color(self_color_override_);
}

int KalmanFilter::normalized_self_color(int raw_color) const
{
    if (raw_color == 0) {
        return 0;
    }
    if (raw_color == 2 || raw_color == 1) {
        return 2;
    }
    return normalized_self_color(self_color_override_);
}

// 卡尔曼滤波器构造函数
KalmanFilter::KalmanFilter(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("kalman_filter_node", node_options)  // ROS2节点初始化
{
    // 创建订阅器：订阅原始点云数据
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10,  // 话题名称和队列大小
        std::bind(&KalmanFilter::callback, this,
                  std::placeholders::_1));  // 回调函数绑定

    // 创建发布器：发布滤波后的点云
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_kalman", 10);

    // 创建发布器：发布雷达数据到哨兵
    radar_pub_ =
        this->create_publisher<vision_interface::msg::Radar2Sentry>(
            "/radar2sentry", 10);

    // 创建发布器：发布检测结果
    radar_detect_pub_ =
        this->create_publisher<vision_interface::msg::DetectResult>(
            "/kalman_detect", 10);

    // 创建订阅器：订阅检测结果（来自 Resolve 节点的坐标转换后数据）
    sub_detect_ =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "resolve_result",
            rclcpp::SensorDataQoS(),  // 使用传感器数据质量服务
            std::bind(&KalmanFilter::detect_callback, this,
                      std::placeholders::_1));

    // 创建订阅器：订阅激光雷达检测结果
    sub_lidar_ =
        this->create_subscription<vision_interface::msg::RadarWarn>(
            "/lidar_detect", 10,
            std::bind(&KalmanFilter::lidar_callback, this,
                      std::placeholders::_1));

    // 创建订阅器：订阅匹配信息
    sub_match_ =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "/match_info", 10,
            std::bind(&KalmanFilter::match_callback, this,
                      std::placeholders::_1));

    sub_referee_identity_ =
        this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
            "/radar/referee/robot_identity", 10,
            std::bind(&KalmanFilter::referee_identity_callback, this,
                      std::placeholders::_1));
    

    this->declare_parameter<int>("kf_available_timeout_ms", 300);
    this->declare_parameter<bool>("use_match_info_color", true);
    this->declare_parameter<int>("self_color_override", 2);
    this->declare_parameter<bool>("use_referee_identity_color", true);
    kf_available_timeout_ms_ =
        this->get_parameter("kf_available_timeout_ms").as_int();
    use_match_info_color_ =
        this->get_parameter("use_match_info_color").as_bool();
    self_color_override_ =
        normalized_self_color(this->get_parameter("self_color_override").as_int());
    use_referee_identity_color_ =
        this->get_parameter("use_referee_identity_color").as_bool();
    referee_identity_color_ = self_color_override_;
    match_info.self_color = self_color_override_;
    // 输出节点启动日志信息
    RCLCPP_INFO(
        this->get_logger(),
        "Kalman_filter_Node has been started. use_match_info_color=%s self_color_override=%d",
        use_match_info_color_ ? "true" : "false",
        self_color_override_);
}

// 匹配信息回调函数
void KalmanFilter::match_callback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    this->match_info = *msg;  // 保存匹配信息
    if (use_match_info_color_) {
        this->match_info.self_color = normalized_self_color(msg->self_color);
    } else {
        this->match_info.self_color = self_color_override_;
    }
    // RCLCPP_INFO(this->get_logger(), "Match_info_callback");  // 日志输出
}

void KalmanFilter::referee_identity_callback(
    const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
{
    if (!msg || !msg->valid) {
        return;
    }
    has_referee_identity_color_ = true;
    referee_identity_color_ = normalized_self_color(msg->self_color);
}

// 检测结果回调函数
void KalmanFilter::detect_callback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    static int callback_count = 0;
    callback_count++;

    rclcpp::Time time = msg->header.stamp;
    if (time.nanoseconds() == 0) {
        time = this->now();
    }

    // 判断当前KF是否可用（近期收到过雷达点云）
    bool kf_active_now = false;
    if (kf_available_ &&
        last_lidar_cluster_stamp_.nanoseconds() != 0 &&
        !KFs.empty()) {
        const double dt_ms =
            std::fabs((time - last_lidar_cluster_stamp_).seconds()) * 1000.0;
        kf_active_now = (dt_ms <= static_cast<double>(kf_available_timeout_ms_));
    }

    // 如果KF当前可用，则不再直传resolve_result到 /kalman_detect
    if (!kf_active_now) {
        vision_interface::msg::DetectResult forward_msg = *msg;

        // 保持原逻辑：根据self_color镜像坐标
        if (effective_self_color() == 0) {
            for (int i = 0; i < 7; i++) {
                if (forward_msg.blue_x[i] != 0 || forward_msg.blue_y[i] != 0) {
                    forward_msg.blue_x[i] = 28 - forward_msg.blue_x[i];
                    forward_msg.blue_y[i] = 15 - forward_msg.blue_y[i];
                }
                if (forward_msg.red_x[i] != 0 || forward_msg.red_y[i] != 0) {
                    forward_msg.red_x[i] = 28 - forward_msg.red_x[i];
                    forward_msg.red_y[i] = 15 - forward_msg.red_y[i];
                }
            }
        }

        radar_detect_pub_->publish(forward_msg);
    }

    // 保留原camera_match逻辑（用于KF融合）
    for (int i = 0; i < 7; i++) {
        pcl::PointXY red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        if (red_point.x == 0 || red_point.y == 0)
            continue;

        for (auto& kf : KFs) {
            kf.camera_match_and_update(time, red_point, 2, i);
        }

        pcl::PointXY blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        if (blue_point.x == 0 || blue_point.y == 0)
            continue;

        for (auto& kf : KFs) {
            kf.camera_match_and_update(time, blue_point, 0, i);
        }
    }
}
// 激光雷达检测回调函数
void KalmanFilter::lidar_callback(
    const vision_interface::msg::RadarWarn::SharedPtr msg)
{
    this->lidar_detect = *msg;  // 保存激光雷达检测结果
}

bool KalmanFilter::is_kf_available_now(const rclcpp::Time& now_time) const
{
    if (!kf_available_) {
        return false;
    }

    if (last_lidar_cluster_stamp_.nanoseconds() == 0) {
        return false;
    }

    const double dt_ms =
        std::fabs((now_time - last_lidar_cluster_stamp_).seconds()) * 1000.0;

    return dt_ms <= static_cast<double>(kf_available_timeout_ms_);
}

// 主回调函数：处理点云数据
void KalmanFilter::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rclcpp::Time time = msg->header.stamp;  // 获取消息时间戳
    if (time.nanoseconds() == 0) {
        time = this->now();
    }

    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
        return;
    }

    last_lidar_cluster_stamp_ = time;
    kf_available_ = true;


    auto         now_time =
        std::chrono::steady_clock::now();  // 记录当前时间用于性能测量

    // 将ROS点云消息转换为PCL点云格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy(
        new pcl::PointCloud<pcl::PointXY>);
    pcl::fromROSMsg(*msg, *cloud);  // ROS消息转PCL

    // 将3D点云转换为2D点云（忽略z坐标）
    for (auto point : cloud->points) {
        pcl::PointXY point_xy;
        point_xy.x = point.x;
        point_xy.y = point.y;
        cloud_xy->points.push_back(point_xy);
    }

    if (cloud_xy->points.size() == 0)  // 如果点云为空则返回
        return;

    // 更新所有卡尔曼滤波器的预测点
    for (auto& kf : KFs) {
        kf.update_predict_point();  // 更新预测位置
        kf.has_updated = false;     // 重置更新标志
    }

    // 点云数据关联与跟踪
    for (auto point : cloud_xy->points) {
        std::vector<int> match_kf_indexs;  // 存储匹配的滤波器索引

        // 查找与当前点匹配的卡尔曼滤波器
        for (int i = 0; i < this->KFs.size(); i++) {
            if (KFs[i].match(point)) {  // 检查点是否匹配滤波器
                match_kf_indexs.push_back(i);  // 记录匹配的滤波器索引
            }
        }

        if (match_kf_indexs.size() == 0) {
            // 没有匹配的滤波器，创建新的卡尔曼滤波器
            Kalman_filter_plus kf(point, time);
            KFs.push_back(kf);
        } else if (match_kf_indexs.size() == 1) {
            // 只有一个匹配的滤波器，直接更新
            KFs[match_kf_indexs[0]].update_lidar(point, time);
        } else {
            // 多个滤波器匹配，选择距离最近的滤波器
            float min_distance = 1000000;  // 初始化最小距离
            int   min_index = 0;           // 最小距离对应的索引

            // 寻找距离最小的滤波器
            for (auto index : match_kf_indexs) {
                float distance =
                    KFs[index].Distance(KFs[index].predict_point, point);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = index;
                }
            }
            // 更新距离最近的滤波器
            KFs[min_index].update_lidar(point, time);
        }
    }

    // 创建带颜色的滤波后点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    // 清理长时间未更新的滤波器并生成可视化点云
    for (int i = KFs.size() - 1; i >= 0; i--) {
        if ((KFs[i].last_time) > 1.5) {  // 如果超过1.5秒未更新
            KFs.erase(KFs.begin() + i);  // 删除该滤波器
        } else {
            // 创建带颜色的点
            pcl::PointXYZRGB point;
            point.x = KFs[i].predict_point.x;  // 预测x坐标
            point.y = KFs[i].predict_point.y;  // 预测y坐标
            point.z = 1.5;                     // 固定高度

            // 根据目标颜色设置点云颜色
            int color = KFs[i].get_color();
            switch (color) {
            case 0:             // 蓝色目标
                point.b = 255;  // 设置蓝色分量
                break;

            case 2:             // 红色目标
                point.r = 255;  // 设置红色分量
                break;

            default:                        // 其他颜色
                point.r = KFs[i].color[0];  // 自定义红色分量
                point.g = KFs[i].color[1];  // 自定义绿色分量
                point.b = KFs[i].color[2];  // 自定义蓝色分量
                break;
            }
            cloud_filtered->points.push_back(point);  // 添加点到点云
        }
    }

    // 发布滤波后的点云
    cloud_filtered->header.frame_id = "rm_frame";  // 设置坐标系
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud_filtered, output);  // PCL点云转ROS消息
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;  // 保持时间戳一致
    pub_->publish(output);                    // 发布消息

    // 性能测量：计算处理耗时
    auto  end_time = std::chrono::steady_clock::now();
    float dur_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - now_time)
                         .count();

    // 创建检测结果消息
    vision_interface::msg::DetectResult detect_msg;

    // 填充检测结果消息
    for (auto kf : KFs) {
        // 【临时修复】暂时注释掉阻塞条件，让所有卡尔曼滤波器的数据都能发布
        // 原问题：只发布有相机检测历史(detect_history)的目标，会导致纯雷达目标被丢弃
        // if (kf.detect_history.size() == 0)  // 跳过无检测历史的滤波器
        //     continue;

        if (kf.get_color() == 0) {         // 蓝色目标
            int number = kf.get_number();  // 获取目标编号
            detect_msg.blue_x[number] = kf.predict_point.x;  // 设置x坐标
            detect_msg.blue_y[number] = kf.predict_point.y;  // 设置y坐标
        }
        if (kf.get_color() == 2) {         // 红色目标
            int number = kf.get_number();  // 获取目标编号
            detect_msg.red_x[number] = kf.predict_point.x;  // 设置x坐标
            detect_msg.red_y[number] = kf.predict_point.y;  // 设置y坐标
        }
    }

    // 坐标转换：根据己方颜色调整坐标系
    if (effective_self_color() == 0) {  // 如果己方是蓝色
        // 【修复】循环7次以支持UAV（索引6）
        for (int i = 0; i < 7; i++) {
            // 蓝色方坐标镜像转换
            if (detect_msg.blue_x[i] != 0 && detect_msg.blue_y[i] != 0) {
                detect_msg.blue_x[i] = 28 - detect_msg.blue_x[i];
                detect_msg.blue_y[i] = 15 - detect_msg.blue_y[i];
            }
            // 红色方坐标镜像转换
            if (detect_msg.red_x[i] != 0 && detect_msg.red_y[i] != 0) {
                detect_msg.red_x[i] = 28 - detect_msg.red_x[i];
                detect_msg.red_y[i] = 15 - detect_msg.red_y[i];
            }
        }
    }

    // 【数据流追踪】记录发布的融合数据
    bool has_kalman_data = false;
    std::string kalman_log = "[KalmanFilter发布] /kalman_detect: ";
    
    for (int i = 0; i < 7; i++) {
        if (detect_msg.blue_x[i] != 0 || detect_msg.blue_y[i] != 0) {
            kalman_log += "B" + std::to_string(i+1) + "(";
            kalman_log += std::to_string((int)detect_msg.blue_x[i]) + ",";
            kalman_log += std::to_string((int)detect_msg.blue_y[i]) + ") ";
            has_kalman_data = true;
        }
        if (detect_msg.red_x[i] != 0 || detect_msg.red_y[i] != 0) {
            kalman_log += "R" + std::to_string(i+1) + "(";
            kalman_log += std::to_string((int)detect_msg.red_x[i]) + ",";
            kalman_log += std::to_string((int)detect_msg.red_y[i]) + ") ";
            has_kalman_data = true;
        }
    }
    
    if (has_kalman_data) {
        RCLCPP_WARN(this->get_logger(), "%s [KF数量=%zu]", kalman_log.c_str(), KFs.size());
    }
    
    radar_detect_pub_->publish(detect_msg);  // 发布检测结果

    // 创建发送给哨兵的雷达消息
    vision_interface::msg::Radar2Sentry radar_msg;

    // 如果己方是蓝色，发送敌方（红色）位置信息给哨兵
    if (effective_self_color() == 0) {
        for (int i = 0; i < 6; i++) {
            radar_msg.radar_enemy_x[i] = detect_msg.red_x[i];  // 敌方x坐标
            radar_msg.radar_enemy_y[i] = detect_msg.red_y[i];  // 敌方y坐标
        }
        radar_pub_->publish(radar_msg);  // 发布雷达消息
    }
}
}  // namespace tdt_radar

// ROS2组件注册宏
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::KalmanFilter)
