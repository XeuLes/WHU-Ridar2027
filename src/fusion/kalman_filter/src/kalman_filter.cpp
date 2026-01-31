// 卡尔曼滤波节点实现文件
// 该文件实现了一个基于卡尔曼滤波的多目标跟踪系统，用于处理雷达点云数据

#include "kalman_filter.h"                    // 卡尔曼滤波器头文件
#include <pcl_conversions/pcl_conversions.h>  // PCL点云与ROS消息转换库
#include "filter_plus.h"                      // 扩展滤波器头文件

namespace tdt_radar {  // 命名空间定义

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

    // 输出节点启动日志信息
    RCLCPP_INFO(this->get_logger(), "Kalman_filter_Node has been started.");
}

// 匹配信息回调函数
void KalmanFilter::match_callback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    this->match_info = *msg;  // 保存匹配信息
    // RCLCPP_INFO(this->get_logger(), "Match_info_callback");  // 日志输出
}

// 检测结果回调函数
void KalmanFilter::detect_callback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    static int callback_count = 0;
    callback_count++;
    
    rclcpp::Time time = msg->header.stamp;  // 获取消息时间戳
    
    // 【数据流追踪】记录接收到的数据
    bool has_vehicle_data = false;
    std::string recv_log = "[KalmanFilter接收] resolve_result (第" + std::to_string(callback_count) + "次): ";
    
    for (int i = 0; i < 7; i++) {
        if (msg->blue_x[i] != 0 || msg->blue_y[i] != 0) {
            recv_log += "B" + std::to_string(i+1) + "(";
            recv_log += std::to_string((int)msg->blue_x[i]) + ",";
            recv_log += std::to_string((int)msg->blue_y[i]) + ") ";
            has_vehicle_data = true;
        }
        if (msg->red_x[i] != 0 || msg->red_y[i] != 0) {
            recv_log += "R" + std::to_string(i+1) + "(";
            recv_log += std::to_string((int)msg->red_x[i]) + ",";
            recv_log += std::to_string((int)msg->red_y[i]) + ") ";
            has_vehicle_data = true;
        }
    }
    
    if (has_vehicle_data) {
        RCLCPP_WARN(this->get_logger(), "%s", recv_log.c_str());
    }
    
    // 【临时修复】直接转发 Resolve 数据到 /kalman_detect，不依赖雷达融合
    // 原问题：如果没有雷达点云 (/livox/lidar_cluster)，主callback()不执行，数据流中断
    vision_interface::msg::DetectResult forward_msg = *msg;
    
    // 保持原逻辑：根据self_color镜像坐标
    if (match_info.self_color == 0) {
        // 【修复】循环7次以支持UAV（索引6），允许单个坐标为0的情况
        for (int i = 0; i < 7; i++) {
            // 改用OR条件，允许x或y其中一个为0的情况
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
    
    // 【调试日志已注释】避免卡顿
    // std::string forward_log = "[KalmanFilter发布] /kalman_detect: ";
    // bool has_forward_data = false;
    // for (int i = 0; i < 7; i++) {
    //     if (forward_msg.blue_x[i] != 0 || forward_msg.blue_y[i] != 0) {
    //         forward_log += "B" + std::to_string(i+1) + "(";
    //         forward_log += std::to_string((int)forward_msg.blue_x[i]) + ",";
    //         forward_log += std::to_string((int)forward_msg.blue_y[i]) + ") ";
    //         has_forward_data = true;
    //     }
    //     if (forward_msg.red_x[i] != 0 || forward_msg.red_y[i] != 0) {
    //         forward_log += "R" + std::to_string(i+1) + "(";
    //         forward_log += std::to_string((int)forward_msg.red_x[i]) + ",";
    //         forward_log += std::to_string((int)forward_msg.red_y[i]) + ") ";
    //         has_forward_data = true;
    //     }
    // }
    // 
    // if (has_forward_data) {
    //     RCLCPP_WARN(this->get_logger(), "%s [直接转发模式]", forward_log.c_str());
    // }
    
    radar_detect_pub_->publish(forward_msg);

    // 保留原camera_match逻辑（用于雷达融合，暂时不影响）
    // 【修复】循环7次以支持UAV的camera_match
    for (int i = 0; i < 7; i++) {
        pcl::PointXY red_point;                    // 创建2D点
        red_point.x = msg->red_x[i];               // 设置x坐标
        red_point.y = msg->red_y[i];               // 设置y坐标
        if (red_point.x == 0 || red_point.y == 0)  // 跳过无效点
            continue;

        // 对所有卡尔曼滤波器进行相机匹配
        for (auto& kf : KFs) {
            kf.camera_match(time, red_point, 2, i);  // 颜色2代表红色
        }

        // 处理蓝色方检测点（颜色编号0）
        pcl::PointXY blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        if (blue_point.x == 0 || blue_point.y == 0)
            continue;

        // 对所有卡尔曼滤波器进行相机匹配
        for (auto& kf : KFs) {
            kf.camera_match(time, blue_point, 0, i);  // 颜色0代表蓝色
        }
    }
}

// 激光雷达检测回调函数
void KalmanFilter::lidar_callback(
    const vision_interface::msg::RadarWarn::SharedPtr msg)
{
    this->lidar_detect = *msg;  // 保存激光雷达检测结果
}

// 主回调函数：处理点云数据
void KalmanFilter::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rclcpp::Time time = msg->header.stamp;  // 获取消息时间戳
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
            KFs[match_kf_indexs[0]].update(point, time);
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
            KFs[min_index].update(point, time);
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
    if (match_info.self_color == 0) {  // 如果己方是蓝色
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
    if (match_info.self_color == 0) {
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