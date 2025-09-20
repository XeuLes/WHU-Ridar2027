#include "dynamic_cloud.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
namespace tdt_radar {  // 自定义命名空间，避免命名冲突
DynamicCloud::DynamicCloud(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node(
          "dynamic_cloud_node",
          node_options),  // 初始化ROS 2节点，名称为"dynamic_cloud_node"
      tf_buffer_(this->get_clock()),  // 初始化TF缓冲区，用于获取坐标变换
      tf_listener_(tf_buffer_)  // 初始化TF监听器，用于监听坐标系变换
{
    RCLCPP_INFO(this->get_logger(),
                "Dynamic_cloud Node start");  // 打印节点启动日志

    // 加载地图点云（PCD文件）
    auto temp_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("config/RM2025.pcd",
                                            *temp_cloud) == -1) {
        PCL_ERROR("Couldn't read file map.pcd \n");  // 加载失败时打印错误
    }

    // // 对地图点云进行体素下采样（减少点数量，提高后续处理效率）
    // pcl::VoxelGrid<pcl::PointXYZ> sor;  // 体素滤波器
    // sor.setInputCloud(temp_cloud);      // 设置输入点云
    // sor.setLeafSize(0.1f, 0.1f, 0.1f);  // 体素大小（0.1m×0.1m×0.1m）
    // auto result = pcl::PointCloud<pcl::PointXYZ>::Ptr(
    //     new pcl::PointCloud<pcl::PointXYZ>);
    // sor.filter(*result);  // 执行下采样
    // map_cloud = result;   // 保存下采样后的地图点云

    map_cloud = temp_cloud;

    // 构建KD-Tree（用于后续快速最近邻搜索）
    kd_Tree.setInputCloud(map_cloud);

    // 初始化订阅者和发布者
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10,  // 订阅激光雷达原始点云话题
        std::bind(&DynamicCloud::callback, this,
                  std::placeholders::_1));  // 绑定回调函数

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10);  // 发布动态点云结果

    other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_other", 10);  // 发布其他感兴趣区域的点云

    detect_pub_ = this->create_publisher<vision_interface::msg::RadarWarn>(
        "/lidar_detect", 10);  // 发布检测警告信息（如飞镖、飞机等）

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                "Dynamic Cloud Launch!");  // 打印启动完成日志
}

void DynamicCloud::GetDynamicCloud(
    pcl::PointCloud<pcl::PointXYZ>&
        input_cloud,  // 输入点云（待检测动态点）
    pcl::PointCloud<pcl::PointXYZ>& output_cloud,  // 输出点云（动态点结果）
    float threshold,  // 距离阈值（超过此值视为动态点）
    int   thread_num)   // 线程数（多线程加速处理）
{
    int K = 1;  // 搜索最近邻的数量（只找最近的1个点）
    std::vector<std::thread>                    threads;  // 线程池
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(
        thread_num);  // 每个线程的局部点云结果

    auto start = std::chrono::system_clock::now();  // 记录开始时间
    int  cloud_size = input_cloud.points.size();    // 输入点云的点数
    int  step = cloud_size / thread_num;  // 每个线程处理的点数

    // 多线程处理：将点云分割到不同线程，并行计算
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread([i, step, cloud_size, &clouds,
                                       &input_cloud, this, threshold, K]() {
            // 每个线程处理自己负责的点范围
            for (int j = i * step; j < (i + 1) * step; j++) {
                std::vector<int> pointIdxNKNSearch(
                    K);  // 存储最近邻点的索引
                std::vector<float> pointNKNSquaredDistance(
                    K);  // 存储最近邻点的平方距离

                // 搜索输入点在地图点云中的最近邻
                if (kd_Tree.nearestKSearch(input_cloud.points[j], K,
                                           pointIdxNKNSearch,
                                           pointNKNSquaredDistance) > 0) {
                    // 若最近距离的平方大于阈值（即距离大于threshold），视为动态点
                    if (pointNKNSquaredDistance[0] > threshold) {
                        clouds[i].points.push_back(input_cloud.points[j]);
                    }
                }
            }
        }));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 合并所有线程的结果到输出点云
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }

    auto end = std::chrono::system_clock::
        now();  // 记录结束时间（未打印，可用于调试耗时）
}

void TransformCloud(
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,   // 输入点云
    pcl::PointCloud<pcl::PointXYZ>& output_cloud,  // 变换后的点云
    geometry_msgs::msg::TransformStamped transform_stamped,  // TF变换信息
    int                                  thread_num)                                          // 线程数
{
    std::vector<std::thread>                    threads;  // 线程池
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(
        thread_num);  // 每个线程的局部结果

    auto start = std::chrono::system_clock::now();  // 记录开始时间
    int  cloud_size = input_cloud.points.size();    // 输入点云点数
    int  step = cloud_size / thread_num;  // 每个线程处理的点数

    // 构建坐标变换矩阵（从TF信息中提取平移和旋转）
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;  // 平移分量
    Eigen::Quaternionf rotation(
        transform_stamped.transform.rotation.w,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z);  // 旋转分量（四元数）
    transform.rotate(rotation);                   // 组合旋转

    // 多线程执行坐标变换
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(
            [i, step, cloud_size, &clouds, &input_cloud, transform]() {
                // 每个线程处理自己的点范围
                for (int j = i * step; j < (i + 1) * step && j < cloud_size;
                     j++) {
                    pcl::PointXYZ point = input_cloud.points[j];  // 当前点
                    Eigen::Vector3f point_vec(point.x, point.y,
                                              point.z);  // 转换为Eigen向量
                    Eigen::Vector3f point_out =
                        transform * point_vec;  // 应用变换

                    // 保存变换后的点
                    pcl::PointXYZ transformed_point;
                    transformed_point.x = point_out.x();
                    transformed_point.y = point_out.y();
                    transformed_point.z = point_out.z();
                    clouds[i].points.push_back(transformed_point);
                }
            }));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }

    // 打印变换耗时（毫秒）
    auto end = std::chrono::system_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "transform time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1000.0);
}

void DynamicCloud::callback(const sensor_msgs::msg::PointCloud2::SharedPtr
                                msg)  // 接收的激光雷达原始点云消息
{
    vision_interface::msg::RadarWarn
        lidar_detect;  // 声明检测警告消息（自定义类型）

    // 定义区域过滤器（Lambda表达式，用于检测特定区域的点）
    auto dart_cloud_filter = [](pcl::PointXYZ& point) {  // 飞镖检测区域
        return (point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
               (point.y > 3.925 && point.y < 4.525) &&
               (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722);
    };

    auto fly_safe_filter =
        [](pcl::PointXYZ& point) {  // 飞机安全区域（起飞区）
            return (point.x > 28 - 2.775 && point.x < 27.5) &&
                   (point.y > 0.2 && point.y < 2.2) &&
                   (point.z > 1.7 && point.z < 3);
        };

    auto fly_warn_filter =
        [](pcl::PointXYZ& point) {  // 飞机警告区域（半场）
            return (point.x > 19.83 && point.x < 28 - 2.7) &&
                   (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
                   (point.z > 1.7 && point.z < 3);
        };

    auto fly_alarm_filter =
        [](pcl::PointXYZ& point) {  // 飞机警报区域（中场）
            return (point.x > 13 && point.x < 20.5) &&
                   (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
                   (point.z > 1.7 && point.z < 3);
        };

    // 1.
    // 将ROS点云消息（sensor_msgs::PointCloud2）转换为PCL点云（pcl::PointXYZ）
    pcl::PointCloud<pcl::PointXYZ> receive_cloud;
    pcl::fromROSMsg(*msg, receive_cloud);

    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();  // 记录处理开始时间

    // 2. 获取坐标变换（从激光雷达坐标系到rm_frame地图坐标系）
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        // 查找变换（时间戳为0表示最新变换）
        transform_stamped = tf_buffer_.lookupTransform(
            "rm_frame", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (tf2::TransformException& ex) {  // 处理变换获取失败的异常
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Transform error: %s",
                     ex.what());
        return;
    }

    // 3. 将原始点云转换到rm_frame坐标系
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    Eigen::Affine3f                transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);
    pcl::transformPointCloud(receive_cloud, transformed_cloud,
                             transform);  // 执行变换

    // 4. 过滤感兴趣区域（ROI）：保留有效场景内的点，去除噪声和无关区域
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;  // 主场景点云
    pcl::PointCloud<pcl::PointXYZ>
        other_filtered_cloud;  // 其他感兴趣区域点云
    for (size_t i = 0; i < transformed_cloud.size(); i++) {
        auto& point = transformed_cloud.points[i];
        // 过滤条件：去除超出场景范围、地面以下或特定干扰区域的点
        if (point.x < 3 || point.x > 28 || point.y < 0 || point.y > 15 ||
            point.z < 0 || point.z > 1.4 ||
            (point.y > 0 && point.y < 5 && point.x > 25) ||
            ((21.5 - 2.9 / sqrt(2)) < (point.x + point.y) &&
             (point.x + point.y) < (21.5 + 2.9 / sqrt(2)) &&
             (-6.5 - 0.9 / sqrt(2)) < (point.y - point.x) &&
             (point.y - point.x) < (-6.5 + 0.9 / sqrt(2)))) {
            // 若点在飞镖或飞机检测区域，则保存到other_filtered_cloud
            if ((point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
                    (point.y > 3.925 && point.y < 4.525) &&
                    (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722) ||
                (point.x > 13 && point.x < 27.5) &&
                    (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
                    (point.z > 1.7 && point.z < 3)) {
                other_filtered_cloud.push_back(point);
            }
            continue;  // 跳过无效点
        }
        filtered_cloud.push_back(point);  // 保留有效点
    }

    // 5. 提取动态点云（调用GetDynamicCloud，阈值0.1m，12线程）
    pcl::PointCloud<pcl::PointXYZ> dynamic_pointcloud;
    GetDynamicCloud(filtered_cloud, dynamic_pointcloud, 0.1, 12);

    // 6. 累积多帧动态点云（平滑结果，减少噪声）
    if (accumulate_count < accumulate_time) {  // 累积帧数不足时（默认3帧）
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
        accumulate_count++;
    } else {  // 累积帧数足够时，滑动窗口更新
        accumulated_clouds_.erase(accumulated_clouds_.begin());
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.erase(other_accumulated_clouds_.begin());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
    }

    // 合并累积的点云
    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;
    for (auto it = accumulated_clouds_.begin();
         it != accumulated_clouds_.end(); ++it) {
        accumulated_cloud += **it;
    }
    pcl::PointCloud<pcl::PointXYZ> other_accumulated_cloud;
    for (auto it = other_accumulated_clouds_.begin();
         it != other_accumulated_clouds_.end(); ++it) {
        other_accumulated_cloud += **it;
    }

    // 7. 发布处理后的点云
    sensor_msgs::msg::PointCloud2 output;
    accumulated_cloud.header.frame_id = "rm_frame";
    other_accumulated_cloud.header.frame_id = "rm_frame";
    pcl::toROSMsg(accumulated_cloud, output);  // PCL点云转ROS消息
    output.header.frame_id = "rm_frame";       // 设置坐标系
    output.header.stamp = msg->header.stamp;   // 保持时间戳同步
    pub_->publish(output);                     // 发布动态点云

    pcl::toROSMsg(other_accumulated_cloud, output);  // 转换其他区域点云
    other_pub_->publish(output);  // 发布其他区域点云

    // 8. 检测飞镖并设置警告状态
    pcl::PointCloud<pcl::PointXYZ> dart_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (dart_cloud_filter(point)) {  // 筛选飞镖区域的点
            dart_cloud.push_back(point);
        }
    }
    if (dart_cloud.size() > 5) {  // 点数超过5时，判定为检测到飞镖
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Find dart cloud!");
        // // 原代码
        // RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Dart cloud size: %d",
        //             dart_cloud.size());

        // 修改后
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Dart cloud size: %zu",
                    dart_cloud.size());

        lidar_detect.dart_state = 1;  // 设置飞镖警告状态
    }

    // 9. 检测飞机并设置警告状态（安全/警告/警报）
    pcl::PointCloud<pcl::PointXYZ> fly_safe_cloud, fly_warn_cloud,
        fly_alarm_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (fly_safe_filter(point))
            fly_safe_cloud.push_back(point);  // 安全区
        if (fly_warn_filter(point))
            fly_warn_cloud.push_back(point);  // 警告区
        if (fly_alarm_filter(point))
            fly_alarm_cloud.push_back(point);  // 警报区
    }
    // 根据点数设置飞机状态（超过40点视为有效）
    if (fly_safe_cloud.size() > 40)
        lidar_detect.fly_state = 1;  // 安全
    if (fly_warn_cloud.size() > 40)
        lidar_detect.fly_state = 2;  // 警告
    if (fly_alarm_cloud.size() > 40)
        lidar_detect.fly_state = 3;  // 警报

    // 打印飞机检测日志
    switch (lidar_detect.fly_state) {
    case 1:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Safe fly object detected!");
        break;
    case 2:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Warn fly object detected!");
        break;
    case 3:
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                     "Alarm fly object detected!");
        break;
    default:
        break;
    }

    // 10. 发布检测警告消息
    detect_pub_->publish(lidar_detect);

    // 11. 打印回调函数总耗时（毫秒）
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "Dynamic callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::DynamicCloud)
