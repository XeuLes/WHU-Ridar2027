#include "dynamic_cloud.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <mutex>

namespace tdt_radar {  // 自定义命名空间


DynamicCloud::DynamicCloud(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("dynamic_cloud_node", node_options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Dynamic_cloud Node start");

    // === 参数声明 ===
    this->declare_parameter<std::string>(
        "map_pcd_path", "/home/user/code/T-DT_Radar-main/src/lidar/localization/config/基础制造_rm_frame.pcd");
    this->declare_parameter<int>("actual_map_min_frames", 30);
    this->declare_parameter<int>("actual_map_min_stable_voxels", 400);
    this->declare_parameter<double>("actual_map_voxel_size", 0.10);
    this->declare_parameter<double>("actual_map_min_observe_ratio", 0.20);

    std::string map_pcd_path;
    this->get_parameter("map_pcd_path", map_pcd_path);
    this->get_parameter("actual_map_min_frames", actual_map_min_frames_);
    this->get_parameter("actual_map_min_stable_voxels", actual_map_min_stable_voxels_);
    this->get_parameter("actual_map_voxel_size", actual_map_voxel_size_);
    this->get_parameter("actual_map_min_observe_ratio", actual_map_min_observe_ratio_);

    // === 加载地图点云 ===
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd_path, *temp_cloud) == -1) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load map pcd: %s", map_pcd_path.c_str());
        return;
    }

    // 对地图点云进行体素下采样
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(temp_cloud);
    sor.setLeafSize(0.2f, 0.2f, 0.2f);
    auto result = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*result);
    prior_map_cloud_ = result;
    kd_Tree.setInputCloud(prior_map_cloud_);
    actual_map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // 初始化订阅者和发布者
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_integrated", 10,
    std::bind(&DynamicCloud::callback, this, std::placeholders::_1));

    localization_status_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
    "/localization/status", 10,
    std::bind(&DynamicCloud::localizationStatusCallback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar_dynamic", 10);
    other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar_other", 10);
    detect_pub_ = this->create_publisher<vision_interface::msg::RadarWarn>("/lidar_detect", 10);

    // [新增] 专门用于发布 ROI 过滤后点云的话题
    roi_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/roi_cloud", 10);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Dynamic Cloud Launch!");
}

void DynamicCloud::GetDynamicCloud(
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud,
    float threshold,
    int thread_num)
{
    int K = 1;
    std::vector<std::thread> threads;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);

    int cloud_size = input_cloud.points.size();
    int step = cloud_size / thread_num;

    // 多线程处理
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread([i, step, cloud_size, &clouds, &input_cloud, this, threshold, K]() {
            for (int j = i * step; j < (i + 1) * step && j < cloud_size; j++) {
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);

                // 在静态地图中搜索最近邻
                if (kd_Tree.nearestKSearch(input_cloud.points[j], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                    // 如果距离大于阈值，认为是动态点
                    if (pointNKNSquaredDistance[0] > threshold) {
                        clouds[i].points.push_back(input_cloud.points[j]);
                    }
                }
            }
        }));
    }

    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
}

void DynamicCloud::localizationStatusCallback(const std_msgs::msg::UInt8::SharedPtr msg)
{
    LocalizationStatus new_status = LocalizationStatus::LOCALIZING;

    switch (msg->data) {
        case 0:
            new_status = LocalizationStatus::LOCALIZING;
            break;
        case 1:
            new_status = LocalizationStatus::PRIOR_SUCCESS;
            break;
        case 2:
            new_status = LocalizationStatus::INIT_ONLY;
            break;
        default:
            RCLCPP_WARN(this->get_logger(),
                        "收到未知 localization status=%u，忽略", msg->data);
            return;
    }

    LocalizationStatus old_status;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        old_status = localization_status_;
        localization_status_ = new_status;
    }

    // 状态迁移逻辑
    if (new_status == LocalizationStatus::LOCALIZING &&
        old_status != LocalizationStatus::LOCALIZING) {
        resetActualMapBuildState();
        {
            std::lock_guard<std::mutex> lock(actual_map_mutex_);
            actual_map_building_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "进入 LOCALIZING，开始累计 actual map");
    }

    if (new_status == LocalizationStatus::PRIOR_SUCCESS) {
        resetActualMapBuildState();
        RCLCPP_INFO(this->get_logger(), "进入 PRIOR_SUCCESS，放弃本轮 actual map，继续使用先验地图");
    }

    if (new_status == LocalizationStatus::INIT_ONLY &&
        old_status != LocalizationStatus::INIT_ONLY) {
        const bool ok = finalizeActualMap();
        RCLCPP_INFO(this->get_logger(),
                    "进入 INIT_ONLY，actual map finalize result = %s",
                    ok ? "READY" : "NOT_READY");
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        refreshRuntimeMapMode();
        RCLCPP_INFO(this->get_logger(),
                    "Localization status updated -> %s, runtime map mode -> %s",
                    localizationStatusToString(localization_status_),
                    mapModeToString(runtime_map_mode_));
    }
}

void DynamicCloud::resetActualMapBuildState()
{
    std::lock_guard<std::mutex> lock(actual_map_mutex_);

    actual_voxel_hit_count_.clear();
    actual_voxel_sum_.clear();

    actual_map_accumulated_frames_ = 0;
    actual_map_ready_ = false;
    actual_map_building_ = false;
    actual_map_finalized_once_ = false;

    actual_map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
}

std::string DynamicCloud::makeVoxelKey(const pcl::PointXYZ& pt) const
{
    const int ix = static_cast<int>(std::floor(pt.x / actual_map_voxel_size_));
    const int iy = static_cast<int>(std::floor(pt.y / actual_map_voxel_size_));
    const int iz = static_cast<int>(std::floor(pt.z / actual_map_voxel_size_));
    return std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(iz);
}

void DynamicCloud::updateActualMapAccumulator(const pcl::PointCloud<pcl::PointXYZ>& roi_cloud)
{
    std::lock_guard<std::mutex> lock(actual_map_mutex_);

    actual_map_building_ = true;
    actual_map_accumulated_frames_++;

    std::unordered_map<std::string, bool> frame_seen;

    for (const auto& pt : roi_cloud.points) {
        const std::string key = makeVoxelKey(pt);

        if (frame_seen.find(key) != frame_seen.end()) {
            continue;
        }
        frame_seen[key] = true;

        actual_voxel_hit_count_[key] += 1;
        actual_voxel_sum_[key] += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
}

bool DynamicCloud::finalizeActualMap()
{
    std::lock_guard<std::mutex> lock(actual_map_mutex_);

    if (actual_map_finalized_once_) {
        return actual_map_ready_;
    }
    actual_map_finalized_once_ = true;
    actual_map_building_ = false;

    if (actual_map_accumulated_frames_ < actual_map_min_frames_) {
        RCLCPP_WARN(this->get_logger(),
                    "actual map 构建失败：累计帧数不足，frames=%d < min_frames=%d",
                    actual_map_accumulated_frames_, actual_map_min_frames_);
        actual_map_ready_ = false;
        return false;
    }

    const int min_hit_count = std::max(
        1,
        static_cast<int>(std::ceil(actual_map_accumulated_frames_ * actual_map_min_observe_ratio_)));

    pcl::PointCloud<pcl::PointXYZ>::Ptr built_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto& kv : actual_voxel_hit_count_) {
        const std::string& key = kv.first;
        const int hit_count = kv.second;

        if (hit_count < min_hit_count) {
            continue;
        }

        auto it_sum = actual_voxel_sum_.find(key);
        if (it_sum == actual_voxel_sum_.end()) {
            continue;
        }

        const Eigen::Vector3f mean = it_sum->second / static_cast<float>(hit_count);

        pcl::PointXYZ pt;
        pt.x = mean.x();
        pt.y = mean.y();
        pt.z = mean.z();
        built_cloud->push_back(pt);
    }

    if (static_cast<int>(built_cloud->size()) < actual_map_min_stable_voxels_) {
        RCLCPP_WARN(this->get_logger(),
                    "actual map 构建失败：稳定体素不足，stable_voxels=%zu < min_stable_voxels=%d",
                    built_cloud->size(), actual_map_min_stable_voxels_);
        actual_map_ready_ = false;
        return false;
    }

    actual_map_cloud_ = built_cloud;
    actual_map_ready_ = true;

    RCLCPP_INFO(this->get_logger(),
                "actual map 构建成功：frames=%d, min_hit=%d, stable_voxels=%zu",
                actual_map_accumulated_frames_, min_hit_count, actual_map_cloud_->size());

    return true;
}

void DynamicCloud::refreshRuntimeMapMode()
{
    switch (localization_status_) {
        case LocalizationStatus::PRIOR_SUCCESS:
            runtime_map_mode_ = RuntimeMapMode::PRIOR_MAP;
            if (prior_map_cloud_ && !prior_map_cloud_->empty()) {
                kd_Tree.setInputCloud(prior_map_cloud_);
            }
            break;

        case LocalizationStatus::LOCALIZING:
            // 这一阶段先仍然保留先验地图差分
            runtime_map_mode_ = RuntimeMapMode::PRIOR_MAP;
            if (prior_map_cloud_ && !prior_map_cloud_->empty()) {
                kd_Tree.setInputCloud(prior_map_cloud_);
            }
            break;

        case LocalizationStatus::INIT_ONLY:
            if (actual_map_ready_ && actual_map_cloud_ && !actual_map_cloud_->empty()) {
                runtime_map_mode_ = RuntimeMapMode::ACTUAL_MAP;
                kd_Tree.setInputCloud(actual_map_cloud_);
            } else {
                runtime_map_mode_ = RuntimeMapMode::NO_VALID_MAP;
            }
            break;

        default:
            runtime_map_mode_ = RuntimeMapMode::NO_VALID_MAP;
            break;
    }
}

const char* DynamicCloud::mapModeToString(RuntimeMapMode mode) const
{
    switch (mode) {
        case RuntimeMapMode::PRIOR_MAP: return "PRIOR_MAP";
        case RuntimeMapMode::ACTUAL_MAP: return "ACTUAL_MAP";
        case RuntimeMapMode::NO_VALID_MAP: return "NO_VALID_MAP";
        default: return "UNKNOWN_MAP_MODE";
    }
}

const char* DynamicCloud::localizationStatusToString(LocalizationStatus status) const
{
    switch (status) {
        case LocalizationStatus::LOCALIZING: return "LOCALIZING";
        case LocalizationStatus::PRIOR_SUCCESS: return "PRIOR_SUCCESS";
        case LocalizationStatus::INIT_ONLY: return "INIT_ONLY";
        default: return "UNKNOWN_LOCALIZATION_STATUS";
    }
}

// TransformCloud 函数保持不变 (略去以节省篇幅，逻辑与之前一致)
void TransformCloud(
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud,
    geometry_msgs::msg::TransformStamped transform_stamped,
    int thread_num)
{
    // ... 原有逻辑 ...
    std::vector<std::thread> threads;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    int cloud_size = input_cloud.points.size();
    int step = cloud_size / thread_num;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(
        transform_stamped.transform.rotation.w,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z);
    transform.rotate(rotation);

    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(
            [i, step, cloud_size, &clouds, &input_cloud, transform]() {
                for (int j = i * step; j < (i + 1) * step && j < cloud_size; j++) {
                    pcl::PointXYZ point = input_cloud.points[j];
                    Eigen::Vector3f point_vec(point.x, point.y, point.z);
                    Eigen::Vector3f point_out = transform * point_vec;
                    pcl::PointXYZ transformed_point;
                    transformed_point.x = point_out.x();
                    transformed_point.y = point_out.y();
                    transformed_point.z = point_out.z();
                    clouds[i].points.push_back(transformed_point);
                }
            }));
    }
    for (auto& t : threads) { t.join(); }
    for (auto& cloud : clouds) { output_cloud += cloud; }
}

void DynamicCloud::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    vision_interface::msg::RadarWarn lidar_detect;

    RuntimeMapMode current_mode;
    LocalizationStatus current_status;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        current_mode = runtime_map_mode_;
        current_status = localization_status_;
    }

    // === 定义区域过滤器 ===
    auto dart_cloud_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
               (point.y > 3.925 && point.y < 4.525) &&
               (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722);
    };

    auto fly_safe_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 2.775 && point.x < 27.5) &&
               (point.y > 0.2 && point.y < 2.2) &&
               (point.z > 1.7 && point.z < 3);
    };

    auto fly_warn_filter = [](pcl::PointXYZ& point) {
        return (point.x > 19.83 && point.x < 28 - 2.7) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };

    auto fly_alarm_filter = [](pcl::PointXYZ& point) {
        return (point.x > 13 && point.x < 20.5) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };

    // 1. 转换ROS消息到PCL
    pcl::PointCloud<pcl::PointXYZ> receive_cloud;
    pcl::fromROSMsg(*msg, receive_cloud);
    
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                     "【阶段1】原始点数：%zu", receive_cloud.size());

    // 2. 获取坐标变换
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        transform_stamped = tf_buffer_.lookupTransform(
            "rm_frame", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Transform error: %s", ex.what());
        return;
    }

    // 3. 坐标变换
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);
    
    pcl::transformPointCloud(receive_cloud, transformed_cloud, transform);

    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                     "【阶段3】变换后点数（rm_frame）：%zu",
    //                     transformed_cloud.size());

    // 4. ROI 过滤
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ> other_filtered_cloud;

    for (size_t i = 0; i < transformed_cloud.size(); i++) {
        auto& point = transformed_cloud.points[i];
        
        // ROI 边界判断 (去除场外点、地面点、高空点)
        bool is_out_of_bounds = (point.x < 10.0 || point.x > 40.0 || 
                                 point.y < 2.0 || point.y > 10.0 || 
                                 point.z < -1 || point.z > 5.0);

        if (is_out_of_bounds) {
            // 虽然是场外点，但检查是否在特殊关注区域（如飞镖/飞机区）
            if ((point.x > 27.0 && point.x < 28.0 && point.y > 3.9 && point.y < 4.6 && point.z > 1.5) ||
                (point.x > 13.0 && point.x < 27.5 && point.y > 0.2 && point.y < 4.5 && point.z > 1.7)) {
                other_filtered_cloud.push_back(point);
            }
            continue; 
        }
        
        // // 己方死区过滤
        // float sum_xy = point.x + point.y;
        // float sub_yx = point.y - point.x;
        // if (sum_xy > 19.4 && sum_xy < 23.6 && sub_yx > -7.2 && sub_yx < -5.8) {
        //     continue; 
        // }

        filtered_cloud.push_back(point); 
    }
    
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                      "【阶段4】ROI过滤后有效点数：%zu",
    //                      filtered_cloud.size());

    // =========================================================
    // 【新增】发布 ROI 过滤后的点云，方便 RViz 查看
    // =========================================================
    {
        sensor_msgs::msg::PointCloud2 roi_msg;
        pcl::toROSMsg(filtered_cloud, roi_msg);
        roi_msg.header.frame_id = "rm_frame";
        roi_msg.header.stamp = msg->header.stamp;
        roi_pub_->publish(roi_msg); // 话题: /livox/roi_cloud
    }

    if (current_status == LocalizationStatus::LOCALIZING) {
    updateActualMapAccumulator(filtered_cloud);

    int actual_frames = 0;
    {
        std::lock_guard<std::mutex> lock(actual_map_mutex_);
        actual_frames = actual_map_accumulated_frames_;
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "actual map 累计中... frames=%d",
                        actual_frames);
    }

    // 5. 提取动态点云 (对比静态地图)
    pcl::PointCloud<pcl::PointXYZ> dynamic_pointcloud;

    if (current_mode == RuntimeMapMode::NO_VALID_MAP) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "当前无有效差分地图，跳过动态差分");
    } else {
        GetDynamicCloud(filtered_cloud, dynamic_pointcloud, 0.03, 12);
    }
    
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                      "【阶段5】动态检测后点数：%zu",
    //                      dynamic_pointcloud.size());

    // 6. 多帧累积
    if (accumulate_count < accumulate_time) {
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.push_back(other_filtered_cloud.makeShared());
        accumulate_count++;
    } else {
        accumulated_clouds_.erase(accumulated_clouds_.begin());
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.erase(other_accumulated_clouds_.begin());
        other_accumulated_clouds_.push_back(other_filtered_cloud.makeShared());
    }

    // 合并累积点云
    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;
    for (auto it = accumulated_clouds_.begin(); it != accumulated_clouds_.end(); ++it) {
        accumulated_cloud += **it;
    }
    pcl::PointCloud<pcl::PointXYZ> other_accumulated_cloud;
    for (auto it = other_accumulated_clouds_.begin(); it != other_accumulated_clouds_.end(); ++it) {
        other_accumulated_cloud += **it;
    }

    // 7. 发布最终结果
    sensor_msgs::msg::PointCloud2 output;
    
    // 发布动态点云
    pcl::toROSMsg(accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    pub_->publish(output);

    // 发布其他区域点云
    pcl::toROSMsg(other_accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    other_pub_->publish(output);

    if (current_status == LocalizationStatus::PRIOR_SUCCESS) {
    // 原来的飞镖 / 飞机检测逻辑
    // 8. 飞镖检测逻辑
    pcl::PointCloud<pcl::PointXYZ> dart_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (dart_cloud_filter(point)) {
            dart_cloud.push_back(point);
        }
    }
    if (dart_cloud.size() > 5) {
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Find dart cloud! Size: %zu", dart_cloud.size());
        lidar_detect.dart_state = 1;
    }

    // 9. 飞机检测逻辑
    pcl::PointCloud<pcl::PointXYZ> fly_safe_cloud, fly_warn_cloud, fly_alarm_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (fly_safe_filter(point)) fly_safe_cloud.push_back(point);
        if (fly_warn_filter(point)) fly_warn_cloud.push_back(point);
        if (fly_alarm_filter(point)) fly_alarm_cloud.push_back(point);
    }
    
    if (fly_safe_cloud.size() > 40) lidar_detect.fly_state = 1;
    if (fly_warn_cloud.size() > 40) lidar_detect.fly_state = 2;
    if (fly_alarm_cloud.size() > 40) lidar_detect.fly_state = 3;

    // 10. 发布检测警告
    detect_pub_->publish(lidar_detect);

    } else {
        lidar_detect.dart_state = 0;
        lidar_detect.fly_state = 0;
    }

    
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::DynamicCloud)
