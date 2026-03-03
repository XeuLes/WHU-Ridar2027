#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <cmath>

#include <boost/shared_ptr.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace tdt_radar {

struct Grid {
    pcl::PointXYZ farthestPoint;
    double maxDistance = -1.0;
};

class Localization : public rclcpp::Node {
public:
    Localization(const rclcpp::NodeOptions& node_options)
        : Node("localization", node_options)
    {
        // ---------------------------------------------------------
        // 1. 参数声明
        // ---------------------------------------------------------
        this->declare_parameter<std::string>("target_pcd_path", "");

        // 初值模式: offline / online
        this->declare_parameter<std::string>("initial_pose_mode", "offline");

        // offline 初值参数
        this->declare_parameter<double>("initial_x", 0.0);
        this->declare_parameter<double>("initial_y", 0.0);
        this->declare_parameter<double>("initial_z", 0.0);
        this->declare_parameter<double>("initial_roll", 0.0);
        this->declare_parameter<double>("initial_pitch", 0.0);
        this->declare_parameter<double>("initial_yaw", 0.0);

        // online 读取两个 yaml
        this->declare_parameter<std::string>("camera_extrinsic_yaml", "");
        this->declare_parameter<std::string>("camera_to_lidar_yaml", "");

        // GICP 总尝试时间超时
        this->declare_parameter<bool>("enable_gicp_total_timeout", true);
        this->declare_parameter<double>("gicp_total_timeout_sec", 10.0);
        this->declare_parameter<bool>("fallback_to_initial_after_timeout", true);

        // GICP 参数
        this->declare_parameter<int>("gicp_max_iterations", 100);
        this->declare_parameter<double>("gicp_transformation_epsilon", 1e-8);
        this->declare_parameter<double>("gicp_euclidean_fitness_epsilon", 1e-6);
        this->declare_parameter<double>("gicp_success_fitness_threshold", 0.2);

        // 累积与滤波
        this->declare_parameter<int>("accumulate_time", 30);
        this->declare_parameter<double>("grid_size_degrees", 0.1);
        this->declare_parameter<double>("downsample_leaf_size", 0.1);

        // ROI
        this->declare_parameter<double>("roi_min_x", 0.0);
        this->declare_parameter<double>("roi_max_x", 12.0);
        this->declare_parameter<double>("roi_min_y", -10.0);
        this->declare_parameter<double>("roi_max_y", 10.0);
        this->declare_parameter<double>("roi_max_z", 5.0);

        // TF 稳定播发频率
        this->declare_parameter<double>("tf_publish_rate_hz", 20.0);

        // ---------------------------------------------------------
        // 2. 读取参数
        // ---------------------------------------------------------
        this->get_parameter("initial_pose_mode", initial_pose_mode_);

        this->get_parameter("enable_gicp_total_timeout", enable_gicp_total_timeout_);
        this->get_parameter("gicp_total_timeout_sec", gicp_total_timeout_sec_);
        this->get_parameter("fallback_to_initial_after_timeout", fallback_to_initial_after_timeout_);

        this->get_parameter("gicp_max_iterations", gicp_max_iterations_);
        this->get_parameter("gicp_transformation_epsilon", gicp_transformation_epsilon_);
        this->get_parameter("gicp_euclidean_fitness_epsilon", gicp_euclidean_fitness_epsilon_);
        this->get_parameter("gicp_success_fitness_threshold", gicp_success_fitness_threshold_);

        this->get_parameter("accumulate_time", accumulate_time_);
        this->get_parameter("grid_size_degrees", grid_size_degrees_);
        this->get_parameter("downsample_leaf_size", downsample_leaf_size_);

        this->get_parameter("roi_min_x", roi_min_x_);
        this->get_parameter("roi_max_x", roi_max_x_);
        this->get_parameter("roi_min_y", roi_min_y_);
        this->get_parameter("roi_max_y", roi_max_y_);
        this->get_parameter("roi_max_z", roi_max_z_);

        this->get_parameter("tf_publish_rate_hz", tf_publish_rate_hz_);
        if (tf_publish_rate_hz_ <= 0.0) {
            tf_publish_rate_hz_ = 20.0;
        }

        // ---------------------------------------------------------
        // 3. 加载地图
        // ---------------------------------------------------------
        std::string target_pcd_file;
        this->get_parameter("target_pcd_path", target_pcd_file);

        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(target_pcd_file, *target_cloud_)) {
            RCLCPP_ERROR(this->get_logger(), "无法加载地图: %s", target_pcd_file.c_str());
            return;
        }
        downsample_cloud(target_cloud_, target_cloud_, static_cast<float>(downsample_leaf_size_));
        RCLCPP_INFO(this->get_logger(), "地图加载成功，点数：%lu", target_cloud_->size());

        // ---------------------------------------------------------
        // 4. 构建初始变换矩阵
        // ---------------------------------------------------------
        bool init_ok = false;
        if (initial_pose_mode_ == "offline") {
            init_ok = buildInitialTransformOffline();
        } else if (initial_pose_mode_ == "online") {
            init_ok = buildInitialTransformOnline();
        } else {
            RCLCPP_ERROR(this->get_logger(),
                         "initial_pose_mode 必须是 'offline' 或 'online'，当前: %s",
                         initial_pose_mode_.c_str());
            return;
        }

        if (!init_ok) {
            RCLCPP_ERROR(this->get_logger(), "初始位姿构建失败，节点退出。");
            return;
        }

        // ---------------------------------------------------------
        // 5. 初始化通信
        // ---------------------------------------------------------
        original_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/original_map", 10);
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/map", 10);
        filter_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filter_map", 10);

        // 如果你后面确认存在旧消息堆积，可以把这里改成 SensorDataQoS / KeepLast(1)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10,
            std::bind(&Localization::callback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 地图点云低频发布
        timer_ = this->create_wall_timer(std::chrono::seconds(10), [this]() {
            sensor_msgs::msg::PointCloud2 target_msg;
            pcl::toROSMsg(*target_cloud_, target_msg);
            target_msg.header.frame_id = "rm_frame";
            target_msg.header.stamp = this->now();
            publisher_->publish(target_msg);
            original_map_pub_->publish(target_msg);
        });

        // TF 高频稳定播发（兜底）
        const auto tf_period =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(1.0 / tf_publish_rate_hz_));
        tf_timer_ = this->create_wall_timer(tf_period, std::bind(&Localization::publishCurrentTF, this));

        RCLCPP_INFO(this->get_logger(),
                    "节点启动完成，正在累积点云以进行自动配准... mode=%s timeout=%s %.2fs tf_rate=%.2fHz",
                    initial_pose_mode_.c_str(),
                    enable_gicp_total_timeout_ ? "enabled" : "disabled",
                    gicp_total_timeout_sec_,
                    tf_publish_rate_hz_);

        // 启动后立刻发一次初始 TF
        publishTF(initial_transform_, this->now());
    }

private:
    // ---------------------------------------------------------
    // 基础工具
    // ---------------------------------------------------------
    void downsample_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& output, float leaf_size)
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxelgrid.setInputCloud(input);
        voxelgrid.filter(*output);
    }

    Eigen::Matrix4f makeTransformFromXYZRPY(
        double x, double y, double z,
        double roll, double pitch, double yaw)
    {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        Eigen::AngleAxisf roll_angle(static_cast<float>(roll), Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitch_angle(static_cast<float>(pitch), Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yaw_angle(static_cast<float>(yaw), Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = yaw_angle * pitch_angle * roll_angle;
        T.block<3, 3>(0, 0) = q.matrix();
        T(0, 3) = static_cast<float>(x);
        T(1, 3) = static_cast<float>(y);
        T(2, 3) = static_cast<float>(z);
        return T;
    }

    void printTransformInfo(const std::string& title, const Eigen::Matrix4f& T)
    {
        float x = T(0, 3);
        float y = T(1, 3);
        float z = T(2, 3);
        Eigen::Affine3f affine(T);
        Eigen::Vector3f rpy = affine.rotation().eulerAngles(2, 1, 0);  // yaw pitch roll

        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "%s", title.c_str());
        RCLCPP_INFO(this->get_logger(), "XYZ: [%.3f, %.3f, %.3f]", x, y, z);
        RCLCPP_INFO(this->get_logger(), "RPY: [%.3f, %.3f, %.3f]", rpy[2], rpy[1], rpy[0]);
        RCLCPP_INFO(this->get_logger(), "========================================");
    }

    bool isZeroStamp(const builtin_interfaces::msg::Time& t) const
    {
        return (t.sec == 0 && t.nanosec == 0);
    }

    rclcpp::Time safeMsgStamp(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        if (!isZeroStamp(msg->header.stamp)) {
            return rclcpp::Time(msg->header.stamp);
        }
        return this->now();
    }

    // ---------------------------------------------------------
    // 当前应该播发哪一个 TF
    // 对外严格二态：
    // - 未成功：initial_transform_
    // - 成功后：final_transform_
    // ---------------------------------------------------------
    Eigen::Matrix4f getCurrentTfTransform() const
    {
        if (has_aligned_) {
            return final_transform_;
        }
        return initial_transform_;
    }

    void publishCurrentTF()
    {
        publishTF(getCurrentTfTransform(), this->now());
    }

    // ---------------------------------------------------------
    // offline 初值
    // ---------------------------------------------------------
    bool buildInitialTransformOffline()
    {
        double x, y, z, r, p, yw;
        this->get_parameter("initial_x", x);
        this->get_parameter("initial_y", y);
        this->get_parameter("initial_z", z);
        this->get_parameter("initial_roll", r);
        this->get_parameter("initial_pitch", p);
        this->get_parameter("initial_yaw", yw);

        initial_transform_ = makeTransformFromXYZRPY(x, y, z, r, p, yw);
        printTransformInfo("已加载初始位姿 (From offline YAML):", initial_transform_);
        return true;
    }

    // ---------------------------------------------------------
    // online 初值：读取两个 yaml 计算
    // ---------------------------------------------------------
    bool readCameraExtrinsicYaml(Eigen::Matrix4d& T_world_camera)
    {
        std::string camera_extrinsic_yaml;
        this->get_parameter("camera_extrinsic_yaml", camera_extrinsic_yaml);

        if (camera_extrinsic_yaml.empty()) {
            RCLCPP_ERROR(this->get_logger(), "camera_extrinsic_yaml 为空");
            return false;
        }

        cv::FileStorage fs(camera_extrinsic_yaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开 camera_extrinsic_yaml: %s",
                         camera_extrinsic_yaml.c_str());
            return false;
        }

        cv::Mat rvec_cv, tvec_cv;
        fs["world_rvec"] >> rvec_cv;
        fs["world_tvec"] >> tvec_cv;
        fs.release();

        if (rvec_cv.empty() || tvec_cv.empty()) {
            RCLCPP_ERROR(this->get_logger(),
                         "camera_extrinsic_yaml 中缺少 world_rvec / world_tvec");
            return false;
        }

        cv::Mat R_cv;
        cv::Rodrigues(rvec_cv, R_cv);

        Eigen::Matrix3d R_world_to_camera;
        Eigen::Vector3d t_world_to_camera;
        cv::cv2eigen(R_cv, R_world_to_camera);
        cv::cv2eigen(tvec_cv, t_world_to_camera);

        Eigen::Matrix4d T_camera_world = Eigen::Matrix4d::Identity();
        T_camera_world.block<3, 3>(0, 0) = R_world_to_camera;
        T_camera_world.block<3, 1>(0, 3) = t_world_to_camera;

        T_world_camera = T_camera_world.inverse();
        return true;
    }

    bool readCameraToLidarYaml(Eigen::Matrix4d& T_camera_lidar)
    {
        std::string camera_to_lidar_yaml;
        this->get_parameter("camera_to_lidar_yaml", camera_to_lidar_yaml);

        if (camera_to_lidar_yaml.empty()) {
            RCLCPP_ERROR(this->get_logger(), "camera_to_lidar_yaml 为空");
            return false;
        }

        cv::FileStorage fs(camera_to_lidar_yaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开 camera_to_lidar_yaml: %s",
                         camera_to_lidar_yaml.c_str());
            return false;
        }

        double dx = 0.0, dy = 0.0, dz = 0.0;
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        bool use_deg = false;

        if (!fs["translation"].empty()) {
            cv::FileNode trans = fs["translation"];
            if (trans.size() == 3) {
                dx = (double)trans[0];
                dy = (double)trans[1];
                dz = (double)trans[2];
            }
        } else {
            if (!fs["dx"].empty()) dx = (double)fs["dx"];
            if (!fs["dy"].empty()) dy = (double)fs["dy"];
            if (!fs["dz"].empty()) dz = (double)fs["dz"];
        }

        if (!fs["rotation_rpy_deg"].empty()) {
            cv::FileNode rpy = fs["rotation_rpy_deg"];
            if (rpy.size() == 3) {
                roll = (double)rpy[0];
                pitch = (double)rpy[1];
                yaw = (double)rpy[2];
                use_deg = true;
            }
        } else if (!fs["rotation_rpy_rad"].empty()) {
            cv::FileNode rpy = fs["rotation_rpy_rad"];
            if (rpy.size() == 3) {
                roll = (double)rpy[0];
                pitch = (double)rpy[1];
                yaw = (double)rpy[2];
                use_deg = false;
            }
        } else {
            if (!fs["roll_deg"].empty() || !fs["pitch_deg"].empty() || !fs["yaw_deg"].empty()) {
                if (!fs["roll_deg"].empty()) roll = (double)fs["roll_deg"];
                if (!fs["pitch_deg"].empty()) pitch = (double)fs["pitch_deg"];
                if (!fs["yaw_deg"].empty()) yaw = (double)fs["yaw_deg"];
                use_deg = true;
            } else {
                if (!fs["roll"].empty()) roll = (double)fs["roll"];
                if (!fs["pitch"].empty()) pitch = (double)fs["pitch"];
                if (!fs["yaw"].empty()) yaw = (double)fs["yaw"];
                use_deg = false;
            }
        }

        fs.release();

        if (use_deg) {
            roll = roll * M_PI / 180.0;
            pitch = pitch * M_PI / 180.0;
            yaw = yaw * M_PI / 180.0;
        }

        Eigen::Matrix3d base_R;
        base_R << 0, -1,  0,
                  0,  0, -1,
                  1,  0,  0;

        Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d extra_R = (yaw_angle * pitch_angle * roll_angle).matrix();

        T_camera_lidar = Eigen::Matrix4d::Identity();
        T_camera_lidar.block<3, 3>(0, 0) = base_R * extra_R;
        T_camera_lidar(0, 3) = dx;
        T_camera_lidar(1, 3) = dy;
        T_camera_lidar(2, 3) = dz;

        return true;
    }

    bool buildInitialTransformOnline()
    {
        Eigen::Matrix4d T_world_camera = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_camera_lidar = Eigen::Matrix4d::Identity();

        if (!readCameraExtrinsicYaml(T_world_camera)) {
            return false;
        }
        if (!readCameraToLidarYaml(T_camera_lidar)) {
            return false;
        }

        Eigen::Matrix4d T_world_lidar = T_world_camera * T_camera_lidar;
        initial_transform_ = T_world_lidar.cast<float>();

        printTransformInfo("已在线计算初始位姿 (camera_extrinsic_yaml + camera_to_lidar_yaml):",
                           initial_transform_);
        return true;
    }

    // ---------------------------------------------------------
    // 回调
    // ---------------------------------------------------------
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        const rclcpp::Time stamp = safeMsgStamp(msg);

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *source_cloud);

        // 先按消息时间发当前有效 TF
        // 未成功 -> 初值
        // 成功后 -> 最终值
        publishTF(getCurrentTfTransform(), stamp);

        // 1. 已锁定：不再继续跑 GICP
        if (has_aligned_) {
            return;
        }

        // 2. 已总超时：不再尝试 GICP，继续使用初值
        if (gicp_timeout_reached_) {
            return;
        }

        // 3. 累积点云
        if (accumulated_clouds_.size() < static_cast<size_t>(accumulate_time_)) {
            accumulated_clouds_.push_back(source_cloud);
            return;
        } else {
            accumulated_clouds_.erase(accumulated_clouds_.begin());
            accumulated_clouds_.push_back(source_cloud);
        }

        // 第一次进入“可尝试 GICP”的阶段，开始总时间计时
        if (!gicp_started_) {
            gicp_started_ = true;
            gicp_start_time_ = this->now();
            RCLCPP_INFO(this->get_logger(), "开始 GICP 自动配准计时");
        }

        // 4. 检查总超时
        if (enable_gicp_total_timeout_) {
            const double elapsed = (this->now() - gicp_start_time_).seconds();
            if (elapsed > gicp_total_timeout_sec_) {
                gicp_timeout_reached_ = true;
                RCLCPP_WARN(this->get_logger(),
                            "GICP 总尝试时间超时(%.3f sec > %.3f sec)，停止自动配准，继续使用初值",
                            elapsed, gicp_total_timeout_sec_);
                publishTF(initial_transform_, stamp);
                return;
            }
        }

        // 5. 栅格去噪
        grid_map_.clear();
        for (auto& cloud : accumulated_clouds_) {
            for (const auto& point : *cloud) {
                if (point.x <= 0) continue;

                const double azimuth = std::atan2(point.y, point.x);
                const double elevation =
                    std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));

                const int az_idx =
                    static_cast<int>(std::floor(azimuth * 180.0 / M_PI / grid_size_degrees_));
                const int el_idx =
                    static_cast<int>(std::floor(elevation * 180.0 / M_PI / grid_size_degrees_));

                const double dist =
                    std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

                auto& grid = grid_map_[{az_idx, el_idx}];
                if (dist > grid.maxDistance) {
                    grid.farthestPoint = point;
                    grid.maxDistance = dist;
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr grid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& kv : grid_map_) {
            const auto& grid = kv.second;
            if (grid.maxDistance > 0.0) {
                grid_cloud->push_back(grid.farthestPoint);
            }
        }

        // 6. ROI 过滤
        pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& point : grid_cloud->points) {
            if (point.x > roi_min_x_ && point.x < roi_max_x_ &&
                point.y > roi_min_y_ && point.y < roi_max_y_ &&
                point.z < roi_max_z_) {
                roi_cloud->push_back(point);
            }
        }

        if (roi_cloud->empty()) {
            return;
        }

        // 7. 下采样
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_source(new pcl::PointCloud<pcl::PointXYZ>());
        downsample_cloud(roi_cloud, downsampled_source, static_cast<float>(downsample_leaf_size_));

        sensor_msgs::msg::PointCloud2 filter_msg;
        pcl::toROSMsg(*downsampled_source, filter_msg);
        filter_msg.header.frame_id = "livox_frame";
        filter_msg.header.stamp = msg->header.stamp;
        if (isZeroStamp(filter_msg.header.stamp)) {
            filter_msg.header.stamp = stamp;
        }
        filter_publisher_->publish(filter_msg);

        // 8. GICP 自动精配准
        // 注意：内部始终使用 initial_transform_ 作为 guess
        RCLCPP_INFO_ONCE(this->get_logger(), "🟢 正在执行 GICP 自动配准...");

        auto gicp =
            pcl::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
        gicp->setMaximumIterations(gicp_max_iterations_);
        gicp->setTransformationEpsilon(gicp_transformation_epsilon_);
        gicp->setEuclideanFitnessEpsilon(gicp_euclidean_fitness_epsilon_);

        gicp->setInputTarget(target_cloud_);
        gicp->setInputSource(downsampled_source);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        gicp->align(*aligned_cloud, initial_transform_);

        const double fit_score = gicp->getFitnessScore();
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "配准中... Score: %.4f", fit_score);

        // 对外绝不使用中间结果
        // 只有满足阈值才锁定 final_transform_
        if (gicp->hasConverged() && fit_score < gicp_success_fitness_threshold_) {
            has_aligned_ = true;
            final_transform_ = gicp->getFinalTransformation();

            RCLCPP_INFO(this->get_logger(), "✅ 配准成功! 位姿已锁定。");
            printTransformInfo("最终结合位姿 (Result):", final_transform_);

            publishTF(final_transform_, stamp);
        }
    }

    void publishTF(const Eigen::Matrix4f& transform_in, const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped ts;
        ts.header.stamp = stamp;
        ts.header.frame_id = "rm_frame";
        ts.child_frame_id = "livox_frame";

        ts.transform.translation.x = transform_in(0, 3);
        ts.transform.translation.y = transform_in(1, 3);
        ts.transform.translation.z = transform_in(2, 3);

        Eigen::Quaternionf q(transform_in.block<3, 3>(0, 0));
        q.normalize();
        ts.transform.rotation.x = q.x();
        ts.transform.rotation.y = q.y();
        ts.transform.rotation.z = q.z();
        ts.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(ts);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Publishing TF rm_frame -> livox_frame using %s transform, stamp=%.3f",
            has_aligned_ ? "final" : "initial",
            stamp.seconds());
    }

private:
    // 状态
    bool has_aligned_ = false;
    bool gicp_started_ = false;
    bool gicp_timeout_reached_ = false;

    // 变换
    Eigen::Matrix4f initial_transform_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f final_transform_ = Eigen::Matrix4f::Identity();

    // 初值模式
    std::string initial_pose_mode_{"offline"};

    // GICP 总时间超时
    bool enable_gicp_total_timeout_{true};
    double gicp_total_timeout_sec_{10.0};
    bool fallback_to_initial_after_timeout_{true};
    rclcpp::Time gicp_start_time_{0, 0, RCL_ROS_TIME};

    // GICP 参数
    int gicp_max_iterations_{100};
    double gicp_transformation_epsilon_{1e-8};
    double gicp_euclidean_fitness_epsilon_{1e-6};
    double gicp_success_fitness_threshold_{0.2};

    // TF 发布
    double tf_publish_rate_hz_{20.0};

    // 数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;

    int accumulate_time_ = 30;
    double grid_size_degrees_ = 0.1;
    double downsample_leaf_size_ = 0.1;

    // ROI
    double roi_min_x_{0.0};
    double roi_max_x_{12.0};
    double roi_min_y_{-10.0};
    double roi_max_y_{10.0};
    double roi_max_z_{5.0};

    std::map<std::pair<int, int>, Grid> grid_map_;

    // ROS
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr original_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

} // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)