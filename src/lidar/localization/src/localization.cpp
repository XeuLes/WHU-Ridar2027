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
#include <boost/shared_ptr.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace tdt_radar {

struct Grid {
    pcl::PointXYZ farthestPoint;
    double        maxDistance = -1.0;
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
        
        // 初始位姿参数 (从 YAML 读取)
        this->declare_parameter<double>("initial_x", 0.0);
        this->declare_parameter<double>("initial_y", 0.0);
        this->declare_parameter<double>("initial_z", 0.0);
        this->declare_parameter<double>("initial_roll", 0.0);
        this->declare_parameter<double>("initial_pitch", 0.0);
        this->declare_parameter<double>("initial_yaw", 0.0);

        // ---------------------------------------------------------
        // 2. 加载地图
        // ---------------------------------------------------------
        std::string target_pcd_file;
        this->get_parameter("target_pcd_path", target_pcd_file);

        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(target_pcd_file, *target_cloud_)) {
            RCLCPP_ERROR(this->get_logger(), "无法加载地图: %s", target_pcd_file.c_str());
            return;
        }
        downsample_cloud(target_cloud_, target_cloud_, 0.1f);
        RCLCPP_INFO(this->get_logger(), "地图加载成功，点数：%lu", target_cloud_->size());

        // ---------------------------------------------------------
        // 3. 构建初始变换矩阵 (Initial Guess)
        // ---------------------------------------------------------
        double x, y, z, r, p, yw;
        this->get_parameter("initial_x", x);
        this->get_parameter("initial_y", y);
        this->get_parameter("initial_z", z);
        this->get_parameter("initial_roll", r);
        this->get_parameter("initial_pitch", p);
        this->get_parameter("initial_yaw", yw);

        initial_transform_ = Eigen::Matrix4f::Identity();
        Eigen::AngleAxisf roll_angle(r, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitch_angle(p, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yaw_angle(yw, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = yaw_angle * pitch_angle * roll_angle;
        
        initial_transform_.block<3, 3>(0, 0) = q.matrix();
        initial_transform_(0, 3) = x;
        initial_transform_(1, 3) = y;
        initial_transform_(2, 3) = z;

        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "已加载初始位姿 (From YAML):");
        RCLCPP_INFO(this->get_logger(), "XYZ: [%.3f, %.3f, %.3f]", x, y, z);
        RCLCPP_INFO(this->get_logger(), "RPY: [%.3f, %.3f, %.3f]", r, p, yw);
        RCLCPP_INFO(this->get_logger(), "========================================");

        // ---------------------------------------------------------
        // 4. 初始化通信
        // ---------------------------------------------------------
        original_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/original_map", 10);
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/map", 10);
        filter_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filter_map", 10);
        
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10,
                std::bind(&Localization::callback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 定时发布地图
        timer_ = this->create_wall_timer(std::chrono::seconds(10), [this]() {
            sensor_msgs::msg::PointCloud2 target_msg;
            pcl::toROSMsg(*target_cloud_, target_msg);
            target_msg.header.frame_id = "rm_frame";
            publisher_->publish(target_msg);
            original_map_pub_->publish(target_msg);
        });

        RCLCPP_INFO(this->get_logger(), "节点启动完成，正在累积点云以进行自动配准...");
    }

private:
    void downsample_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& output, float leaf_size)
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxelgrid.setInputCloud(input);
        voxelgrid.filter(*output);
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 1. 如果已经配准成功，只发布锁定的TF并返回
        if (has_aligned_) {
            publishTF(transform, msg->header.stamp);
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *source_cloud);

        // 2. 累积点云 (增加稠密度)
        if (accumulated_clouds_.size() < accumulate_time) {
            accumulated_clouds_.push_back(source_cloud);
            return; // 继续等待累积
        } else {
            accumulated_clouds_.erase(accumulated_clouds_.begin());
            accumulated_clouds_.push_back(source_cloud);
        }

        // 3. 栅格去噪
        gridMap.clear();
        for (auto& cloud : accumulated_clouds_) {
            for (const auto& point : *cloud) {
                if (point.x <= 0) continue;
                double azimuth = std::atan2(point.y, point.x);
                double elevation = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));
                int az_idx = static_cast<int>(floor(azimuth * 180 / M_PI / gridSizeDegrees));
                int el_idx = static_cast<int>(floor(elevation * 180 / M_PI / gridSizeDegrees));
                double dist = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
                auto& grid = gridMap[{az_idx, el_idx}];
                if (dist > grid.maxDistance) {
                    grid.farthestPoint = point;
                    grid.maxDistance = dist;
                }
            }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr grid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& [key, grid] : gridMap) {
            if (grid.maxDistance > 0.0) grid_cloud->push_back(grid.farthestPoint);
        }

        // 4. ROI 过滤
        pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& point : grid_cloud->points) {
            if (point.x > 0 && point.x < 12 && point.y > -10 && point.y < 10 && point.z < 5) {
                roi_cloud->push_back(point);
            }
        }
        if (roi_cloud->empty()) return;

        // 5. 下采样
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_source(new pcl::PointCloud<pcl::PointXYZ>());
        downsample_cloud(roi_cloud, downsampled_source, 0.1f);
        
        sensor_msgs::msg::PointCloud2 filter_msg;
        pcl::toROSMsg(*downsampled_source, filter_msg);
        filter_msg.header.frame_id = "livox_frame";
        filter_publisher_->publish(filter_msg);

        // 6. GICP 自动精配准
        RCLCPP_INFO_ONCE(this->get_logger(), "🟢 正在执行 GICP 自动配准...");
        
        auto gicp = pcl::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
        gicp->setMaximumIterations(100);
        gicp->setTransformationEpsilon(1e-8);
        gicp->setEuclideanFitnessEpsilon(1e-6);
        
        gicp->setInputTarget(target_cloud_);
        gicp->setInputSource(downsampled_source);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        // --- 核心：使用初始位姿进行配准 ---
        gicp->align(*aligned_cloud, initial_transform_);

        double fit_score = gicp->getFitnessScore();
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "配准中... Score: %.4f", fit_score);

        if (fit_score < 0.2) {
            has_aligned_ = true;
            transform = gicp->getFinalTransformation();
            
            RCLCPP_INFO(this->get_logger(), "✅ 配准成功! 位姿已锁定。");
            
            // --- 输出结果 ---
            float x = transform(0, 3);
            float y = transform(1, 3);
            float z = transform(2, 3);
            Eigen::Affine3f affine(transform);
            Eigen::Vector3f rpy = affine.rotation().eulerAngles(2, 1, 0); 
            
            RCLCPP_INFO(this->get_logger(), "********************************************");
            RCLCPP_INFO(this->get_logger(), "最终结合位姿 (Result):");
            RCLCPP_INFO(this->get_logger(), "XYZ: [%.3f, %.3f, %.3f]", x, y, z);
            RCLCPP_INFO(this->get_logger(), "RPY: [%.3f, %.3f, %.3f]", rpy[2], rpy[1], rpy[0]);
            RCLCPP_INFO(this->get_logger(), "********************************************");
        }
        
        // 如果还没锁定，发布初始位姿；锁定后发布最终位姿
        publishTF(has_aligned_ ? transform : initial_transform_, msg->header.stamp);
    }

    void publishTF(const Eigen::Matrix4f& transform, const builtin_interfaces::msg::Time& stamp)
    {
        geometry_msgs::msg::TransformStamped ts;
        ts.header.stamp = stamp;
        ts.header.frame_id = "rm_frame";
        ts.child_frame_id = "livox_frame";
        
        ts.transform.translation.x = transform(0, 3);
        ts.transform.translation.y = transform(1, 3);
        ts.transform.translation.z = transform(2, 3);
        
        Eigen::Quaternionf q(transform.block<3, 3>(0, 0));
        ts.transform.rotation.x = q.x();
        ts.transform.rotation.y = q.y();
        ts.transform.rotation.z = q.z();
        ts.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(ts);
    }

    bool has_aligned_ = false;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f initial_transform_ = Eigen::Matrix4f::Identity();

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    int accumulate_time = 30;
    
    double gridSizeDegrees = 0.1;
    std::map<std::pair<int, int>, Grid> gridMap;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr original_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
} // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)