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
        std::string target_pcd_file = "config/RM2025.pcd";
        // 从pcd读取场地点云
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(target_pcd_file, *target_cloud_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load %s",
                         target_pcd_file.c_str());
            return;
        }

        subscription_ =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10,
                std::bind(&Localization::callback, this,
                          std::placeholders::_1));

        // 发布场地点云到 /livox/map 话题
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/livox/map", 10);
        filter_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/filter_map", 10);

        //为map_pub设置计时器
        timer_ =
            this->create_wall_timer(std::chrono::seconds(10), [this]() {
                sensor_msgs::msg::PointCloud2 target_msg;
                pcl::toROSMsg(*target_cloud_, target_msg);
                target_msg.header.frame_id = "rm_frame";
                publisher_->publish(target_msg);
            });

        // 初始化TF广播
        tf_broadcaster_ =
            std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    // 回调函数：接收激光雷达原始点云消息（/livox/lidar话题）
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 判断是否已完成配准（has_aligned_为false时执行配准逻辑，true时仅发布TF）
        if (!has_aligned_) {
            // 1. 将ROS点云消息转换为PCL格式（PointXYZ类型，便于PCL库处理）
            pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(
                new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(*msg, *source_cloud);

            // 2. 累积多帧点云（共累积20帧，提高配准稳定性）
            if (accumulated_clouds_.size() <
                accumulate_time) {  // 若累积帧数不足20
                accumulated_clouds_.push_back(
                    source_cloud);  // 添加当前帧到累积列表
                return;  // 退出回调，等待累积足够帧数
            } else {     // 若累积帧数达到20
                accumulated_clouds_.erase(
                    accumulated_clouds_.begin());  // 移除最早的一帧
                accumulated_clouds_.push_back(
                    source_cloud);  // 添加当前帧（保持窗口大小为20）
            }

            // 3.
            // 构建栅格地图：每个栅格保留最远点（过滤近点噪声，保留环境结构）
            // 遍历累积的每帧点云
            for (auto accumulated_cloud : accumulated_clouds_) {
                // 遍历当前帧的每个点
                for (const auto& point : *accumulated_cloud) {
                    // 计算点的方位角（xy平面内与x轴夹角，单位弧度）
                    double azimuth = std::atan2(point.y, point.x);
                    // 计算点的仰角（与xy平面的夹角，单位弧度）
                    double elevation =
                        std::atan2(point.z, std::sqrt(point.x * point.x +
                                                      point.y * point.y));

                    // 将角度转换为栅格索引（栅格大小为0.1°×0.1°）
                    int azimuthIndex = static_cast<int>(
                        floor(azimuth * 180.0 / M_PI /
                              gridSizeDegrees));  // 弧度→度→栅格索引
                    int elevationIndex = static_cast<int>(
                        floor(elevation * 180.0 / M_PI / gridSizeDegrees));

                    // 计算点到原点的距离（用于筛选最远点）
                    double distance =
                        std::sqrt(point.x * point.x + point.y * point.y +
                                  point.z * point.z);

                    // 更新栅格地图：每个栅格仅保留距离最远的点（更可能是环境固定结构）
                    auto& grid = gridMap[std::make_pair(
                        azimuthIndex, elevationIndex)];  // 获取当前栅格
                    if (distance >
                        grid.maxDistance) {  // 若当前点比栅格中已存点更远
                        grid.farthestPoint = point;   // 更新为当前点
                        grid.maxDistance = distance;  // 更新最大距离
                    }
                }
            }

            // 4. 从栅格地图提取点云（仅保留有效栅格的最远点）
            pcl::PointCloud<pcl::PointXYZ>::Ptr result(
                new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& item : gridMap) {  // 遍历栅格地图的每个条目
                if (item.second.maxDistance >
                    0.0) {  // 过滤无效栅格（无点的栅格）
                    result->push_back(
                        item.second.farthestPoint);  // 添加最远点到结果点云
                }
            }

            // 5. 过滤感兴趣区域（ROI）：仅保留x∈(5,30)、y∈(-10,8)、z<7的点
            // 目的：去除超出场景范围的噪声点，减少计算量
            pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(
                new pcl::PointCloud<pcl::PointXYZ>());
            for (auto point : result->points) {
                if (point.x > 5 && point.x < 30 && point.y > -10 &&
                    point.y < 8 && point.z < 7) {
                    final_cloud->push_back(point);  // 保留ROI内的点
                }
            }

            // 6. 下采样（体素网格滤波）：减少点数量，加速配准计算
            pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(
                new pcl::PointCloud<pcl::PointXYZ>());
            pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;  // 体素网格过滤器
            voxelgrid.setLeafSize(0.1f, 0.1f,
                                  0.1f);  // 体素大小：0.1m×0.1m×0.1m

            // 对目标点云（地图点云）下采样
            voxelgrid.setInputCloud(
                target_cloud_);  // 输入：从PCD加载的场地点云
            voxelgrid.filter(*downsampled);  // 执行下采样
            *target_cloud_ = *downsampled;  // 更新目标点云为下采样结果

            // 对源点云（累积+栅格处理后的点云）下采样
            voxelgrid.setInputCloud(final_cloud);  // 输入：ROI过滤后的点云
            voxelgrid.filter(*downsampled);  // 执行下采样
            source_cloud = downsampled;  // 更新源点云为下采样结果

            // 7. 发布过滤后的源点云（用于可视化调试）
            sensor_msgs::msg::PointCloud2 filter_msg;
            pcl::toROSMsg(*source_cloud, filter_msg);  // PCL点云→ROS消息
            filter_msg.header.frame_id =
                "livox_frame";  // 坐标系：激光雷达坐标系
            filter_publisher_->publish(
                filter_msg);  // 发布到/filter_map话题

            // 8.
            // 执行GICP点云配准（广义迭代最近点算法，用于计算激光雷达与地图的位姿变换）
            std::cout << "--- pcl::GICP ---" << std::endl;
            // 创建GICP配准器（相比ICP，对初始位姿误差更鲁棒）
            boost::shared_ptr<pcl::GeneralizedIterativeClosestPoint<
                pcl::PointXYZ, pcl::PointXYZ>>
                gicp(new pcl::GeneralizedIterativeClosestPoint<
                     pcl::PointXYZ, pcl::PointXYZ>());
            // 调用align函数执行配准，返回变换矩阵（source→target的坐标变换）
            transform = align(gicp, target_cloud_, source_cloud);
        }

        // 9.
        // 发布TF变换（无论是否完成配准，均发布激光雷达坐标系到地图坐标系的变换）
        publishTF(transform);
    }

    pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Matrix4
    align(boost::shared_ptr<pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>>
                                                     registration,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
    {
        registration->setInputTarget(target_cloud);
        registration->setInputSource(source_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
            new pcl::PointCloud<pcl::PointXYZ>());
        auto t1 = std::chrono::system_clock::now();
        registration->align(*aligned);
        auto t2 = std::chrono::system_clock::now();
        // std::cout << "calib time   : " <<
        // std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
        // t1).count() << "[msec]" << std::endl;
        RCLCPP_WARN(this->get_logger(), "calib result : %f",
                    registration->getFitnessScore());

        if (registration->getFitnessScore() < 0.2) {
            has_aligned_ = true;
        }

        //打印变换矩阵
        return registration->getFinalTransformation();
    }

    void publishTF(const Eigen::Matrix4f& transform)
    {
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->now();
        transform_stamped.header.frame_id = "rm_frame";
        transform_stamped.child_frame_id = "livox_frame";
        transform_stamped.transform.translation.x = transform(0, 3);
        transform_stamped.transform.translation.y = transform(1, 3);
        transform_stamped.transform.translation.z = transform(2, 3);
        Eigen::Matrix3f    rotation = transform.block<3, 3>(0, 0);
        Eigen::Quaternionf q(rotation);
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(transform_stamped);
    }

    bool                                has_aligned_ = false;
    Eigen::Matrix4f                     transform;
    std::string                         target_pcd_file_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        subscription_;
    //创建一个数组用来积分点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr              accumulated_cloud_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    int                                              accumulate_time = 20;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        filter_publisher_;
    //为map_pub设置计时器
    rclcpp::TimerBase::SharedPtr timer_;
    double                       gridSizeDegrees = 0.1;  // 0.1°*0.1°
    std::map<std::pair<int, int>, Grid> gridMap;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
}  // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)
