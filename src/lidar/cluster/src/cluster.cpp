#include "cluster.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/duration.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace tdt_radar {

// 构造函数：初始化订阅者和发布者
Cluster::Cluster(const rclcpp::NodeOptions& node_options)
    : Node("cluster_node", node_options)
{
    RCLCPP_INFO(this->get_logger(), "cluster_node start");

    // 原始数据流：订阅/livox/lidar_dynamic，发布到/livox/lidar_cluster
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10,
        std::bind(&Cluster::callback_dynamic, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10);
}

// 原始数据流回调函数：处理/livox/lidar_dynamic
void Cluster::callback_dynamic(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    process_cluster(msg, pub_);  // 调用通用聚类处理函数
}

// 通用聚类处理函数：复用聚类算法逻辑
void Cluster::process_cluster(
    const sensor_msgs::msg::PointCloud2::SharedPtr              msg,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
{
    // 记录处理开始时间
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();

    // 转换ROS点云到PCL格式
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // 空点云直接返回
    if (cloud->empty()) {
        return;
    }

    // 创建KD-Tree用于近邻搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // 欧式聚类参数设置
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.75);  // 聚类距离阈值
    ec.setMinClusterSize(5);      // 最小聚类点数
    ec.setMaxClusterSize(1000);   // 最大聚类点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    // 执行聚类
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    // 计算每个聚类的中心点
    pcl::PointCloud<pcl::PointXYZ>* out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (int idx : indices.indices) {
            cloud_cluster->points.push_back(cloud->points[idx]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 计算中心点（质心）
        pcl::PointXYZ center;
        for (auto& pt : cloud_cluster->points) {
            center.x += pt.x;
            center.y += pt.y;
            center.z += pt.z;
        }
        center.x /= cloud_cluster->points.size();
        center.y /= cloud_cluster->points.size();
        center.z /= cloud_cluster->points.size();
        out_cloud->points.push_back(center);
    }

    // 转换为ROS消息并发布
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*out_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    pub->publish(output);

    // 计算并打印处理耗时
    // std::chrono::steady_clock::time_point t2 =
    //     std::chrono::steady_clock::now();
    // RCLCPP_INFO(
    //     this->get_logger(), "Cluster process time: %f ms",
    //     std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
    //             .count() /
    //         1000.0);
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Cluster)
