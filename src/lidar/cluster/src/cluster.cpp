#include "cluster.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/duration.hpp>
namespace tdt_radar {

Cluster::Cluster(const rclcpp::NodeOptions& node_options)
    : Node("cluster_node", node_options)
{
    RCLCPP_INFO(this->get_logger(), "cluster_node start");
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10,
        std::bind(&Cluster::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10);
}

void Cluster::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 记录回调函数开始时间（用于计算处理耗时）
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();

    // 创建PCL点云对象（存储转换后的点云数据）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // 将ROS消息类型的点云（sensor_msgs::msg::PointCloud2）转换为PCL格式（pcl::PointCloud<pcl::PointXYZ>）
    pcl::fromROSMsg(*msg, *cloud);

    // 如果点云为空，则直接返回（避免后续无效计算）
    if (cloud->empty()) {
        return;
    }

    // 创建KD-Tree对象（用于聚类时的近邻搜索）
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    // 为KD-Tree设置输入点云（后续聚类将基于此点云进行搜索）
    tree->setInputCloud(cloud);

    // （未使用）记录当前系统时间（可能为预留的时间戳逻辑）
    auto time = std::chrono::system_clock::now();

    // 创建欧式聚类对象（用于实现基于距离的聚类）
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    // 设置聚类容忍度（两点间距离小于0.25米时视为同一聚类）
    ec.setClusterTolerance(0.25);

    // 设置最小聚类点数（过滤掉点数少于5的噪声聚类）
    ec.setMinClusterSize(5);

    // 设置最大聚类点数（过滤掉点数过多的无效聚类，如大面积干扰）
    ec.setMaxClusterSize(1000);

    // 设置聚类使用的搜索方法（此处为前面创建的KD-Tree）
    ec.setSearchMethod(tree);

    // 设置输入点云（待聚类的动态点云）
    ec.setInputCloud(cloud);

    // 执行聚类，结果存储在cluster_indices中（每个元素是一个聚类的点索引集合）
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    // 创建输出点云对象（存储每个聚类的中心点）
    pcl::PointCloud<pcl::PointXYZ>* out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // 遍历所有聚类结果
    for (auto it = cluster_indices.begin(); it != cluster_indices.end();
         ++it) {
        // 为当前聚类创建临时点云（存储该聚类的所有点）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
            new pcl::PointCloud<pcl::PointXYZ>);

        // 将聚类索引对应的点添加到临时点云中
        for (auto pit = it->indices.begin(); pit != it->indices.end();
             ++pit) {
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }

        // 设置临时点云的属性（宽度=点数量，高度=1表示无序点云，is_dense=True表示无无效点）
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 计算当前聚类的中心点（质心）
        pcl::PointXYZ move_point;
        for (auto point : cloud_cluster->points) {
            move_point.x += point.x;  // 累加x坐标
            move_point.y += point.y;  // 累加y坐标
            move_point.z += point.z;  // 累加z坐标
        }
        move_point.x /= cloud_cluster->points.size();  // 求x坐标平均值
        move_point.y /= cloud_cluster->points.size();  // 求y坐标平均值
        move_point.z /= cloud_cluster->points.size();  // 求z坐标平均值

        // 将中心点添加到输出点云中
        out_cloud->points.push_back(move_point);
    }

    // 创建ROS消息类型的点云（用于发布聚类结果）
    sensor_msgs::msg::PointCloud2 output;

    // 将PCL格式的输出点云（中心点集合）转换为ROS消息格式
    pcl::toROSMsg(*out_cloud, output);

    // 设置输出点云的坐标系（与输入保持一致，统一为rm_frame）
    output.header.frame_id = "rm_frame";

    // 设置输出点云的时间戳（与输入点云保持一致，确保时间同步）
    output.header.stamp = msg->header.stamp;

    // 发布聚类结果（中心点云）到话题/livox/lidar_cluster
    pub_->publish(output);

    // 记录回调函数结束时间
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();

    // 打印聚类处理耗时（毫秒）
    RCLCPP_INFO(
        this->get_logger(), "Cluster callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Cluster)