#include <pcl/kdtree/kdtree.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace tdt_radar {
class Cluster : public rclcpp::Node {
public:
    Cluster(const rclcpp::NodeOptions& node_options);
    ~Cluster() {}

private:
    // 原始数据流：订阅/livox/lidar_dynamic，发布/livox/lidar_cluster
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;

    // 原始数据流回调函数（处理/livox/lidar_dynamic）
    void
    callback_dynamic(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // 通用聚类处理函数（复用聚类逻辑，通过发布者参数区分输出）
    void process_cluster(
        const sensor_msgs::msg::PointCloud2::SharedPtr              msg,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);

    // 原始头文件中已有的点云累积容器
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
};
}  // namespace tdt_radar
