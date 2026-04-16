#pragma once

#include <radar_decision_comm/msg/referee_robot_identity.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>
#include <vision_interface/msg/radar2_sentry.hpp>
#include <vision_interface/msg/radar_warn.hpp>
#include "filter_plus.h"
#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace tdt_radar {
class KalmanFilter : public rclcpp::Node {
public:
    KalmanFilter(const rclcpp::NodeOptions& node_options);
    ~KalmanFilter() {}

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        sub_detect_;
    rclcpp::Subscription<vision_interface::msg::RadarWarn>::SharedPtr
        sub_lidar_;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        sub_match_;
    rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr
        sub_referee_identity_;
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar_pub_;
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        radar_detect_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void detect_callback(
        const vision_interface::msg::DetectResult::SharedPtr msg);
    void
    lidar_callback(const vision_interface::msg::RadarWarn::SharedPtr msg);
    void
    match_callback(const vision_interface::msg::MatchInfo::SharedPtr msg);
    void referee_identity_callback(
        const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg);
    bool is_kf_available_now(const rclcpp::Time& now_time) const;
    int normalized_self_color(int raw_color) const;
    int effective_self_color() const;
    std::vector<Kalman_filter_plus>  KFs;
    vision_interface::msg::RadarWarn lidar_detect;
    vision_interface::msg::MatchInfo match_info;
    rclcpp::Time last_lidar_cluster_stamp_{0, 0, RCL_ROS_TIME};
    bool kf_available_{false};
    int kf_available_timeout_ms_{300};   // 可调
    bool use_match_info_color_{true};
    int self_color_override_{2};
    bool use_referee_identity_color_{true};
    int referee_identity_color_{2};
    bool has_referee_identity_color_{false};
};
}  // namespace tdt_radar
