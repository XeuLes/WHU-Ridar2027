#ifndef RADAR_RESOLVE_H
#define RADAR_RESOLVE_H

#include <memory>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "radar_utils.h"
#include "vision_interface/msg/detect_result.hpp"
#include "vision_interface/msg/match_info.hpp"

namespace tdt_radar {

class Resolve final : public rclcpp::Node {
public:
    explicit Resolve(const rclcpp::NodeOptions& options);

    ~Resolve() override;

    void callback(const std::shared_ptr<geometry_msgs::msg::Vector3> msg);
    void DetectCallback(
        const vision_interface::msg::DetectResult::SharedPtr msg);
    void MatchInfoCallback(
        const vision_interface::msg::MatchInfo::SharedPtr msg);

    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr point_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;

    parser* parser_{nullptr};

    cv::Mat minimap;
    int     markers[6]{};
    int16_t match_time{};
    uint8_t robot_hp[16]{};

private:
    // existing outputs
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pub;  // camera_point3D
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        pub_radar;  // resolve_result

    // UAV final output
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr
        pub_uav_positions_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pub_uav_intersection_debug_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pub_uav_cluster_debug_;

    // UAV lidar other cluster input
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        sub_uav_other_cloud_;

    // cache raw lidar_other points
    std::mutex   uav_other_mutex_;
    std::vector<pcl::PointXYZ> uav_other_points_;
    rclcpp::Time uav_other_stamp_{0, 0, RCL_ROS_TIME};

    // parameters
    bool is_test_{false};

    double uav_default_out_x_{0.0};
    double uav_default_out_y_{0.0};
    double uav_default_out_z_{0.0};

    int    uav_other_timeout_ms_{500};
    double uav_default_z_{1.95};
    double uav_huber_delta_xy_{0.8};
    double uav_w_vision_xy_{1.0};
    double uav_w_lidar_xy_{0.7};
    double uav_assign_gate_m_{3.0};
    double uav_bbox_roi_x_ratio_{0.30};
    double uav_bbox_roi_y_ratio_{0.00};
    double uav_bbox_roi_w_ratio_{0.40};
    double uav_bbox_roi_h_ratio_{0.40};
    double uav_proj_pixel_margin_{6.0};
    double uav_point_min_z_{1.2};
    double uav_point_max_z_{4.5};
    double uav_cluster_tolerance_{0.60};
    int    uav_cluster_min_size_{6};
    int    uav_cluster_max_size_{2000};
    double uav_cluster_score_dist_weight_{0.02};
    double uav_cluster_score_size_weight_{0.08};

    // 3.4 z-plane filter params (per UAV: 0=blue, 1=red)
    double uav_z_ema_alpha_{0.90};
    double uav_z_jump_max_{0.80};
    bool   uav_z_filt_inited_[2]{false, false};
    double uav_z_plane_filt_[2]{0.0, 0.0};

    // 2.1A KF (per UAV)
    bool         uav_kf_inited_[2]{false, false};
    rclcpp::Time uav_kf_stamp_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                  rclcpp::Time(0, 0, RCL_ROS_TIME)};
    rclcpp::Time uav_kf_last_update_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                        rclcpp::Time(0, 0, RCL_ROS_TIME)};

    // state [x y z vx vy vz]
    double uav_x_[2]{0.0, 0.0};
    double uav_y_[2]{0.0, 0.0};
    double uav_z_[2]{0.0, 0.0};
    double uav_vx_[2]{0.0, 0.0};
    double uav_vy_[2]{0.0, 0.0};
    double uav_vz_[2]{0.0, 0.0};
    double uav_P_[2][6][6]{};  // covariance

    // KF params
    int    uav_lost_timeout_ms_{500};
    double uav_q_pos_{2.0};
    double uav_q_vel_{5.0};
    double uav_r_meas_xy_{0.25};
    double uav_r_meas_z_{0.16};
    double uav_gate_maha_{16.27};

    // callbacks
    void UavOtherCloudCallback(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // helpers
    float GetFilteredZPlane(int idx, double z_raw);
    cv::Rect2f BuildUavRoiFromBbox(float x, float y, float w, float h) const;
    bool SolveUavFrustumMeasurement(
        const cv::Rect2f& roi,
        const std::vector<pcl::PointXYZ>& points,
        double& out_x,
        double& out_y,
        double& out_z,
        std::vector<cv::Point2f>* debug_uvs = nullptr,
        std::vector<cv::Point2f>* selected_cluster_uvs = nullptr) const;
    void  UavKalmanReset(int idx);
    void  UavKalmanInit(int idx, double x, double y, double z,
                        const rclcpp::Time& stamp);
    void  UavKalmanPredict(int idx, const rclcpp::Time& stamp);
    void  UavKalmanUpdate(int idx, double mx, double my, double mz);
};



}  // namespace tdt_radar

#endif
