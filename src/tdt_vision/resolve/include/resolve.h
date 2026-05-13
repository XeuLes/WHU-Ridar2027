#ifndef RADAR_RESOLVE_H
#define RADAR_RESOLVE_H

#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "radar_utils.h"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
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
    void RefereeIdentityCallback(
        const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg);

    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr point_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;
    rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr
        referee_identity_sub;

    parser* parser_{nullptr};

    cv::Mat minimap;
    int     markers[6]{};
    int16_t match_time{};
    int8_t  self_color_{-1};
    uint8_t robot_hp[16]{};

private:
    struct UavFrustumMeasurement {
        bool valid{false};
        bool frustum_coord_valid{false};
        double frustum_x{0.0};
        double frustum_y{0.0};
        double frustum_z{0.0};
        double cluster_x{0.0};
        double cluster_y{0.0};
        double cluster_z{0.0};
        double x{0.0};
        double y{0.0};
        double z{0.0};
        double cluster_z_median{0.0};
        double ray_dist_median{std::numeric_limits<double>::infinity()};
        double ray_dist_spread{std::numeric_limits<double>::infinity()};
        double depth_median{0.0};
        int cluster_size{0};
        int input_size{0};
        int z_valid_size{0};
        int roi_point_size{0};
        int cluster_count{0};
        std::vector<cv::Point2f> debug_uvs;
        std::vector<cv::Point2f> cluster_uvs;
    };

    enum class UavMeasurementSource : int {
        Frustum = 0,
        LidarOnly = 1,
        VisionZ = 2,
        DefaultPose = 3,
    };

    struct UavPointCloudMeasurement {
        bool valid{false};
        double x{0.0};
        double y{0.0};
        double z{0.0};
        double cluster_z_median{0.0};
        double seed_distance{std::numeric_limits<double>::infinity()};
        double cluster_spread{std::numeric_limits<double>::infinity()};
        int cluster_size{0};
        int input_size{0};
        int z_valid_size{0};
        int cluster_count{0};
    };

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
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr
        pub_uav_resolve_debug_status_;

    // UAV lidar other cluster input
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        sub_uav_other_cloud_;

    // cache raw lidar_other points
    std::mutex   uav_other_mutex_;
    std::vector<pcl::PointXYZ> uav_other_points_;
    rclcpp::Time uav_other_stamp_{0, 0, RCL_ROS_TIME};

    // parameters
    bool is_test_{false};
    double vision_field_x_margin_m_{0.5};
    double vision_field_y_margin_m_{0.5};
    bool uav_publish_default_output_{true};
    bool use_match_info_color_{true};
    bool use_referee_identity_color_{true};
    bool has_referee_identity_color_{false};
    bool uav_use_enemy_default_by_identity_{true};

    int8_t self_color_override_{-1};
    int8_t referee_identity_color_{-1};

    double uav_config_default_out_x_[2]{0.0, 0.0};
    double uav_config_default_out_y_[2]{0.0, 0.0};
    double uav_config_default_out_z_[2]{0.0, 0.0};

    double uav_enemy_default_x_{0.0};
    double uav_enemy_default_y_{0.0};
    double uav_enemy_default_z_{0.0};

    double uav_default_out_x_[2]{0.0, 0.0};
    double uav_default_out_y_[2]{0.0, 0.0};
    double uav_default_out_z_[2]{0.0, 0.0};

    int8_t NormalizeSelfColor(int raw_color, int8_t fallback) const;
    void RefreshEffectiveSelfColorFromFallback();
    void RefreshUavDefaultOutputs();
    bool HasEnemyDefaultPose() const;

    int    uav_other_timeout_ms_{500};
    double uav_default_z_{1.95};
    double uav_huber_delta_xy_{0.8};
    double uav_w_vision_xy_{1.0};
    double uav_w_lidar_xy_{0.7};
    double uav_assign_gate_m_{3.0};
    double uav_bbox_roi_x_ratio_{0.00};
    double uav_bbox_roi_y_ratio_{0.00};
    double uav_bbox_roi_w_ratio_{1.00};
    double uav_bbox_roi_h_ratio_{1.00};
    double uav_proj_pixel_margin_{6.0};
    double uav_point_min_z_{1.2};
    double uav_point_max_z_{4.5};
    double uav_cluster_tolerance_{0.60};
    int    uav_cluster_min_size_{6};
    int    uav_cluster_max_size_{2000};
    double uav_cluster_score_dist_weight_{0.02};  // legacy param, kept for compatibility
    double uav_cluster_score_ray_weight_{1.0};
    double uav_cluster_score_spread_weight_{0.6};
    double uav_cluster_score_size_weight_{0.08};
    double uav_center_ray_min_depth_{0.2};
    double uav_rep_ray_weight_{1.0};
    double uav_rep_depth_weight_{0.8};
    double uav_quality_ray_sigma_{0.45};
    double uav_quality_spread_sigma_{0.25};
    double uav_quality_size_ref_{12.0};
    double uav_quality_measurement_threshold_{0.60};
    double uav_quality_z_prior_threshold_{0.75};
    int    uav_degrade_l0_to_l1_ms_{200};
    int    uav_degrade_l1_to_l2_ms_{250};
    int    uav_degrade_l2_to_l3_ms_{500};
    double uav_lidar_only_assign_gate_m_{3.0};
    double uav_lidar_only_cluster_spread_max_m_{1.2};
    int    uav_lidar_only_seed_timeout_ms_{1500};
    bool   uav_lidar_only_use_half_roi_{true};
    double uav_lidar_only_half_split_y_{7.5};
    double uav_world_to_field_y_offset_{15.0};
    double uav_field_min_x_{0.0};
    double uav_field_max_x_{28.0};
    double uav_field_min_y_{0.0};
    double uav_field_max_y_{15.0};
    double uav_field_margin_m_{0.20};
    double uav_z_prior_alpha_{0.90};
    int    uav_z_prior_window_ms_{1200};
    int    uav_z_prior_update_min_interval_ms_{250};

    // 3.4 z-plane filter params (per UAV: 0=blue, 1=red)
    double uav_z_ema_alpha_{0.90};
    double uav_z_jump_max_{0.80};
    bool   uav_z_filt_inited_[2]{false, false};
    double uav_z_plane_filt_[2]{0.0, 0.0};
    bool   uav_z_prior_inited_[2]{false, false};
    double uav_z_prior_[2]{0.0, 0.0};
    rclcpp::Time uav_z_prior_last_update_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                             rclcpp::Time(0, 0, RCL_ROS_TIME)};
    std::deque<std::pair<rclcpp::Time, double>> uav_z_prior_samples_[2];

    // 2.1A KF (per UAV)
    bool         uav_kf_inited_[2]{false, false};
    rclcpp::Time uav_kf_stamp_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                  rclcpp::Time(0, 0, RCL_ROS_TIME)};
    rclcpp::Time uav_kf_last_update_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                        rclcpp::Time(0, 0, RCL_ROS_TIME)};

    UavMeasurementSource uav_active_source_[2]{
        UavMeasurementSource::DefaultPose,
        UavMeasurementSource::DefaultPose};
    rclcpp::Time uav_source_last_update_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
                                            rclcpp::Time(0, 0, RCL_ROS_TIME)};
    bool uav_last_measurement_valid_[2]{false, false};
    double uav_last_measurement_x_[2]{0.0, 0.0};
    double uav_last_measurement_y_[2]{0.0, 0.0};
    double uav_last_measurement_z_[2]{0.0, 0.0};
    rclcpp::Time uav_last_measurement_stamp_[2]{rclcpp::Time(0, 0, RCL_ROS_TIME),
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
    double ClampUavZ(double z) const;
    void InferUavFieldTransform();
    cv::Point3f UavFieldToCameraWorld(const pcl::PointXYZ& point) const;
    cv::Vec3d UavCameraWorldToField(const cv::Vec3d& point) const;
    bool IsUavFieldPointInside(double x, double y) const;
    bool IntersectUavPixelWithZ(const cv::Point2f& pixel,
                                double z_plane,
                                cv::Vec3d& point_field,
                                cv::Vec3d* point_camera_world = nullptr) const;
    bool BuildUavCenterRay(const cv::Point2f& pixel,
                           cv::Vec3d& origin_world,
                           cv::Vec3d& dir_world) const;
    double EvaluateUavMeasurementQuality(double conf,
                                         const UavFrustumMeasurement& meas) const;
    double GetUavFallbackZ(int idx) const;
    void UpdateUavZPrior(int idx, double z_raw, const rclcpp::Time& stamp);
    const char* UavSourceName(UavMeasurementSource source) const;
    int UavSourcePriority(UavMeasurementSource source) const;
    int UavDegradeHoldMs(UavMeasurementSource from,
                         UavMeasurementSource to) const;
    bool BuildUavLidarOnlySeed(int idx,
                               const rclcpp::Time& stamp,
                               cv::Vec3d& seed) const;
    bool IsUavLidarOnlyClusterAllowedForSide(int idx, double y) const;
    bool SolveUavFrustumMeasurement(
        const cv::Rect2f& roi,
        const std::vector<pcl::PointXYZ>& points,
        UavFrustumMeasurement& out) const;
    bool SolveUavLidarOnlyMeasurement(
        int idx,
        const std::vector<pcl::PointXYZ>& points,
        const rclcpp::Time& stamp,
        UavPointCloudMeasurement& out) const;
    void  UavKalmanReset(int idx);
    void  UavKalmanInit(int idx, double x, double y, double z,
                        const rclcpp::Time& stamp);
    void  UavKalmanPredict(int idx, const rclcpp::Time& stamp);
    void  UavKalmanUpdate(int idx, double mx, double my, double mz);
};



}  // namespace tdt_radar

#endif
