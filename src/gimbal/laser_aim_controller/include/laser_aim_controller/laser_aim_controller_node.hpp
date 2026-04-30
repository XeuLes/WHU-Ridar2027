#pragma once

#include <Eigen/Core>

#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <auto_aim_fast/msg/aim_target.hpp>

#include <cstdint>
#include <mutex>
#include <string>

#include "laser_aim_controller/angle_solver.hpp"
#include "laser_aim_controller/laser_geometry.hpp"

namespace laser_aim_controller
{

class LaserAimControllerNode : public rclcpp::Node
{
public:
  explicit LaserAimControllerNode(const rclcpp::NodeOptions & options);

private:
  struct GimbalState
  {
    double yaw_deg{0.0};
    double pitch_deg{0.0};
    bool valid{false};
  };

  struct TargetObservation
  {
    bool received{false};
    bool has_target{false};
    bool has_pose{false};

    double u{0.0};
    double v{0.0};

    double tx{0.0};
    double ty{0.0};
    double tz{0.0};
    double distance{0.0};

    rclcpp::Time rx_time{0, 0, RCL_ROS_TIME};
    uint64_t seq{0};
  };

  enum class ControlMode
  {
    HOLD = 0,
    FINE_TRACK,
    PNP_ONLY_TRACK,
    LOST_RECOVER
  };

  void declareAndLoadParameters();

  bool isFinite3(double x, double y, double z) const;
  double clampValue(double v, double vmin, double vmax) const;
  double rad2deg(double rad) const;

  double interpolateByErrorPx(
    double err_px,
    double far_v,
    double mid_v,
    double near_v,
    double very_near_v,
    double ultra_near_v) const;

  double averageFramesToAlpha(double avg_frames) const;

  double computeVelocityLimitedAxisCommand(
    double desired_pos_deg,
    double measured_pos_deg,
    double & state_vel_deg_s,
    double dt_s,
    double max_vel_deg_s,
    double max_acc_deg_s2,
    double stop_error_deg) const;

  bool computePnpIncrement(
    const TargetObservation & obs,
    CommandAngles & inc_pnp,
    Eigen::Vector3d * target_from_laser) const;

  bool computePixelCorrection(
    double laser_pixel_u,
    double laser_pixel_v,
    double & out_yaw_deg,
    double & out_pitch_deg,
    double & pixel_err_px,
    double & pixel_err_yaw_deg,
    double & pixel_err_pitch_deg,
    double & zone_scale,
    double & zone_step_max_deg,
    double & filter_alpha,
    std::string & zone_name) const;

  double depthCompensatedLaserPixelV(const TargetObservation & obs, bool pnp_valid) const;

  void applyManualOffset(double & yaw_deg, double & pitch_deg) const;
  void publishCommand(double yaw_deg, double pitch_deg);

  void resetPixelFilter();
  void resetControlStateTo(double yaw_deg, double pitch_deg);
  const char * modeToString(ControlMode mode) const;

  void targetCallback(const auto_aim_fast::msg::AimTarget::SharedPtr msg);
  void gimbalAnglesCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void quaternionCallback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg);
  void roughCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void manualCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void controlLoop();

private:
  std::string target_topic_;
  std::string gimbal_angles_topic_;
  std::string gimbal_quaternion_topic_;
  std::string rough_command_topic_;
  std::string manual_command_topic_;
  std::string command_topic_;
  std::string gimbal_angle_order_;
  std::string increment_mode_;

  bool publish_debug_{true};
  bool enable_output_filter_{false};
  bool fallback_to_rough_when_no_target_{true};
  bool enable_pnp_assist_{true};

  double output_filter_alpha_{0.15};
  double min_valid_z_m_{0.05};
  double control_rate_hz_{100.0};
  double target_timeout_s_{0.12};
  double feedback_snap_deg_{1.2};
  double lost_auto_hold_timeout_s_{2.0};

  double fine_track_max_vel_deg_s_{5.0};
  double fine_track_max_acc_deg_s2_{60.0};
  double recover_max_vel_deg_s_{8.0};
  double recover_max_acc_deg_s2_{90.0};
  double stop_error_deg_{0.02};

  double pnp_exceed_threshold_deg_{5.0};
  double pnp_assist_gain_{0.18};
  double pnp_assist_max_deg_{0.05};
  double pnp_assist_lpf_alpha_{0.18};
  double pnp_assist_valid_inc_deg_{2.0};

  double fine_single_step_max_abs_deg_{0.30};

  double camera_fx_{1.0};
  double camera_fy_{1.0};
  double camera_cx_{0.0};
  double camera_cy_{0.0};

  double laser_pixel_u_{923.0};
  double laser_pixel_v_{368.0};
  bool laser_pixel_v_depth_comp_enable_{true};
  double laser_pixel_v_ref_distance_m_{13.5};
  double laser_pixel_v_ref_px_{368.0};
  double laser_pixel_v_slope_px_per_m_{5.0};
  double laser_pixel_v_min_px_{330.0};
  double laser_pixel_v_max_px_{430.0};

  double pixel_far_error_px_{300.0};
  double pixel_near_error_px_{100.0};
  double pixel_very_near_error_px_{80.0};
  double pixel_ultra_near_error_px_{40.0};

  int pixel_far_avg_frames_{3};
  int pixel_mid_avg_frames_{2};
  int pixel_near_avg_frames_{1};
  int pixel_very_near_avg_frames_{1};
  int pixel_ultra_near_avg_frames_{1};

  int pixel_far_cmd_interval_ms_{0};
  int pixel_mid_cmd_interval_ms_{40};
  int pixel_near_cmd_interval_ms_{100};
  int pixel_very_near_cmd_interval_ms_{160};
  int pixel_ultra_near_cmd_interval_ms_{260};

  double pixel_far_scale_{1.00};
  double pixel_mid_scale_{0.65};
  double pixel_near_scale_{0.35};
  double pixel_very_near_scale_{0.20};
  double pixel_ultra_near_scale_{0.12};

  double pixel_far_step_max_deg_{0.50};
  double pixel_mid_step_max_deg_{0.30};
  double pixel_near_step_max_deg_{0.10};
  double pixel_very_near_step_max_deg_{0.10};
  double pixel_ultra_near_step_max_deg_{0.05};

  double pixel_deadzone_deg_{0.02};
  double pixel_tolerance_yaw_deg_{0.02};
  double pixel_tolerance_pitch_deg_{0.02};

  double laser_offset_x_{0.0};
  double laser_offset_y_{-0.05};
  double laser_offset_z_{0.0};

  LaserGeometry laser_geometry_;
  AngleSolver angle_solver_;

  mutable std::mutex data_mutex_;
  TargetObservation latest_target_;
  GimbalState current_gimbal_;

  bool has_rough_cmd_{false};
  double rough_yaw_deg_{0.0};
  double rough_pitch_deg_{0.0};
  bool rough_target_observed_{true};

  bool has_manual_cmd_{false};
  double manual_yaw_deg_{0.0};
  double manual_pitch_deg_{0.0};

  bool has_filtered_pixel_{false};
  double filtered_target_u_{0.0};
  double filtered_target_v_{0.0};
  uint64_t last_filtered_target_seq_{0};

  bool controller_state_initialized_{false};
  bool has_auto_command_reference_{false};
  double command_vel_yaw_deg_s_{0.0};
  double command_vel_pitch_deg_s_{0.0};
  double last_auto_cmd_yaw_deg_{0.0};
  double last_auto_cmd_pitch_deg_{0.0};

  bool has_last_output_{false};
  double last_yaw_out_deg_{0.0};
  double last_pitch_out_deg_{0.0};

  bool has_hold_auto_cmd_{false};
  double hold_auto_yaw_deg_{0.0};
  double hold_auto_pitch_deg_{0.0};
  bool has_tracked_target_auto_cmd_{false};
  bool lost_auto_hold_active_{false};
  bool lost_auto_hold_expired_logged_{false};
  rclcpp::Time lost_auto_hold_start_{0, 0, RCL_ROS_TIME};

  double pnp_bias_yaw_deg_{0.0};
  double pnp_bias_pitch_deg_{0.0};

  std::string search_stage_mode_{"manual"};

  double scan_window_half_yaw_deg_{1.5};
  double scan_window_half_pitch_deg_{1.5};
  double scan_period_s_{4.0};
  double scan_phase_x_rad_{0.0};
  double scan_phase_y_rad_{1.57079632679};

  double wide_scan_window_half_yaw_deg_{10.0};
  double wide_scan_window_half_pitch_deg_{30.0};
  double wide_scan_speed_scale_{3.0};
  double wide_scan_pitch_max_speed_deg_s_{10.0};
  double wide_scan_fov_overlap_{0.65};
  int wide_scan_max_pitch_cycles_per_yaw_sweep_{64};
  double wide_scan_camera_image_width_px_{1624.0};
  double wide_scan_camera_image_height_px_{1240.0};

  ControlMode mode_{ControlMode::HOLD};
  ControlMode last_mode_{ControlMode::HOLD};

  uint64_t target_seq_counter_{0};
  rclcpp::Time last_control_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<auto_aim_fast::msg::AimTarget>::SharedPtr target_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr gimbal_angles_sub_;
  rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr quaternion_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rough_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manual_cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace laser_aim_controller
