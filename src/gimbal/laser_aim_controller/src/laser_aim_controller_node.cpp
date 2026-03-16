#include "laser_aim_controller/laser_aim_controller_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laser_aim_controller
{

namespace
{
constexpr double kRadToDeg = 57.2957795130823208768;
constexpr double kPi = 3.14159265358979323846;
}

LaserAimControllerNode::LaserAimControllerNode(const rclcpp::NodeOptions & options)
: Node("laser_aim_controller_node", options)
{
  declareAndLoadParameters();

  target_sub_ = this->create_subscription<auto_aim_fast::msg::AimTarget>(
    target_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LaserAimControllerNode::targetCallback, this, std::placeholders::_1));

  gimbal_angles_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    gimbal_angles_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LaserAimControllerNode::gimbalAnglesCallback, this, std::placeholders::_1));

  quaternion_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
    gimbal_quaternion_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LaserAimControllerNode::quaternionCallback, this, std::placeholders::_1));

  rough_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    rough_command_topic_, rclcpp::QoS(10),
    std::bind(&LaserAimControllerNode::roughCommandCallback, this, std::placeholders::_1));

  manual_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    manual_command_topic_, rclcpp::QoS(10),
    std::bind(&LaserAimControllerNode::manualCommandCallback, this, std::placeholders::_1));

  cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(command_topic_, 10);

  const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, control_rate_hz_));
  control_timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&LaserAimControllerNode::controlLoop, this));

  RCLCPP_INFO(this->get_logger(), "laser_aim_controller_node started.");
  RCLCPP_INFO(this->get_logger(), "target_topic: %s", target_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_angles_topic: %s", gimbal_angles_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_quaternion_topic: %s", gimbal_quaternion_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "rough_command_topic: %s", rough_command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "manual_command_topic: %s", manual_command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "command_topic: %s", command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "increment_mode: %s", increment_mode_.c_str());
  RCLCPP_INFO(this->get_logger(), "control_rate_hz: %.1f", control_rate_hz_);
  RCLCPP_INFO(this->get_logger(), "gimbal_angle_order: %s", gimbal_angle_order_.c_str());
  RCLCPP_INFO(this->get_logger(), "axis convention: [yaw, pitch], yaw-left-positive, pitch-up-positive");
  RCLCPP_INFO(this->get_logger(), "fallback_to_rough_when_no_target: %s",
    fallback_to_rough_when_no_target_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "pixel-primary auto mode with pnp assist: %s",
    enable_pnp_assist_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "search_stage_mode: %s", search_stage_mode_.c_str());
  RCLCPP_INFO(this->get_logger(), "scan_window_half_yaw_deg: %.3f", scan_window_half_yaw_deg_);
  RCLCPP_INFO(this->get_logger(), "scan_window_half_pitch_deg: %.3f", scan_window_half_pitch_deg_);
  RCLCPP_INFO(this->get_logger(), "scan_period_s: %.3f", scan_period_s_);
}

void LaserAimControllerNode::declareAndLoadParameters()
{
  this->declare_parameter<std::string>("target_topic", "/auto_aim/target");
  this->declare_parameter<std::string>("gimbal_angles_topic", "/gimbal/angles_true");
  this->declare_parameter<std::string>("gimbal_quaternion_topic", "/gimbal/quaternion_true");
  this->declare_parameter<std::string>("rough_command_topic", "/gimbal/rough_command_angles");
  this->declare_parameter<std::string>("manual_command_topic", "/gimbal/command_angle_manual");
  this->declare_parameter<std::string>("command_topic", "/gimbal/command_angles");

  // 只允许 yaw_pitch
  this->declare_parameter<std::string>("gimbal_angle_order", "yaw_pitch");
  this->declare_parameter<std::string>("increment_mode", "auto");

  this->declare_parameter<bool>("publish_debug", true);
  this->declare_parameter<bool>("enable_output_filter", false);
  this->declare_parameter<bool>("fallback_to_rough_when_no_target", true);
  this->declare_parameter<bool>("enable_pnp_assist", true);

  // 搜索阶段链路选择：
  // manual: 有粗瞄=rough+manual；无粗瞄=hold+manual
  // scan  : 有粗瞄=rough附近扫描；无粗瞄=hold；manual不参与
  this->declare_parameter<std::string>("search_stage_mode", "scan");
  this->declare_parameter<double>("scan_window_half_yaw_deg", 1.5);     // 3度窗口 => 半窗1.5
  this->declare_parameter<double>("scan_window_half_pitch_deg", 1.5);   // 3度窗口 => 半窗1.5
  this->declare_parameter<double>("scan_period_s", 4.0);                // 总周期4s
  this->declare_parameter<double>("scan_phase_x_rad", 0.0);
  this->declare_parameter<double>("scan_phase_y_rad", 1.57079632679);   // pi/2

  this->declare_parameter<double>("output_filter_alpha", 0.15);
  this->declare_parameter<double>("min_valid_z_m", 0.05);
  this->declare_parameter<double>("control_rate_hz", 100.0);
  this->declare_parameter<double>("target_timeout_s", 0.12);
  this->declare_parameter<double>("feedback_snap_deg", 1.2);

  this->declare_parameter<double>("fine_track_max_vel_deg_s", 14.0);
  this->declare_parameter<double>("fine_track_max_acc_deg_s2", 320.0);
  this->declare_parameter<double>("recover_max_vel_deg_s", 70.0);
  this->declare_parameter<double>("recover_max_acc_deg_s2", 1200.0);
  this->declare_parameter<double>("stop_error_deg", 0.03);

  this->declare_parameter<double>("pnp_exceed_threshold_deg", 5.0);
  this->declare_parameter<double>("pnp_assist_gain", 0.18);
  this->declare_parameter<double>("pnp_assist_max_deg", 0.05);
  this->declare_parameter<double>("pnp_assist_lpf_alpha", 0.18);
  this->declare_parameter<double>("pnp_assist_valid_inc_deg", 2.0);

  this->declare_parameter<std::vector<double>>(
    "camera_offset_xyz", std::vector<double>{0.0, 0.0, 0.0});
  this->declare_parameter<std::vector<double>>(
    "laser_offset_xyz", std::vector<double>{0.0, -0.05, 0.0});

  this->declare_parameter<double>("yaw_inc_min_deg", -5.0);
  this->declare_parameter<double>("yaw_inc_max_deg", 5.0);
  this->declare_parameter<double>("pitch_inc_min_deg", -5.0);
  this->declare_parameter<double>("pitch_inc_max_deg", 5.0);
  this->declare_parameter<double>("fine_single_step_max_abs_deg", 2.0);

  this->declare_parameter<double>("camera_fx", 15389.868272);
  this->declare_parameter<double>("camera_fy", 15116.554824);
  this->declare_parameter<double>("camera_cx", 841.988257);
  this->declare_parameter<double>("camera_cy", 675.935439);
  this->declare_parameter<std::vector<double>>(
    "laser_pixel_uv", std::vector<double>{803.0, 137.0});

  this->declare_parameter<double>("pixel_far_error_px", 300.0);
  this->declare_parameter<double>("pixel_near_error_px", 100.0);
  this->declare_parameter<double>("pixel_very_near_error_px", 80.0);
  this->declare_parameter<double>("pixel_ultra_near_error_px", 40.0);

  this->declare_parameter<int>("pixel_far_avg_frames", 3);
  this->declare_parameter<int>("pixel_mid_avg_frames", 2);
  this->declare_parameter<int>("pixel_near_avg_frames", 1);
  this->declare_parameter<int>("pixel_very_near_avg_frames", 1);
  this->declare_parameter<int>("pixel_ultra_near_avg_frames", 1);

  this->declare_parameter<int>("pixel_far_cmd_interval_ms", 0);
  this->declare_parameter<int>("pixel_mid_cmd_interval_ms", 40);
  this->declare_parameter<int>("pixel_near_cmd_interval_ms", 100);
  this->declare_parameter<int>("pixel_very_near_cmd_interval_ms", 160);
  this->declare_parameter<int>("pixel_ultra_near_cmd_interval_ms", 260);

  this->declare_parameter<double>("pixel_far_scale", 1.00);
  this->declare_parameter<double>("pixel_mid_scale", 0.65);
  this->declare_parameter<double>("pixel_near_scale", 0.35);
  this->declare_parameter<double>("pixel_very_near_scale", 0.20);
  this->declare_parameter<double>("pixel_ultra_near_scale", 0.12);

  this->declare_parameter<double>("pixel_far_step_max_deg", 2.00);
  this->declare_parameter<double>("pixel_mid_step_max_deg", 1.00);
  this->declare_parameter<double>("pixel_near_step_max_deg", 0.50);
  this->declare_parameter<double>("pixel_very_near_step_max_deg", 0.10);
  this->declare_parameter<double>("pixel_ultra_near_step_max_deg", 0.05);

  this->declare_parameter<double>("pixel_deadzone_deg", 0.02);
  this->declare_parameter<double>("pixel_tolerance_yaw_deg", 0.02);
  this->declare_parameter<double>("pixel_tolerance_pitch_deg", 0.02);

  target_topic_ = this->get_parameter("target_topic").as_string();
  gimbal_angles_topic_ = this->get_parameter("gimbal_angles_topic").as_string();
  gimbal_quaternion_topic_ = this->get_parameter("gimbal_quaternion_topic").as_string();
  rough_command_topic_ = this->get_parameter("rough_command_topic").as_string();
  manual_command_topic_ = this->get_parameter("manual_command_topic").as_string();
  command_topic_ = this->get_parameter("command_topic").as_string();
  gimbal_angle_order_ = this->get_parameter("gimbal_angle_order").as_string();
  increment_mode_ = this->get_parameter("increment_mode").as_string();

  if (gimbal_angle_order_ != "yaw_pitch") {
    throw std::runtime_error(
      "laser_aim_controller only accepts gimbal_angle_order='yaw_pitch' "
      "(msg->data[0]=yaw, msg->data[1]=pitch).");
  }

  publish_debug_ = this->get_parameter("publish_debug").as_bool();
  enable_output_filter_ = this->get_parameter("enable_output_filter").as_bool();
  fallback_to_rough_when_no_target_ =
    this->get_parameter("fallback_to_rough_when_no_target").as_bool();
  enable_pnp_assist_ = this->get_parameter("enable_pnp_assist").as_bool();

  search_stage_mode_ = this->get_parameter("search_stage_mode").as_string();
  if (search_stage_mode_ != "manual" && search_stage_mode_ != "scan") {
    throw std::runtime_error(
      "search_stage_mode must be 'manual' or 'scan'");
  }
  scan_window_half_yaw_deg_ = this->get_parameter("scan_window_half_yaw_deg").as_double();
  scan_window_half_pitch_deg_ = this->get_parameter("scan_window_half_pitch_deg").as_double();
  scan_period_s_ = this->get_parameter("scan_period_s").as_double();
  scan_phase_x_rad_ = this->get_parameter("scan_phase_x_rad").as_double();
  scan_phase_y_rad_ = this->get_parameter("scan_phase_y_rad").as_double();

  output_filter_alpha_ = this->get_parameter("output_filter_alpha").as_double();
  min_valid_z_m_ = this->get_parameter("min_valid_z_m").as_double();
  control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();
  target_timeout_s_ = this->get_parameter("target_timeout_s").as_double();
  feedback_snap_deg_ = this->get_parameter("feedback_snap_deg").as_double();

  fine_track_max_vel_deg_s_ = this->get_parameter("fine_track_max_vel_deg_s").as_double();
  fine_track_max_acc_deg_s2_ = this->get_parameter("fine_track_max_acc_deg_s2").as_double();
  recover_max_vel_deg_s_ = this->get_parameter("recover_max_vel_deg_s").as_double();
  recover_max_acc_deg_s2_ = this->get_parameter("recover_max_acc_deg_s2").as_double();
  stop_error_deg_ = this->get_parameter("stop_error_deg").as_double();

  pnp_exceed_threshold_deg_ = this->get_parameter("pnp_exceed_threshold_deg").as_double();
  pnp_assist_gain_ = this->get_parameter("pnp_assist_gain").as_double();
  pnp_assist_max_deg_ = this->get_parameter("pnp_assist_max_deg").as_double();
  pnp_assist_lpf_alpha_ = this->get_parameter("pnp_assist_lpf_alpha").as_double();
  pnp_assist_valid_inc_deg_ = this->get_parameter("pnp_assist_valid_inc_deg").as_double();

  fine_single_step_max_abs_deg_ =
    this->get_parameter("fine_single_step_max_abs_deg").as_double();

  camera_fx_ = this->get_parameter("camera_fx").as_double();
  camera_fy_ = this->get_parameter("camera_fy").as_double();
  camera_cx_ = this->get_parameter("camera_cx").as_double();
  camera_cy_ = this->get_parameter("camera_cy").as_double();

  const auto laser_uv = this->get_parameter("laser_pixel_uv").as_double_array();
  if (laser_uv.size() >= 2) {
    laser_pixel_u_ = laser_uv[0];
    laser_pixel_v_ = laser_uv[1];
  }

  pixel_far_error_px_ = this->get_parameter("pixel_far_error_px").as_double();
  pixel_near_error_px_ = this->get_parameter("pixel_near_error_px").as_double();
  pixel_very_near_error_px_ = this->get_parameter("pixel_very_near_error_px").as_double();
  pixel_ultra_near_error_px_ = this->get_parameter("pixel_ultra_near_error_px").as_double();

  pixel_far_avg_frames_ = this->get_parameter("pixel_far_avg_frames").as_int();
  pixel_mid_avg_frames_ = this->get_parameter("pixel_mid_avg_frames").as_int();
  pixel_near_avg_frames_ = this->get_parameter("pixel_near_avg_frames").as_int();
  pixel_very_near_avg_frames_ = this->get_parameter("pixel_very_near_avg_frames").as_int();
  pixel_ultra_near_avg_frames_ = this->get_parameter("pixel_ultra_near_avg_frames").as_int();

  pixel_far_cmd_interval_ms_ = this->get_parameter("pixel_far_cmd_interval_ms").as_int();
  pixel_mid_cmd_interval_ms_ = this->get_parameter("pixel_mid_cmd_interval_ms").as_int();
  pixel_near_cmd_interval_ms_ = this->get_parameter("pixel_near_cmd_interval_ms").as_int();
  pixel_very_near_cmd_interval_ms_ = this->get_parameter("pixel_very_near_cmd_interval_ms").as_int();
  pixel_ultra_near_cmd_interval_ms_ = this->get_parameter("pixel_ultra_near_cmd_interval_ms").as_int();

  pixel_far_scale_ = this->get_parameter("pixel_far_scale").as_double();
  pixel_mid_scale_ = this->get_parameter("pixel_mid_scale").as_double();
  pixel_near_scale_ = this->get_parameter("pixel_near_scale").as_double();
  pixel_very_near_scale_ = this->get_parameter("pixel_very_near_scale").as_double();
  pixel_ultra_near_scale_ = this->get_parameter("pixel_ultra_near_scale").as_double();

  pixel_far_step_max_deg_ = this->get_parameter("pixel_far_step_max_deg").as_double();
  pixel_mid_step_max_deg_ = this->get_parameter("pixel_mid_step_max_deg").as_double();
  pixel_near_step_max_deg_ = this->get_parameter("pixel_near_step_max_deg").as_double();
  pixel_very_near_step_max_deg_ = this->get_parameter("pixel_very_near_step_max_deg").as_double();
  pixel_ultra_near_step_max_deg_ = this->get_parameter("pixel_ultra_near_step_max_deg").as_double();

  pixel_deadzone_deg_ = this->get_parameter("pixel_deadzone_deg").as_double();
  pixel_tolerance_yaw_deg_ = this->get_parameter("pixel_tolerance_yaw_deg").as_double();
  pixel_tolerance_pitch_deg_ = this->get_parameter("pixel_tolerance_pitch_deg").as_double();

  const auto cam_offset = this->get_parameter("camera_offset_xyz").as_double_array();
  const auto laser_offset = this->get_parameter("laser_offset_xyz").as_double_array();
  if (cam_offset.size() != 3 || laser_offset.size() != 3) {
    throw std::runtime_error("camera_offset_xyz and laser_offset_xyz must both have size 3");
  }

  laser_geometry_.setCameraOffset(Eigen::Vector3d(cam_offset[0], cam_offset[1], cam_offset[2]));
  laser_geometry_.setLaserOffset(Eigen::Vector3d(laser_offset[0], laser_offset[1], laser_offset[2]));

  laser_offset_x_ = laser_offset[0];
  laser_offset_y_ = laser_offset[1];
  laser_offset_z_ = laser_offset[2];

  angle_solver_.setLimits(
    this->get_parameter("yaw_inc_min_deg").as_double(),
    this->get_parameter("yaw_inc_max_deg").as_double(),
    this->get_parameter("pitch_inc_min_deg").as_double(),
    this->get_parameter("pitch_inc_max_deg").as_double());
}

bool LaserAimControllerNode::isFinite3(double x, double y, double z) const
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

double LaserAimControllerNode::clampValue(double v, double vmin, double vmax) const
{
  return std::max(vmin, std::min(v, vmax));
}

double LaserAimControllerNode::rad2deg(double rad) const
{
  return rad * kRadToDeg;
}

double LaserAimControllerNode::interpolateByErrorPx(
  double err_px,
  double far_v,
  double mid_v,
  double near_v,
  double very_near_v,
  double ultra_near_v) const
{
  const double e = std::max(0.0, err_px);

  auto lerp = [](double a, double b, double t) {
    return a + (b - a) * t;
  };

  if (e >= pixel_far_error_px_) {
    return far_v;
  }
  if (e >= pixel_near_error_px_) {
    const double denom = std::max(1e-6, pixel_far_error_px_ - pixel_near_error_px_);
    const double t = (e - pixel_near_error_px_) / denom;
    return lerp(mid_v, far_v, clampValue(t, 0.0, 1.0));
  }
  if (e >= pixel_very_near_error_px_) {
    const double denom = std::max(1e-6, pixel_near_error_px_ - pixel_very_near_error_px_);
    const double t = (e - pixel_very_near_error_px_) / denom;
    return lerp(very_near_v, near_v, clampValue(t, 0.0, 1.0));
  }
  if (e >= pixel_ultra_near_error_px_) {
    const double denom = std::max(1e-6, pixel_very_near_error_px_ - pixel_ultra_near_error_px_);
    const double t = (e - pixel_ultra_near_error_px_) / denom;
    return lerp(ultra_near_v, very_near_v, clampValue(t, 0.0, 1.0));
  }
  return ultra_near_v;
}

double LaserAimControllerNode::averageFramesToAlpha(double avg_frames) const
{
  const double n = std::max(1.0, avg_frames);
  return clampValue(2.0 / (n + 1.0), 0.02, 1.0);
}

double LaserAimControllerNode::computeVelocityLimitedAxisCommand(
  double desired_pos_deg,
  double measured_pos_deg,
  double & state_vel_deg_s,
  double dt_s,
  double max_vel_deg_s,
  double max_acc_deg_s2,
  double stop_error_deg) const
{
  if (dt_s <= 1e-6) {
    return measured_pos_deg;
  }

  const double pos_err = desired_pos_deg - measured_pos_deg;
  const double desired_vel =
    (std::abs(pos_err) <= stop_error_deg) ? 0.0
    : clampValue(pos_err / dt_s, -max_vel_deg_s, max_vel_deg_s);

  const double max_dv = std::max(1e-6, max_acc_deg_s2) * dt_s;
  const double vel_err = desired_vel - state_vel_deg_s;
  state_vel_deg_s += clampValue(vel_err, -max_dv, max_dv);
  state_vel_deg_s = clampValue(state_vel_deg_s, -max_vel_deg_s, max_vel_deg_s);

  if (std::abs(pos_err) <= stop_error_deg && std::abs(state_vel_deg_s) < 0.02) {
    state_vel_deg_s = 0.0;
    return desired_pos_deg;
  }

  double step = state_vel_deg_s * dt_s;
  step = clampValue(step, -std::abs(pos_err), std::abs(pos_err));
  return measured_pos_deg + step;
}

bool LaserAimControllerNode::computePnpIncrement(
  const TargetObservation & obs,
  CommandAngles & inc_pnp,
  Eigen::Vector3d * target_from_laser) const
{
  inc_pnp.valid = false;
  inc_pnp.yaw_deg = 0.0;
  inc_pnp.pitch_deg = 0.0;

  if (!obs.has_pose || !isFinite3(obs.tx, obs.ty, obs.tz) || obs.tz <= min_valid_z_m_) {
    return false;
  }

  const Eigen::Vector3d target_cam(obs.tx, obs.ty, obs.tz);
  const Eigen::Vector3d target_laser = laser_geometry_.targetFromLaser(target_cam);
  if (target_from_laser != nullptr) {
    *target_from_laser = target_laser;
  }

  inc_pnp = angle_solver_.solveIncrement(target_laser);
  return inc_pnp.valid;
}

bool LaserAimControllerNode::computePixelCorrection(
  double & out_yaw_deg,
  double & out_pitch_deg,
  double & pixel_err_px,
  double & pixel_err_yaw_deg,
  double & pixel_err_pitch_deg,
  double & zone_scale,
  double & zone_step_max_deg,
  double & filter_alpha,
  std::string & zone_name) const
{
  out_yaw_deg = 0.0;
  out_pitch_deg = 0.0;
  pixel_err_px = 0.0;
  pixel_err_yaw_deg = 0.0;
  pixel_err_pitch_deg = 0.0;
  zone_scale = pixel_far_scale_;
  zone_step_max_deg = pixel_far_step_max_deg_;
  filter_alpha = 1.0;
  zone_name = "far";

  if (!has_filtered_pixel_) {
    return false;
  }

  const double du = filtered_target_u_ - laser_pixel_u_;
  const double dv = filtered_target_v_ - laser_pixel_v_;
  pixel_err_px = std::abs(du) + std::abs(dv);

  zone_scale = interpolateByErrorPx(
    pixel_err_px,
    pixel_far_scale_, pixel_mid_scale_, pixel_near_scale_,
    pixel_very_near_scale_, pixel_ultra_near_scale_);

  zone_step_max_deg = interpolateByErrorPx(
    pixel_err_px,
    pixel_far_step_max_deg_, pixel_mid_step_max_deg_, pixel_near_step_max_deg_,
    pixel_very_near_step_max_deg_, pixel_ultra_near_step_max_deg_);
  zone_step_max_deg = std::min(zone_step_max_deg, fine_single_step_max_abs_deg_);

  const double avg_frames = interpolateByErrorPx(
    pixel_err_px,
    static_cast<double>(pixel_far_avg_frames_),
    static_cast<double>(pixel_mid_avg_frames_),
    static_cast<double>(pixel_near_avg_frames_),
    static_cast<double>(pixel_very_near_avg_frames_),
    static_cast<double>(pixel_ultra_near_avg_frames_));
  filter_alpha = averageFramesToAlpha(avg_frames);

  if (pixel_err_px <= pixel_ultra_near_error_px_) {
    zone_name = "ultra_near";
  } else if (pixel_err_px <= pixel_very_near_error_px_) {
    zone_name = "very_near";
  } else if (pixel_err_px <= pixel_near_error_px_) {
    zone_name = "near";
  } else if (pixel_err_px <= pixel_far_error_px_) {
    zone_name = "mid";
  } else {
    zone_name = "far";
  }

  // yaw 左为正：目标在左时 u 更小，error 为负，纠偏后输出应为正
  const double target_yaw_ray_deg = rad2deg(std::atan2(filtered_target_u_ - camera_cx_, camera_fx_));
  const double laser_yaw_ray_deg = rad2deg(std::atan2(laser_pixel_u_ - camera_cx_, camera_fx_));
  pixel_err_yaw_deg = target_yaw_ray_deg - laser_yaw_ray_deg;

  // pitch 上为正：图像上方 v 更小，经负号后得到正 pitch
  const double target_pitch_ray_deg = rad2deg(std::atan2(filtered_target_v_ - camera_cy_, camera_fy_));
  const double laser_pitch_ray_deg = rad2deg(std::atan2(laser_pixel_v_ - camera_cy_, camera_fy_));
  pixel_err_pitch_deg = target_pitch_ray_deg - laser_pitch_ray_deg;

  double corr_yaw_deg = -pixel_err_yaw_deg * zone_scale;
  double corr_pitch_deg = -pixel_err_pitch_deg * zone_scale;

  if (std::abs(pixel_err_yaw_deg) <= pixel_tolerance_yaw_deg_) {
    corr_yaw_deg = 0.0;
  } else if (std::abs(corr_yaw_deg) < pixel_deadzone_deg_) {
    corr_yaw_deg = std::copysign(pixel_deadzone_deg_, corr_yaw_deg);
  }

  if (std::abs(pixel_err_pitch_deg) <= pixel_tolerance_pitch_deg_) {
    corr_pitch_deg = 0.0;
  } else if (std::abs(corr_pitch_deg) < pixel_deadzone_deg_) {
    corr_pitch_deg = std::copysign(pixel_deadzone_deg_, corr_pitch_deg);
  }

  out_yaw_deg = clampValue(corr_yaw_deg, -zone_step_max_deg, zone_step_max_deg);
  out_pitch_deg = clampValue(corr_pitch_deg, -zone_step_max_deg, zone_step_max_deg);
  return true;
}

void LaserAimControllerNode::applyManualOffset(double & yaw_deg, double & pitch_deg) const
{
  if (!has_manual_cmd_) {
    return;
  }
  yaw_deg += manual_yaw_deg_;
  pitch_deg += manual_pitch_deg_;
}

void LaserAimControllerNode::publishCommand(double yaw_deg, double pitch_deg)
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(2);
  msg.data[0] = yaw_deg;
  msg.data[1] = pitch_deg;
  cmd_pub_->publish(msg);
}

void LaserAimControllerNode::resetPixelFilter()
{
  has_filtered_pixel_ = false;
  filtered_target_u_ = 0.0;
  filtered_target_v_ = 0.0;
  last_filtered_target_seq_ = 0;
  pnp_bias_yaw_deg_ = 0.0;
  pnp_bias_pitch_deg_ = 0.0;
}

void LaserAimControllerNode::resetControlStateTo(double yaw_deg, double pitch_deg)
{
  command_vel_yaw_deg_s_ = 0.0;
  command_vel_pitch_deg_s_ = 0.0;
  controller_state_initialized_ = true;
  last_auto_cmd_yaw_deg_ = yaw_deg;
  last_auto_cmd_pitch_deg_ = pitch_deg;
  has_last_output_ = false;
}

const char * LaserAimControllerNode::modeToString(ControlMode mode) const
{
  switch (mode) {
    case ControlMode::HOLD:
      return "HOLD";
    case ControlMode::FINE_TRACK:
      return "FINE_TRACK";
    case ControlMode::PNP_ONLY_TRACK:
      return "PNP_ONLY_TRACK";
    case ControlMode::LOST_RECOVER:
      return "LOST_RECOVER";
    default:
      return "UNKNOWN";
  }
}

void LaserAimControllerNode::targetCallback(const auto_aim_fast::msg::AimTarget::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_target_.received = true;
  latest_target_.has_target = msg->has_target;
  latest_target_.has_pose = msg->has_pose;
  latest_target_.u = static_cast<double>(msg->u);
  latest_target_.v = static_cast<double>(msg->v);
  latest_target_.tx = msg->tx;
  latest_target_.ty = msg->ty;
  latest_target_.tz = msg->tz;
  latest_target_.rx_time = this->get_clock()->now();
  latest_target_.seq = ++target_seq_counter_;
}

void LaserAimControllerNode::gimbalAnglesCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  current_gimbal_.yaw_deg = msg->data[0];
  current_gimbal_.pitch_deg = msg->data[1];
  current_gimbal_.valid =
    std::isfinite(current_gimbal_.yaw_deg) &&
    std::isfinite(current_gimbal_.pitch_deg);
}

void LaserAimControllerNode::quaternionCallback(
  const geometry_msgs::msg::QuaternionStamped::SharedPtr)
{
  // 保留接口
}

void LaserAimControllerNode::roughCommandCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  rough_yaw_deg_ = msg->data[0];
  rough_pitch_deg_ = msg->data[1];
  has_rough_cmd_ = std::isfinite(rough_yaw_deg_) && std::isfinite(rough_pitch_deg_);
}

void LaserAimControllerNode::manualCommandCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  manual_yaw_deg_ = msg->data[0];
  manual_pitch_deg_ = msg->data[1];
  has_manual_cmd_ = std::isfinite(manual_yaw_deg_) && std::isfinite(manual_pitch_deg_);
  if (!has_manual_cmd_) {
    manual_yaw_deg_ = 0.0;
    manual_pitch_deg_ = 0.0;
  }
}

void LaserAimControllerNode::controlLoop()
{
  const rclcpp::Time now = this->get_clock()->now();

  TargetObservation obs;
  GimbalState gimbal;
  bool has_rough = false;
  double rough_yaw = 0.0;
  double rough_pitch = 0.0;
  bool has_manual = false;
  double manual_yaw = 0.0;
  double manual_pitch = 0.0;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    obs = latest_target_;
    gimbal = current_gimbal_;
    has_rough = has_rough_cmd_;
    rough_yaw = rough_yaw_deg_;
    rough_pitch = rough_pitch_deg_;
    has_manual = has_manual_cmd_;
    manual_yaw = manual_yaw_deg_;
    manual_pitch = manual_pitch_deg_;
  }

  double dt_s = 1.0 / std::max(1.0, control_rate_hz_);
  if (last_control_time_.nanoseconds() > 0) {
    dt_s = (now - last_control_time_).seconds();
    dt_s = clampValue(dt_s, 1e-4, 0.1);
  }
  last_control_time_ = now;

  if (!gimbal.valid) {
    return;
  }

  const bool target_fresh =
    obs.received && ((now - obs.rx_time).seconds() <= target_timeout_s_);

  const bool explicit_lost = obs.received && !obs.has_target;
  const bool target_stale = obs.received && !target_fresh;

  const bool pixel_valid =
    target_fresh && obs.has_target &&
    std::isfinite(obs.u) && std::isfinite(obs.v) &&
    !(std::abs(obs.u) < 1e-6 && std::abs(obs.v) < 1e-6);

  const bool pnp_valid =
    target_fresh && obs.has_target && obs.has_pose &&
    isFinite3(obs.tx, obs.ty, obs.tz) && obs.tz > min_valid_z_m_;

  if (pixel_valid && obs.seq != last_filtered_target_seq_) {
    const double err_px_raw =
      std::abs(obs.u - laser_pixel_u_) + std::abs(obs.v - laser_pixel_v_);
    const double avg_frames = interpolateByErrorPx(
      err_px_raw,
      static_cast<double>(pixel_far_avg_frames_),
      static_cast<double>(pixel_mid_avg_frames_),
      static_cast<double>(pixel_near_avg_frames_),
      static_cast<double>(pixel_very_near_avg_frames_),
      static_cast<double>(pixel_ultra_near_avg_frames_));
    const double alpha = averageFramesToAlpha(avg_frames);

    if (!has_filtered_pixel_) {
      filtered_target_u_ = obs.u;
      filtered_target_v_ = obs.v;
      has_filtered_pixel_ = true;
    } else {
      filtered_target_u_ = alpha * obs.u + (1.0 - alpha) * filtered_target_u_;
      filtered_target_v_ = alpha * obs.v + (1.0 - alpha) * filtered_target_v_;
    }
    last_filtered_target_seq_ = obs.seq;
  } else if (!pixel_valid) {
    resetPixelFilter();
  }

  mode_ = ControlMode::HOLD;
  if (increment_mode_ == "pnp") {
    if (pnp_valid) {
      mode_ = ControlMode::PNP_ONLY_TRACK;
    } else if ((explicit_lost || target_stale) && has_rough && fallback_to_rough_when_no_target_) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  } else if (increment_mode_ == "pixel") {
    if (pixel_valid && has_filtered_pixel_) {
      mode_ = ControlMode::FINE_TRACK;
    } else if ((explicit_lost || target_stale) && has_rough && fallback_to_rough_when_no_target_) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  } else {
    if (pixel_valid && has_filtered_pixel_) {
      mode_ = ControlMode::FINE_TRACK;
    } else if (pnp_valid) {
      mode_ = ControlMode::PNP_ONLY_TRACK;
    } else if ((explicit_lost || target_stale) && has_rough && fallback_to_rough_when_no_target_) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  }

  if (!controller_state_initialized_) {
    resetControlStateTo(gimbal.yaw_deg, gimbal.pitch_deg);
  }

  if ((mode_ == ControlMode::FINE_TRACK || mode_ == ControlMode::PNP_ONLY_TRACK) &&
      (std::abs(last_auto_cmd_yaw_deg_ - gimbal.yaw_deg) > feedback_snap_deg_ ||
       std::abs(last_auto_cmd_pitch_deg_ - gimbal.pitch_deg) > feedback_snap_deg_))
  {
    resetControlStateTo(gimbal.yaw_deg, gimbal.pitch_deg);
  }

  if (mode_ != last_mode_) {
    command_vel_yaw_deg_s_ = 0.0;
    command_vel_pitch_deg_s_ = 0.0;
    has_last_output_ = false;
    if (mode_ != ControlMode::FINE_TRACK) {
      pnp_bias_yaw_deg_ = 0.0;
      pnp_bias_pitch_deg_ = 0.0;
    }
    last_mode_ = mode_;
  }

  if (mode_ == ControlMode::HOLD && !has_hold_auto_cmd_ && gimbal.valid) {
    hold_auto_yaw_deg_ = gimbal.yaw_deg;
    hold_auto_pitch_deg_ = gimbal.pitch_deg;
    has_hold_auto_cmd_ = true;
  }

  double desired_yaw = gimbal.yaw_deg;
  double desired_pitch = gimbal.pitch_deg;

  double pixel_corr_yaw = 0.0;
  double pixel_corr_pitch = 0.0;
  double pixel_err_px = 0.0;
  double pixel_err_yaw_deg = 0.0;
  double pixel_err_pitch_deg = 0.0;
  double zone_scale = 0.0;
  double zone_step_max_deg = fine_single_step_max_abs_deg_;
  double filter_alpha = 1.0;
  std::string zone_name = "none";

  // pixel 调试中间量
  double pixel_du = 0.0;
  double pixel_dv = 0.0;
  double target_yaw_ray_deg = 0.0;
  double laser_yaw_ray_deg = 0.0;
  double target_pitch_ray_deg = 0.0;
  double laser_pitch_ray_deg = 0.0;
  double pixel_corr_yaw_before_deadzone = 0.0;
  double pixel_corr_pitch_before_deadzone = 0.0;
  double pixel_corr_yaw_after_deadzone = 0.0;
  double pixel_corr_pitch_after_deadzone = 0.0;
  double pixel_corr_yaw_after_clamp = 0.0;
  double pixel_corr_pitch_after_clamp = 0.0;

  CommandAngles inc_pnp;
  inc_pnp.valid = false;
  inc_pnp.yaw_deg = 0.0;
  inc_pnp.pitch_deg = 0.0;

  // pnp 调试中间量
  Eigen::Vector3d target_cam_dbg(0.0, 0.0, 0.0);
  bool pnp_has_pose_dbg = false;
  Eigen::Vector3d target_from_laser(0.0, 0.0, 0.0);

  if (pnp_valid) {
    pnp_has_pose_dbg = true;
    target_cam_dbg = Eigen::Vector3d(obs.tx, obs.ty, obs.tz);
  }

  const bool inc_pnp_ok = computePnpIncrement(obs, inc_pnp, &target_from_laser);

  if (mode_ == ControlMode::FINE_TRACK) {
    const bool pixel_ok = computePixelCorrection(
      pixel_corr_yaw, pixel_corr_pitch,
      pixel_err_px,
      pixel_err_yaw_deg, pixel_err_pitch_deg,
      zone_scale, zone_step_max_deg,
      filter_alpha, zone_name);

    if (pixel_ok && has_filtered_pixel_) {
      pixel_du = filtered_target_u_ - laser_pixel_u_;
      pixel_dv = filtered_target_v_ - laser_pixel_v_;

      target_yaw_ray_deg =
        rad2deg(std::atan2(filtered_target_u_ - camera_cx_, camera_fx_));
      laser_yaw_ray_deg =
        rad2deg(std::atan2(laser_pixel_u_ - camera_cx_, camera_fx_));

      target_pitch_ray_deg =
        rad2deg(std::atan2(filtered_target_v_ - camera_cy_, camera_fy_));
      laser_pitch_ray_deg =
        rad2deg(std::atan2(laser_pixel_v_ - camera_cy_, camera_fy_));

      // 严格按 computePixelCorrection() 当前逻辑复算，不改符号
      pixel_corr_yaw_before_deadzone = -pixel_err_yaw_deg * zone_scale;
      pixel_corr_pitch_before_deadzone = -pixel_err_pitch_deg * zone_scale;

      pixel_corr_yaw_after_deadzone = pixel_corr_yaw_before_deadzone;
      pixel_corr_pitch_after_deadzone = pixel_corr_pitch_before_deadzone;

      if (std::abs(pixel_err_yaw_deg) <= pixel_tolerance_yaw_deg_) {
        pixel_corr_yaw_after_deadzone = 0.0;
      } else if (std::abs(pixel_corr_yaw_after_deadzone) < pixel_deadzone_deg_) {
        pixel_corr_yaw_after_deadzone =
          std::copysign(pixel_deadzone_deg_, pixel_corr_yaw_after_deadzone);
      }

      if (std::abs(pixel_err_pitch_deg) <= pixel_tolerance_pitch_deg_) {
        pixel_corr_pitch_after_deadzone = 0.0;
      } else if (std::abs(pixel_corr_pitch_after_deadzone) < pixel_deadzone_deg_) {
        pixel_corr_pitch_after_deadzone =
          std::copysign(pixel_deadzone_deg_, pixel_corr_pitch_after_deadzone);
      }

      pixel_corr_yaw_after_clamp =
        clampValue(pixel_corr_yaw_after_deadzone, -zone_step_max_deg, zone_step_max_deg);
      pixel_corr_pitch_after_clamp =
        clampValue(pixel_corr_pitch_after_deadzone, -zone_step_max_deg, zone_step_max_deg);
    }

    double total_corr_yaw = pixel_ok ? pixel_corr_yaw : 0.0;
    double total_corr_pitch = pixel_ok ? pixel_corr_pitch : 0.0;

    if (increment_mode_ == "auto" && enable_pnp_assist_ && inc_pnp_ok) {
      const bool pnp_healthy =
        std::abs(inc_pnp.yaw_deg) <= std::min(pnp_exceed_threshold_deg_, pnp_assist_valid_inc_deg_) &&
        std::abs(inc_pnp.pitch_deg) <= std::min(pnp_exceed_threshold_deg_, pnp_assist_valid_inc_deg_);

      const double target_bias_yaw = pnp_healthy ?
        clampValue(inc_pnp.yaw_deg * pnp_assist_gain_, -pnp_assist_max_deg_, pnp_assist_max_deg_) : 0.0;
      const double target_bias_pitch = pnp_healthy ?
        clampValue(inc_pnp.pitch_deg * pnp_assist_gain_, -pnp_assist_max_deg_, pnp_assist_max_deg_) : 0.0;

      pnp_bias_yaw_deg_ = pnp_assist_lpf_alpha_ * target_bias_yaw +
        (1.0 - pnp_assist_lpf_alpha_) * pnp_bias_yaw_deg_;
      pnp_bias_pitch_deg_ = pnp_assist_lpf_alpha_ * target_bias_pitch +
        (1.0 - pnp_assist_lpf_alpha_) * pnp_bias_pitch_deg_;

      if (pixel_ok) {
        if (std::abs(total_corr_yaw) < 2.0 * std::abs(pnp_bias_yaw_deg_) ||
            std::signbit(total_corr_yaw) == std::signbit(pnp_bias_yaw_deg_)) {
          total_corr_yaw += pnp_bias_yaw_deg_;
        }
        if (std::abs(total_corr_pitch) < 2.0 * std::abs(pnp_bias_pitch_deg_) ||
            std::signbit(total_corr_pitch) == std::signbit(pnp_bias_pitch_deg_)) {
          total_corr_pitch += pnp_bias_pitch_deg_;
        }
      }
    } else {
      pnp_bias_yaw_deg_ *= (1.0 - pnp_assist_lpf_alpha_);
      pnp_bias_pitch_deg_ *= (1.0 - pnp_assist_lpf_alpha_);
    }

    total_corr_yaw = clampValue(
      total_corr_yaw, -fine_single_step_max_abs_deg_, fine_single_step_max_abs_deg_);
    total_corr_pitch = clampValue(
      total_corr_pitch, -fine_single_step_max_abs_deg_, fine_single_step_max_abs_deg_);

    desired_yaw = gimbal.yaw_deg + total_corr_yaw;
    desired_pitch = gimbal.pitch_deg + total_corr_pitch;
  } else if (mode_ == ControlMode::PNP_ONLY_TRACK) {
    if (inc_pnp_ok) {
      const double corr_yaw = clampValue(
        inc_pnp.yaw_deg, -fine_single_step_max_abs_deg_, fine_single_step_max_abs_deg_);
      const double corr_pitch = clampValue(
        inc_pnp.pitch_deg, -fine_single_step_max_abs_deg_, fine_single_step_max_abs_deg_);
      desired_yaw = gimbal.yaw_deg + corr_yaw;
      desired_pitch = gimbal.pitch_deg + corr_pitch;
    }
    pnp_bias_yaw_deg_ = 0.0;
    pnp_bias_pitch_deg_ = 0.0;
  } else if (mode_ == ControlMode::LOST_RECOVER) {
    if (search_stage_mode_ == "scan") {
      if (has_rough) {
        const double t = now.seconds();
        const double wx = 2.0 * kPi / std::max(1e-6, scan_period_s_);
        const double wy = 3.0 * wx;  // 频率比 1:3

        const double scan_yaw =
          scan_window_half_yaw_deg_ * std::sin(wx * t + scan_phase_x_rad_);
        const double scan_pitch =
          scan_window_half_pitch_deg_ * std::sin(wy * t + scan_phase_y_rad_);

        desired_yaw = rough_yaw + scan_yaw;
        desired_pitch = rough_pitch + scan_pitch;
      } else if (has_hold_auto_cmd_) {
        // scan模式无粗瞄：全hold
        desired_yaw = hold_auto_yaw_deg_;
        desired_pitch = hold_auto_pitch_deg_;
      } else {
        desired_yaw = gimbal.yaw_deg;
        desired_pitch = gimbal.pitch_deg;
      }
    } else {
      // manual 搜索链路
      if (has_rough) {
        desired_yaw = rough_yaw;
        desired_pitch = rough_pitch;
      } else if (has_hold_auto_cmd_) {
        desired_yaw = hold_auto_yaw_deg_;
        desired_pitch = hold_auto_pitch_deg_;
      } else {
        desired_yaw = gimbal.yaw_deg;
        desired_pitch = gimbal.pitch_deg;
      }
    }

    pnp_bias_yaw_deg_ = 0.0;
    pnp_bias_pitch_deg_ = 0.0;
  } else {
    // HOLD
    if (has_hold_auto_cmd_) {
      desired_yaw = hold_auto_yaw_deg_;
      desired_pitch = hold_auto_pitch_deg_;
    } else {
      desired_yaw = gimbal.yaw_deg;
      desired_pitch = gimbal.pitch_deg;
    }

    pnp_bias_yaw_deg_ = 0.0;
    pnp_bias_pitch_deg_ = 0.0;
  }

  const double max_vel =
    (mode_ == ControlMode::LOST_RECOVER) ? recover_max_vel_deg_s_ : fine_track_max_vel_deg_s_;
  const double max_acc =
    (mode_ == ControlMode::LOST_RECOVER) ? recover_max_acc_deg_s2_ : fine_track_max_acc_deg_s2_;

  double auto_yaw = computeVelocityLimitedAxisCommand(
    desired_yaw, gimbal.yaw_deg, command_vel_yaw_deg_s_, dt_s, max_vel, max_acc, stop_error_deg_);
  double auto_pitch = computeVelocityLimitedAxisCommand(
    desired_pitch, gimbal.pitch_deg, command_vel_pitch_deg_s_, dt_s, max_vel, max_acc, stop_error_deg_);

  if (enable_output_filter_) {
    if (!has_last_output_) {
      last_yaw_out_deg_ = auto_yaw;
      last_pitch_out_deg_ = auto_pitch;
      has_last_output_ = true;
    } else {
      const double a = clampValue(output_filter_alpha_, 0.0, 1.0);
      last_yaw_out_deg_ = a * auto_yaw + (1.0 - a) * last_yaw_out_deg_;
      last_pitch_out_deg_ = a * auto_pitch + (1.0 - a) * last_pitch_out_deg_;
    }
    auto_yaw = last_yaw_out_deg_;
    auto_pitch = last_pitch_out_deg_;
  }

  last_auto_cmd_yaw_deg_ = auto_yaw;
  last_auto_cmd_pitch_deg_ = auto_pitch;

  if (mode_ == ControlMode::FINE_TRACK || mode_ == ControlMode::PNP_ONLY_TRACK) {
    hold_auto_yaw_deg_ = auto_yaw;
    hold_auto_pitch_deg_ = auto_pitch;
    has_hold_auto_cmd_ = true;
  }

  double final_yaw = auto_yaw;
  double final_pitch = auto_pitch;

  // manual 只允许在“无目标搜索 + manual链路”里生效
  const bool allow_manual_in_final =
    (mode_ == ControlMode::HOLD || mode_ == ControlMode::LOST_RECOVER) &&
    (search_stage_mode_ == "manual");

  if (allow_manual_in_final && has_manual) {
    final_yaw += manual_yaw;
    final_pitch += manual_pitch;
  }

  publishCommand(final_yaw, final_pitch);

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[PIXEL_RAW] valid=%s raw_uv=(%.2f, %.2f) filt_uv=(%.2f, %.2f) laser_uv=(%.2f, %.2f) "
      "du=%.2f dv=%.2f err_px=%.2f avg_alpha=%.3f zone=%s",
      (pixel_valid && has_filtered_pixel_) ? "true" : "false",
      obs.u, obs.v,
      has_filtered_pixel_ ? filtered_target_u_ : 0.0,
      has_filtered_pixel_ ? filtered_target_v_ : 0.0,
      laser_pixel_u_, laser_pixel_v_,
      pixel_du, pixel_dv,
      pixel_err_px, filter_alpha,
      zone_name.c_str());

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[PIXEL_SOLVE] "
      "target_ray=(yaw=%.4f pitch=%.4f) laser_ray=(yaw=%.4f pitch=%.4f) "
      "err_deg=(yaw=%.4f pitch=%.4f) "
      "corr_pre=(yaw=%.4f pitch=%.4f) "
      "corr_deadzone=(yaw=%.4f pitch=%.4f) "
      "corr_clamp=(yaw=%.4f pitch=%.4f) "
      "tol=(yaw=%.4f pitch=%.4f) deadzone=%.4f scale=%.4f step_max=%.4f",
      target_yaw_ray_deg, target_pitch_ray_deg,
      laser_yaw_ray_deg, laser_pitch_ray_deg,
      pixel_err_yaw_deg, pixel_err_pitch_deg,
      pixel_corr_yaw_before_deadzone, pixel_corr_pitch_before_deadzone,
      pixel_corr_yaw_after_deadzone, pixel_corr_pitch_after_deadzone,
      pixel_corr_yaw_after_clamp, pixel_corr_pitch_after_clamp,
      pixel_tolerance_yaw_deg_, pixel_tolerance_pitch_deg_,
      pixel_deadzone_deg_, zone_scale, zone_step_max_deg);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[PNP_SOLVE] valid=%s has_pose=%s "
      "obs_t=(%.4f, %.4f, %.4f) "
      "target_cam=(%.4f, %.4f, %.4f) "
      "target_laser=(%.4f, %.4f, %.4f) "
      "laser_offset=(%.4f, %.4f, %.4f) "
      "inc_pnp=(yaw=%.4f pitch=%.4f) "
      "assist_bias=(yaw=%.4f pitch=%.4f)",
      inc_pnp_ok ? "true" : "false",
      pnp_has_pose_dbg ? "true" : "false",
      obs.tx, obs.ty, obs.tz,
      target_cam_dbg.x(), target_cam_dbg.y(), target_cam_dbg.z(),
      target_from_laser.x(), target_from_laser.y(), target_from_laser.z(),
      laser_offset_x_, laser_offset_y_, laser_offset_z_,
      inc_pnp.yaw_deg, inc_pnp.pitch_deg,
      pnp_bias_yaw_deg_, pnp_bias_pitch_deg_);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 200,
      "mode=%s search_mode=%s target_fresh=%s has_target=%s pixel_valid=%s pnp_valid=%s has_rough=%s "
      "gimbal=(%.3f, %.3f) hold_auto=(%.3f, %.3f) rough=(%.3f, %.3f) "
      "pixel_err=(px=%.1f yaw=%.3f pitch=%.3f) zone=%s scale=%.3f step_max=%.3f "
      "pnp=(valid=%s yaw=%.3f pitch=%.3f bias=(%.3f, %.3f)) "
      "desired=(%.3f, %.3f) auto=(%.3f, %.3f) manual=(%.3f, %.3f) final=(%.3f, %.3f)",
      modeToString(mode_),
      search_stage_mode_.c_str(),
      target_fresh ? "true" : "false",
      obs.has_target ? "true" : "false",
      pixel_valid ? "true" : "false",
      pnp_valid ? "true" : "false",
      has_rough ? "true" : "false",
      gimbal.yaw_deg,
      gimbal.pitch_deg,
      has_hold_auto_cmd_ ? hold_auto_yaw_deg_ : 0.0,
      has_hold_auto_cmd_ ? hold_auto_pitch_deg_ : 0.0,
      has_rough ? rough_yaw : 0.0,
      has_rough ? rough_pitch : 0.0,
      pixel_err_px,
      pixel_err_yaw_deg,
      pixel_err_pitch_deg,
      zone_name.c_str(),
      zone_scale,
      zone_step_max_deg,
      inc_pnp_ok ? "true" : "false",
      inc_pnp.yaw_deg,
      inc_pnp.pitch_deg,
      pnp_bias_yaw_deg_,
      pnp_bias_pitch_deg_,
      desired_yaw,
      desired_pitch,
      auto_yaw,
      auto_pitch,
      (allow_manual_in_final && has_manual) ? manual_yaw : 0.0,
      (allow_manual_in_final && has_manual) ? manual_pitch : 0.0,
      final_yaw,
      final_pitch);
  }
}

}  // namespace laser_aim_controller