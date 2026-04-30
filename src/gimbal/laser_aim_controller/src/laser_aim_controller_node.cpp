#include "laser_aim_controller/laser_aim_controller_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace laser_aim_controller
{

namespace
{
constexpr double kRadToDeg = 57.2957795130823208768;
constexpr double kPi = 3.14159265358979323846;

double clampFree(double v, double vmin, double vmax)
{
  return std::max(vmin, std::min(v, vmax));
}

double signum(double v)
{
  if (v > 0.0) {
    return 1.0;
  }
  if (v < 0.0) {
    return -1.0;
  }
  return 0.0;
}

struct TrackPlannerParams
{
  double low_speed_delay_s = 0.040;
  double high_speed_delay_s = 0.075;
  double speed_threshold_deg_s = 6.0;
  double prediction_max_lead_deg = 3.0;

  double measurement_alpha = 0.55;
  double velocity_alpha = 0.30;

  double pnp_consistency_deg = 0.35;
  double pnp_reject_deg = 1.20;
  double pnp_blend_weight = 0.10;

  double track_feedback_reset_deg = 4.0;

  bool manual_always_add = true;
};

struct OutputFilterParams
{
  double track_alpha = 1.0;
  double pnp_only_alpha = 1.0;
  double recover_alpha = 0.25;
  double hold_alpha = 0.20;
  double scan_alpha = 1.0;
};

struct AxisTrackState
{
  AxisTrackState()
  : initialized(false),
    has_measurement(false),
    meas_pos_deg(0.0),
    meas_vel_deg_s(0.0),
    meas_stamp(0, 0, RCL_ROS_TIME),
    ref_pos_deg(0.0),
    ref_vel_deg_s(0.0)
  {}

  bool initialized;
  bool has_measurement;

  double meas_pos_deg;
  double meas_vel_deg_s;
  rclcpp::Time meas_stamp;

  double ref_pos_deg;
  double ref_vel_deg_s;
};

struct TrackPlannerState
{
  TrackPlannerState()
  : initialized(false),
    yaw(),
    pitch(),
    last_prediction_delay_s(0.0),
    last_pnp_consistent(false),
    last_measurement_source("none"),
    last_track_meas_yaw_deg(0.0),
    last_track_meas_pitch_deg(0.0),
    last_pred_target_yaw_deg(0.0),
    last_pred_target_pitch_deg(0.0)
  {}

  bool initialized;
  AxisTrackState yaw;
  AxisTrackState pitch;

  double last_prediction_delay_s;
  bool last_pnp_consistent;
  std::string last_measurement_source;
  double last_track_meas_yaw_deg;
  double last_track_meas_pitch_deg;
  double last_pred_target_yaw_deg;
  double last_pred_target_pitch_deg;
};

struct RecoverScanParams
{
  double rough_timeout_s = 0.25;
  double rough_measurement_alpha = 0.30;
  double rough_velocity_alpha = 0.18;
  double rough_prediction_delay_s = 0.05;
  double rough_prediction_max_lead_deg = 2.0;

  double center_max_vel_deg_s = 25.0;
  double center_max_acc_deg_s2 = 300.0;

  double yaw_envelope_full_on_err_deg = 0.20;
  double yaw_envelope_full_off_err_deg = 0.80;
  double pitch_envelope_full_on_err_deg = 0.35;
  double pitch_envelope_full_off_err_deg = 1.20;

  double envelope_rise_s = 0.60;
  double envelope_fall_s = 0.20;

  double yaw_freq_scale = 0.45;
  double pitch_freq_scale = 0.90;

  double yaw_amp_scale = 0.20;
  double yaw_env_max = 0.35;

  double pitch_amp_up_scale = 1.00;
  double pitch_amp_down_scale = 0.35;
  double pitch_up_bias_deg = 0.30;
};

struct RecoverScanState
{
  RecoverScanState()
  : initialized(false),
    center_yaw(),
    center_pitch(),
    scan_phase_yaw_rad(0.0),
    scan_phase_pitch_rad(0.0),
    scan_env_yaw(0.0),
    scan_env_pitch(0.0),
    last_center_target_yaw_deg(0.0),
    last_center_target_pitch_deg(0.0),
    last_center_ref_yaw_deg(0.0),
    last_center_ref_pitch_deg(0.0),
    last_scan_yaw_deg(0.0),
    last_scan_pitch_deg(0.0)
  {}

  bool initialized;

  AxisTrackState center_yaw;
  AxisTrackState center_pitch;

  double scan_phase_yaw_rad;
  double scan_phase_pitch_rad;

  double scan_env_yaw;
  double scan_env_pitch;

  double last_center_target_yaw_deg;
  double last_center_target_pitch_deg;
  double last_center_ref_yaw_deg;
  double last_center_ref_pitch_deg;
  double last_scan_yaw_deg;
  double last_scan_pitch_deg;
};

std::unordered_map<const LaserAimControllerNode *, TrackPlannerParams> g_track_planner_params;
std::unordered_map<const LaserAimControllerNode *, OutputFilterParams> g_output_filter_params;
std::unordered_map<const LaserAimControllerNode *, TrackPlannerState> g_track_planner_state;

std::unordered_map<const LaserAimControllerNode *, RecoverScanParams> g_recover_scan_params;
std::unordered_map<const LaserAimControllerNode *, RecoverScanState> g_recover_scan_state;
std::unordered_map<const LaserAimControllerNode *, int64_t> g_rough_cmd_stamp_ns;

void resetAxisTrackState(
  AxisTrackState & state,
  double pos_deg,
  const rclcpp::Time & stamp)
{
  state.initialized = true;
  state.has_measurement = false;
  state.meas_pos_deg = pos_deg;
  state.meas_vel_deg_s = 0.0;
  state.meas_stamp = stamp;
  state.ref_pos_deg = pos_deg;
  state.ref_vel_deg_s = 0.0;
}

void resetTrackPlannerState(
  TrackPlannerState & state,
  double yaw_deg,
  double pitch_deg,
  const rclcpp::Time & stamp)
{
  resetAxisTrackState(state.yaw, yaw_deg, stamp);
  resetAxisTrackState(state.pitch, pitch_deg, stamp);
  state.initialized = true;
  state.last_prediction_delay_s = 0.0;
  state.last_pnp_consistent = false;
  state.last_measurement_source = "none";
  state.last_track_meas_yaw_deg = yaw_deg;
  state.last_track_meas_pitch_deg = pitch_deg;
  state.last_pred_target_yaw_deg = yaw_deg;
  state.last_pred_target_pitch_deg = pitch_deg;
}

void resetRecoverScanState(
  RecoverScanState & state,
  double yaw_deg,
  double pitch_deg,
  const rclcpp::Time & stamp)
{
  resetAxisTrackState(state.center_yaw, yaw_deg, stamp);
  resetAxisTrackState(state.center_pitch, pitch_deg, stamp);
  state.initialized = true;
  state.scan_phase_yaw_rad = 0.0;
  state.scan_phase_pitch_rad = 0.0;
  state.scan_env_yaw = 0.0;
  state.scan_env_pitch = 0.0;
  state.last_center_target_yaw_deg = yaw_deg;
  state.last_center_target_pitch_deg = pitch_deg;
  state.last_center_ref_yaw_deg = yaw_deg;
  state.last_center_ref_pitch_deg = pitch_deg;
  state.last_scan_yaw_deg = 0.0;
  state.last_scan_pitch_deg = 0.0;
}

void updateAxisMeasurement(
  AxisTrackState & state,
  double raw_pos_deg,
  const rclcpp::Time & stamp,
  double pos_alpha,
  double vel_alpha)
{
  pos_alpha = clampFree(pos_alpha, 0.02, 1.0);
  vel_alpha = clampFree(vel_alpha, 0.02, 1.0);

  if (!state.initialized) {
    resetAxisTrackState(state, raw_pos_deg, stamp);
  }

  if (!state.has_measurement) {
    state.meas_pos_deg = raw_pos_deg;
    state.meas_vel_deg_s = 0.0;
    state.meas_stamp = stamp;
    state.has_measurement = true;
    return;
  }

  double dt_s = (stamp - state.meas_stamp).seconds();
  dt_s = clampFree(dt_s, 1e-4, 0.2);

  const double prev_pos = state.meas_pos_deg;
  const double filt_pos = pos_alpha * raw_pos_deg + (1.0 - pos_alpha) * prev_pos;
  const double raw_vel = (filt_pos - prev_pos) / dt_s;

  state.meas_pos_deg = filt_pos;
  state.meas_vel_deg_s =
    vel_alpha * raw_vel + (1.0 - vel_alpha) * state.meas_vel_deg_s;
  state.meas_stamp = stamp;
  state.has_measurement = true;
}

double choosePredictionDelay(
  double angular_speed_deg_s,
  const TrackPlannerParams & params)
{
  const double s = std::max(0.0, angular_speed_deg_s);
  const double thresh = std::max(1e-3, params.speed_threshold_deg_s);

  if (s <= thresh) {
    return params.low_speed_delay_s;
  }
  if (s >= 2.0 * thresh) {
    return params.high_speed_delay_s;
  }

  const double t = (s - thresh) / thresh;
  return params.low_speed_delay_s +
    (params.high_speed_delay_s - params.low_speed_delay_s) * clampFree(t, 0.0, 1.0);
}

double predictAxisTarget(
  const AxisTrackState & state,
  const rclcpp::Time & now,
  double extra_delay_s,
  double max_lead_deg)
{
  if (!state.initialized) {
    return 0.0;
  }
  if (!state.has_measurement) {
    return state.ref_pos_deg;
  }

  const double age_s = clampFree((now - state.meas_stamp).seconds(), 0.0, 0.3);
  const double lead_s = std::max(0.0, age_s + extra_delay_s);
  const double raw_pred = state.meas_pos_deg + state.meas_vel_deg_s * lead_s;
  const double lead_delta = clampFree(
    raw_pred - state.meas_pos_deg,
    -std::abs(max_lead_deg),
    std::abs(max_lead_deg));
  return state.meas_pos_deg + lead_delta;
}

void stepAxisReference(
  AxisTrackState & state,
  double target_pos_deg,
  double dt_s,
  double max_vel_deg_s,
  double max_acc_deg_s2)
{
  if (!state.initialized) {
    resetAxisTrackState(state, target_pos_deg, rclcpp::Time(0, 0, RCL_ROS_TIME));
  }

  dt_s = clampFree(dt_s, 1e-4, 0.1);
  max_vel_deg_s = std::max(1e-3, max_vel_deg_s);
  max_acc_deg_s2 = std::max(1e-3, max_acc_deg_s2);

  const double dist = target_pos_deg - state.ref_pos_deg;
  const double abs_dist = std::abs(dist);

  if (abs_dist <= 1e-5 && std::abs(state.ref_vel_deg_s) <= 1e-3) {
    state.ref_pos_deg = target_pos_deg;
    state.ref_vel_deg_s = 0.0;
    return;
  }

  double desired_vel_mag = std::sqrt(std::max(0.0, 2.0 * max_acc_deg_s2 * abs_dist));
  desired_vel_mag = std::min(desired_vel_mag, max_vel_deg_s);
  const double desired_vel = signum(dist) * desired_vel_mag;

  const double max_dv = max_acc_deg_s2 * dt_s;
  const double vel_err = desired_vel - state.ref_vel_deg_s;
  state.ref_vel_deg_s += clampFree(vel_err, -max_dv, max_dv);
  state.ref_vel_deg_s = clampFree(state.ref_vel_deg_s, -max_vel_deg_s, max_vel_deg_s);

  double step = state.ref_vel_deg_s * dt_s;
  if (std::abs(step) > abs_dist && std::signbit(step) == std::signbit(dist)) {
    state.ref_pos_deg = target_pos_deg;
    if (desired_vel_mag < 0.05) {
      state.ref_vel_deg_s = 0.0;
    }
    return;
  }

  state.ref_pos_deg += step;

  if (std::abs(target_pos_deg - state.ref_pos_deg) < 0.01 &&
      std::abs(state.ref_vel_deg_s) < 0.05)
  {
    state.ref_pos_deg = target_pos_deg;
    state.ref_vel_deg_s = 0.0;
  }
}

double moveToward(double current, double target, double max_step)
{
  if (target > current) {
    return std::min(target, current + max_step);
  }
  return std::max(target, current - max_step);
}

double wrapAngleRad(double x)
{
  while (x > kPi) {
    x -= 2.0 * kPi;
  }
  while (x < -kPi) {
    x += 2.0 * kPi;
  }
  return x;
}

}  // namespace

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
  RCLCPP_INFO(this->get_logger(), "lost_auto_hold_timeout_s: %.3f", lost_auto_hold_timeout_s_);
  RCLCPP_INFO(
    this->get_logger(),
    "laser pixel v depth compensation: %s ref=(%.3f m, %.2f px) slope=%.3f px/m clamp=[%.1f, %.1f]",
    laser_pixel_v_depth_comp_enable_ ? "true" : "false",
    laser_pixel_v_ref_distance_m_,
    laser_pixel_v_ref_px_,
    laser_pixel_v_slope_px_per_m_,
    laser_pixel_v_min_px_,
    laser_pixel_v_max_px_);
  RCLCPP_INFO(this->get_logger(), "scan_window_half_yaw_deg: %.3f", scan_window_half_yaw_deg_);
  RCLCPP_INFO(this->get_logger(), "scan_window_half_pitch_deg: %.3f", scan_window_half_pitch_deg_);
  RCLCPP_INFO(this->get_logger(), "scan_period_s: %.3f", scan_period_s_);

  const auto & planner_params = g_track_planner_params[this];
  const auto & filter_params = g_output_filter_params[this];
  const auto & recover_scan_params = g_recover_scan_params[this];

  RCLCPP_INFO(this->get_logger(),
    "delay-aware planner: low_delay=%.3f high_delay=%.3f speed_thresh=%.3f max_lead=%.3f",
    planner_params.low_speed_delay_s,
    planner_params.high_speed_delay_s,
    planner_params.speed_threshold_deg_s,
    planner_params.prediction_max_lead_deg);
  RCLCPP_INFO(this->get_logger(),
    "pnp gating: consistency=%.3f reject=%.3f blend_weight=%.3f",
    planner_params.pnp_consistency_deg,
    planner_params.pnp_reject_deg,
    planner_params.pnp_blend_weight);
  RCLCPP_INFO(this->get_logger(),
    "output_filter_alpha by mode: track=%.3f pnp_only=%.3f recover=%.3f hold=%.3f scan=%.3f",
    filter_params.track_alpha,
    filter_params.pnp_only_alpha,
    filter_params.recover_alpha,
    filter_params.hold_alpha,
    filter_params.scan_alpha);
  RCLCPP_INFO(this->get_logger(),
    "recover_scan: rough_timeout=%.3f rough_alpha=%.3f rough_vel_alpha=%.3f rough_delay=%.3f",
    recover_scan_params.rough_timeout_s,
    recover_scan_params.rough_measurement_alpha,
    recover_scan_params.rough_velocity_alpha,
    recover_scan_params.rough_prediction_delay_s);
  RCLCPP_INFO(this->get_logger(),
    "recover_scan: yaw_env_max=%.3f yaw_amp_scale=%.3f pitch_up_scale=%.3f pitch_down_scale=%.3f pitch_up_bias=%.3f",
    recover_scan_params.yaw_env_max,
    recover_scan_params.yaw_amp_scale,
    recover_scan_params.pitch_amp_up_scale,
    recover_scan_params.pitch_amp_down_scale,
    recover_scan_params.pitch_up_bias_deg);
  RCLCPP_INFO(this->get_logger(),
    "recover_scan: freq_scale(yaw=%.3f pitch=%.3f)",
    recover_scan_params.yaw_freq_scale,
    recover_scan_params.pitch_freq_scale);
}

void LaserAimControllerNode::declareAndLoadParameters()
{
  this->declare_parameter<std::string>("target_topic", "/auto_aim/target");
  this->declare_parameter<std::string>("gimbal_angles_topic", "/gimbal/angles_true");
  this->declare_parameter<std::string>("gimbal_quaternion_topic", "/gimbal/quaternion_true");
  this->declare_parameter<std::string>("rough_command_topic", "/gimbal/rough_command_angles");
  this->declare_parameter<std::string>("manual_command_topic", "/gimbal/command_angle_manual");
  this->declare_parameter<std::string>("command_topic", "/gimbal/command_angles");

  this->declare_parameter<std::string>("gimbal_angle_order", "yaw_pitch");
  this->declare_parameter<std::string>("increment_mode", "auto");

  this->declare_parameter<bool>("publish_debug", true);
  this->declare_parameter<bool>("enable_output_filter", false);
  this->declare_parameter<bool>("fallback_to_rough_when_no_target", true);
  this->declare_parameter<bool>("enable_pnp_assist", true);

  this->declare_parameter<std::string>("search_stage_mode", "scan");
  this->declare_parameter<double>("scan_window_half_yaw_deg", 1.5);
  this->declare_parameter<double>("scan_window_half_pitch_deg", 1.5);
  this->declare_parameter<double>("scan_period_s", 4.0);
  this->declare_parameter<double>("scan_phase_x_rad", 0.0);
  this->declare_parameter<double>("scan_phase_y_rad", 1.57079632679);

  this->declare_parameter<double>("output_filter_alpha", 0.15);
  this->declare_parameter<double>("track_output_filter_alpha", 1.0);
  this->declare_parameter<double>("pnp_only_output_filter_alpha", 1.0);
  this->declare_parameter<double>("recover_output_filter_alpha", 0.25);
  this->declare_parameter<double>("hold_output_filter_alpha", 0.20);
  this->declare_parameter<double>("scan_output_filter_alpha", 0.30);

  this->declare_parameter<double>("min_valid_z_m", 0.05);
  this->declare_parameter<double>("control_rate_hz", 100.0);
  this->declare_parameter<double>("target_timeout_s", 0.12);
  this->declare_parameter<double>("feedback_snap_deg", 0.5);
  this->declare_parameter<double>("lost_auto_hold_timeout_s", 2.0);

  this->declare_parameter<double>("fine_track_max_vel_deg_s", 20.0);
  this->declare_parameter<double>("fine_track_max_acc_deg_s2", 520.0);
  this->declare_parameter<double>("recover_max_vel_deg_s", 160.0);
  this->declare_parameter<double>("recover_max_acc_deg_s2", 3000.0);
  this->declare_parameter<double>("stop_error_deg", 0.08);

  this->declare_parameter<double>("pnp_exceed_threshold_deg", 5.0);
  this->declare_parameter<double>("pnp_assist_gain", 0.06);
  this->declare_parameter<double>("pnp_assist_max_deg", 0.02);
  this->declare_parameter<double>("pnp_assist_lpf_alpha", 0.10);
  this->declare_parameter<double>("pnp_assist_valid_inc_deg", 1.0);

  this->declare_parameter<double>("track_low_speed_delay_s", 0.040);
  this->declare_parameter<double>("track_high_speed_delay_s", 0.075);
  this->declare_parameter<double>("track_speed_threshold_deg_s", 6.0);
  this->declare_parameter<double>("track_prediction_max_lead_deg", 3.0);
  this->declare_parameter<double>("track_measurement_alpha", 0.55);
  this->declare_parameter<double>("track_velocity_alpha", 0.30);
  this->declare_parameter<double>("track_pnp_consistency_deg", 0.35);
  this->declare_parameter<double>("track_pnp_reject_deg", 1.20);
  this->declare_parameter<double>("track_pnp_blend_weight", 0.10);
  this->declare_parameter<double>("track_feedback_reset_deg", 4.0);
  this->declare_parameter<bool>("manual_always_add", true);

  this->declare_parameter<double>("recover_rough_timeout_s", 0.25);
  this->declare_parameter<double>("recover_rough_measurement_alpha", 0.30);
  this->declare_parameter<double>("recover_rough_velocity_alpha", 0.18);
  this->declare_parameter<double>("recover_rough_prediction_delay_s", 0.05);
  this->declare_parameter<double>("recover_rough_prediction_max_lead_deg", 2.0);

  this->declare_parameter<double>("recover_scan_center_max_vel_deg_s", 25.0);
  this->declare_parameter<double>("recover_scan_center_max_acc_deg_s2", 300.0);

  this->declare_parameter<double>("recover_scan_yaw_envelope_full_on_err_deg", 0.20);
  this->declare_parameter<double>("recover_scan_yaw_envelope_full_off_err_deg", 0.80);
  this->declare_parameter<double>("recover_scan_pitch_envelope_full_on_err_deg", 0.35);
  this->declare_parameter<double>("recover_scan_pitch_envelope_full_off_err_deg", 1.20);

  this->declare_parameter<double>("recover_scan_envelope_rise_s", 0.60);
  this->declare_parameter<double>("recover_scan_envelope_fall_s", 0.20);

  this->declare_parameter<double>("recover_scan_yaw_freq_scale", 0.45);
  this->declare_parameter<double>("recover_scan_pitch_freq_scale", 0.90);

  this->declare_parameter<double>("recover_scan_yaw_amp_scale", 0.20);
  this->declare_parameter<double>("recover_scan_yaw_env_max", 0.35);

  this->declare_parameter<double>("recover_scan_pitch_amp_up_scale", 1.00);
  this->declare_parameter<double>("recover_scan_pitch_amp_down_scale", 0.35);
  this->declare_parameter<double>("recover_scan_pitch_up_bias_deg", 0.30);

  this->declare_parameter<std::vector<double>>(
    "camera_offset_xyz", std::vector<double>{0.0, -0.021, 0.0});
  this->declare_parameter<std::vector<double>>(
    "laser_offset_xyz", std::vector<double>{0.0, -0.056, 0.0});

  this->declare_parameter<double>("yaw_inc_min_deg", -5.0);
  this->declare_parameter<double>("yaw_inc_max_deg", 5.0);
  this->declare_parameter<double>("pitch_inc_min_deg", -5.0);
  this->declare_parameter<double>("pitch_inc_max_deg", 5.0);
  this->declare_parameter<double>("fine_single_step_max_abs_deg", 2.0);

  this->declare_parameter<double>("camera_fx", 33000.0);
  this->declare_parameter<double>("camera_fy", 33000.0);
  this->declare_parameter<double>("camera_cx", 426.297277);
  this->declare_parameter<double>("camera_cy", 567.094694);
  this->declare_parameter<std::vector<double>>(
    "laser_pixel_uv", std::vector<double>{923.0, 368.0});
  this->declare_parameter<bool>("laser_pixel_v_depth_comp_enable", true);
  this->declare_parameter<double>("laser_pixel_v_ref_distance_m", 13.5);
  this->declare_parameter<double>("laser_pixel_v_ref_px", 368.0);
  this->declare_parameter<double>("laser_pixel_v_slope_px_per_m", 5.0);
  this->declare_parameter<double>("laser_pixel_v_min_px", 330.0);
  this->declare_parameter<double>("laser_pixel_v_max_px", 430.0);

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
  this->declare_parameter<double>("pixel_near_step_max_deg", 0.90);
  this->declare_parameter<double>("pixel_very_near_step_max_deg", 0.80);
  this->declare_parameter<double>("pixel_ultra_near_step_max_deg", 0.50);

  this->declare_parameter<double>("pixel_deadzone_deg", 0.03);
  this->declare_parameter<double>("pixel_tolerance_yaw_deg", 0.02);
  this->declare_parameter<double>("pixel_tolerance_pitch_deg", 0.02);

  this->declare_parameter<bool>("pixel_direction_excite_enable", false);
  this->declare_parameter<int>("pixel_direction_excite_trigger_frames", 3);
  this->declare_parameter<int>("pixel_direction_excite_boost_hold_frames", 3);
  this->declare_parameter<double>("pixel_direction_excite_cmd_min_effective_deg", 0.03);
  this->declare_parameter<double>("pixel_direction_excite_feedback_min_effective_deg", 0.005);
  this->declare_parameter<double>("pixel_direction_excite_corr_min_deg", 0.05);
  this->declare_parameter<double>("pixel_direction_excite_ratio", 0.45);
  this->declare_parameter<double>("pixel_direction_excite_min_deg", 0.08);
  this->declare_parameter<double>("pixel_direction_excite_max_deg", 0.30);
  this->declare_parameter<int>("pixel_direction_reverse_clear_frames", 2);
  this->declare_parameter<double>("pixel_direction_reverse_clear_vel_deg_s", 0.02);

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
  lost_auto_hold_timeout_s_ =
    std::max(0.0, this->get_parameter("lost_auto_hold_timeout_s").as_double());

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
  laser_pixel_v_depth_comp_enable_ =
    this->get_parameter("laser_pixel_v_depth_comp_enable").as_bool();
  laser_pixel_v_ref_distance_m_ =
    this->get_parameter("laser_pixel_v_ref_distance_m").as_double();
  laser_pixel_v_ref_px_ = this->get_parameter("laser_pixel_v_ref_px").as_double();
  laser_pixel_v_slope_px_per_m_ =
    this->get_parameter("laser_pixel_v_slope_px_per_m").as_double();
  laser_pixel_v_min_px_ = this->get_parameter("laser_pixel_v_min_px").as_double();
  laser_pixel_v_max_px_ = this->get_parameter("laser_pixel_v_max_px").as_double();
  if (laser_pixel_v_min_px_ > laser_pixel_v_max_px_) {
    std::swap(laser_pixel_v_min_px_, laser_pixel_v_max_px_);
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

  TrackPlannerParams planner_params;
  planner_params.low_speed_delay_s =
    std::max(0.0, this->get_parameter("track_low_speed_delay_s").as_double());
  planner_params.high_speed_delay_s =
    std::max(planner_params.low_speed_delay_s,
      this->get_parameter("track_high_speed_delay_s").as_double());
  planner_params.speed_threshold_deg_s =
    std::max(1e-3, this->get_parameter("track_speed_threshold_deg_s").as_double());
  planner_params.prediction_max_lead_deg =
    std::max(0.0, this->get_parameter("track_prediction_max_lead_deg").as_double());
  planner_params.measurement_alpha =
    clampValue(this->get_parameter("track_measurement_alpha").as_double(), 0.02, 1.0);
  planner_params.velocity_alpha =
    clampValue(this->get_parameter("track_velocity_alpha").as_double(), 0.02, 1.0);
  planner_params.pnp_consistency_deg =
    std::max(0.0, this->get_parameter("track_pnp_consistency_deg").as_double());
  planner_params.pnp_reject_deg =
    std::max(planner_params.pnp_consistency_deg,
      this->get_parameter("track_pnp_reject_deg").as_double());
  planner_params.pnp_blend_weight =
    clampValue(this->get_parameter("track_pnp_blend_weight").as_double(), 0.0, 0.5);
  planner_params.track_feedback_reset_deg =
    std::max(0.5, this->get_parameter("track_feedback_reset_deg").as_double());
  planner_params.manual_always_add =
    this->get_parameter("manual_always_add").as_bool();
  g_track_planner_params[this] = planner_params;

  OutputFilterParams filter_params;
  filter_params.track_alpha =
    clampValue(this->get_parameter("track_output_filter_alpha").as_double(), 0.0, 1.0);
  filter_params.pnp_only_alpha =
    clampValue(this->get_parameter("pnp_only_output_filter_alpha").as_double(), 0.0, 1.0);
  filter_params.recover_alpha =
    clampValue(this->get_parameter("recover_output_filter_alpha").as_double(), 0.0, 1.0);
  filter_params.hold_alpha =
    clampValue(this->get_parameter("hold_output_filter_alpha").as_double(), 0.0, 1.0);
  filter_params.scan_alpha =
    clampValue(this->get_parameter("scan_output_filter_alpha").as_double(), 0.0, 1.0);
  g_output_filter_params[this] = filter_params;

  RecoverScanParams recover_scan_params;
  recover_scan_params.rough_timeout_s =
    std::max(0.01, this->get_parameter("recover_rough_timeout_s").as_double());
  recover_scan_params.rough_measurement_alpha =
    clampValue(this->get_parameter("recover_rough_measurement_alpha").as_double(), 0.02, 1.0);
  recover_scan_params.rough_velocity_alpha =
    clampValue(this->get_parameter("recover_rough_velocity_alpha").as_double(), 0.02, 1.0);
  recover_scan_params.rough_prediction_delay_s =
    std::max(0.0, this->get_parameter("recover_rough_prediction_delay_s").as_double());
  recover_scan_params.rough_prediction_max_lead_deg =
    std::max(0.0, this->get_parameter("recover_rough_prediction_max_lead_deg").as_double());

  recover_scan_params.center_max_vel_deg_s =
    std::max(1e-3, this->get_parameter("recover_scan_center_max_vel_deg_s").as_double());
  recover_scan_params.center_max_acc_deg_s2 =
    std::max(1e-3, this->get_parameter("recover_scan_center_max_acc_deg_s2").as_double());

  recover_scan_params.yaw_envelope_full_on_err_deg =
    std::max(0.0, this->get_parameter("recover_scan_yaw_envelope_full_on_err_deg").as_double());
  recover_scan_params.yaw_envelope_full_off_err_deg =
    std::max(recover_scan_params.yaw_envelope_full_on_err_deg,
      this->get_parameter("recover_scan_yaw_envelope_full_off_err_deg").as_double());
  recover_scan_params.pitch_envelope_full_on_err_deg =
    std::max(0.0, this->get_parameter("recover_scan_pitch_envelope_full_on_err_deg").as_double());
  recover_scan_params.pitch_envelope_full_off_err_deg =
    std::max(recover_scan_params.pitch_envelope_full_on_err_deg,
      this->get_parameter("recover_scan_pitch_envelope_full_off_err_deg").as_double());

  recover_scan_params.envelope_rise_s =
    std::max(0.05, this->get_parameter("recover_scan_envelope_rise_s").as_double());
  recover_scan_params.envelope_fall_s =
    std::max(0.05, this->get_parameter("recover_scan_envelope_fall_s").as_double());

  recover_scan_params.yaw_freq_scale =
    std::max(0.01, this->get_parameter("recover_scan_yaw_freq_scale").as_double());
  recover_scan_params.pitch_freq_scale =
    std::max(0.01, this->get_parameter("recover_scan_pitch_freq_scale").as_double());

  recover_scan_params.yaw_amp_scale =
    std::max(0.0, this->get_parameter("recover_scan_yaw_amp_scale").as_double());
  recover_scan_params.yaw_env_max =
    clampValue(this->get_parameter("recover_scan_yaw_env_max").as_double(), 0.0, 1.0);

  recover_scan_params.pitch_amp_up_scale =
    std::max(0.0, this->get_parameter("recover_scan_pitch_amp_up_scale").as_double());
  recover_scan_params.pitch_amp_down_scale =
    std::max(0.0, this->get_parameter("recover_scan_pitch_amp_down_scale").as_double());
  recover_scan_params.pitch_up_bias_deg =
    this->get_parameter("recover_scan_pitch_up_bias_deg").as_double();

  g_recover_scan_params[this] = recover_scan_params;

  g_track_planner_state[this] = TrackPlannerState();
  g_recover_scan_state[this] = RecoverScanState();
  g_rough_cmd_stamp_ns[this] = 0;

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
  const double max_dv = std::max(1e-6, max_acc_deg_s2) * dt_s;

  if (std::abs(pos_err) <= stop_error_deg) {
    state_vel_deg_s += clampValue(-state_vel_deg_s, -2.0 * max_dv, 2.0 * max_dv);
    if (std::abs(state_vel_deg_s) < 0.05) {
      state_vel_deg_s = 0.0;
      return desired_pos_deg;
    }
  } else if (std::abs(state_vel_deg_s) > 0.02 &&
             std::signbit(state_vel_deg_s) != std::signbit(pos_err)) {
    state_vel_deg_s += clampValue(-state_vel_deg_s, -2.0 * max_dv, 2.0 * max_dv);
    if (std::abs(state_vel_deg_s) < 0.05) {
      state_vel_deg_s = 0.0;
    }
  }

  const double desired_vel =
    (std::abs(pos_err) <= stop_error_deg) ? 0.0
    : clampValue(pos_err / dt_s, -max_vel_deg_s, max_vel_deg_s);

  const double vel_err = desired_vel - state_vel_deg_s;
  state_vel_deg_s += clampValue(vel_err, -max_dv, max_dv);
  state_vel_deg_s = clampValue(state_vel_deg_s, -max_vel_deg_s, max_vel_deg_s);

  if (std::abs(pos_err) <= stop_error_deg && std::abs(state_vel_deg_s) < 0.05) {
    state_vel_deg_s = 0.0;
    return desired_pos_deg;
  }

  double step = state_vel_deg_s * dt_s;
  if (std::abs(pos_err) > stop_error_deg && step * pos_err < 0.0) {
    state_vel_deg_s = 0.0;
    return measured_pos_deg;
  }
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

  const double du = filtered_target_u_ - laser_pixel_u;
  const double dv = filtered_target_v_ - laser_pixel_v;
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

  const double target_yaw_ray_deg = rad2deg(std::atan2(filtered_target_u_ - camera_cx_, camera_fx_));
  const double laser_yaw_ray_deg = rad2deg(std::atan2(laser_pixel_u - camera_cx_, camera_fx_));
  pixel_err_yaw_deg = target_yaw_ray_deg - laser_yaw_ray_deg;

  const double target_pitch_ray_deg = rad2deg(std::atan2(filtered_target_v_ - camera_cy_, camera_fy_));
  const double laser_pitch_ray_deg = rad2deg(std::atan2(laser_pixel_v - camera_cy_, camera_fy_));
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

double LaserAimControllerNode::depthCompensatedLaserPixelV(
  const TargetObservation & obs,
  bool pnp_valid) const
{
  if (!laser_pixel_v_depth_comp_enable_ || !pnp_valid) {
    return laser_pixel_v_;
  }

  double distance_m = obs.distance;
  if (!std::isfinite(distance_m) || distance_m <= min_valid_z_m_) {
    distance_m = obs.tz;
  }
  if (!std::isfinite(distance_m) || distance_m <= min_valid_z_m_) {
    return laser_pixel_v_;
  }

  const double raw_v =
    laser_pixel_v_ref_px_ +
    laser_pixel_v_slope_px_per_m_ * (distance_m - laser_pixel_v_ref_distance_m_);
  return clampValue(raw_v, laser_pixel_v_min_px_, laser_pixel_v_max_px_);
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
  has_auto_command_reference_ = true;
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
  latest_target_.distance = msg->distance;
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
  if (has_rough_cmd_) {
    g_rough_cmd_stamp_ns[this] = this->get_clock()->now().nanoseconds();
  }
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
  int64_t rough_stamp_ns = 0;
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
    auto rough_it = g_rough_cmd_stamp_ns.find(this);
    if (rough_it != g_rough_cmd_stamp_ns.end()) {
      rough_stamp_ns = rough_it->second;
    }
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

  auto & planner_params = g_track_planner_params[this];
  auto & filter_params = g_output_filter_params[this];
  auto & planner_state = g_track_planner_state[this];
  auto & recover_scan_params = g_recover_scan_params[this];
  auto & recover_scan_state = g_recover_scan_state[this];

  const bool target_fresh =
    obs.received && ((now - obs.rx_time).seconds() <= target_timeout_s_);

  const bool visual_missing = !target_fresh || !obs.has_target;

  const bool pixel_valid =
    target_fresh && obs.has_target &&
    std::isfinite(obs.u) && std::isfinite(obs.v) &&
    !(std::abs(obs.u) < 1e-6 && std::abs(obs.v) < 1e-6);

  const bool pnp_valid =
    target_fresh && obs.has_target && obs.has_pose &&
    isFinite3(obs.tx, obs.ty, obs.tz) && obs.tz > min_valid_z_m_;

  const double active_laser_pixel_u = laser_pixel_u_;
  const double active_laser_pixel_v = depthCompensatedLaserPixelV(obs, pnp_valid);

  if (pixel_valid && obs.seq != last_filtered_target_seq_) {
    const double err_px_raw =
      std::abs(obs.u - active_laser_pixel_u) + std::abs(obs.v - active_laser_pixel_v);
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

  const ControlMode prev_mode = last_mode_;
  mode_ = ControlMode::HOLD;
  const bool rough_fallback_ready =
    visual_missing && has_rough && fallback_to_rough_when_no_target_;
  if (increment_mode_ == "pnp") {
    if (pnp_valid) {
      mode_ = ControlMode::PNP_ONLY_TRACK;
    } else if (rough_fallback_ready) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  } else if (increment_mode_ == "pixel") {
    if (pixel_valid && has_filtered_pixel_) {
      mode_ = ControlMode::FINE_TRACK;
    } else if (rough_fallback_ready) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  } else {
    if (pixel_valid && has_filtered_pixel_) {
      mode_ = ControlMode::FINE_TRACK;
    } else if (pnp_valid) {
      mode_ = ControlMode::PNP_ONLY_TRACK;
    } else if (rough_fallback_ready) {
      mode_ = ControlMode::LOST_RECOVER;
    }
  }

  const bool tracking_mode =
    (mode_ == ControlMode::FINE_TRACK || mode_ == ControlMode::PNP_ONLY_TRACK);
  if (tracking_mode && !gimbal.valid) {
    return;
  }

  auto commandReferenceYaw = [this]() -> double {
    return (has_auto_command_reference_ && std::isfinite(last_auto_cmd_yaw_deg_))
      ? last_auto_cmd_yaw_deg_
      : 0.0;
  };
  auto commandReferencePitch = [this]() -> double {
    return (has_auto_command_reference_ && std::isfinite(last_auto_cmd_pitch_deg_))
      ? last_auto_cmd_pitch_deg_
      : 0.0;
  };

  if (!controller_state_initialized_) {
    resetControlStateTo(
      tracking_mode ? gimbal.yaw_deg : commandReferenceYaw(),
      tracking_mode ? gimbal.pitch_deg : commandReferencePitch());
  }

  double open_loop_yaw = commandReferenceYaw();
  double open_loop_pitch = commandReferencePitch();

  if (!planner_state.initialized) {
    resetTrackPlannerState(
      planner_state,
      tracking_mode ? gimbal.yaw_deg : open_loop_yaw,
      tracking_mode ? gimbal.pitch_deg : open_loop_pitch,
      now);
  }

  const bool mode_changed = (mode_ != prev_mode);

  if (mode_changed) {
    command_vel_yaw_deg_s_ = 0.0;
    command_vel_pitch_deg_s_ = 0.0;
    has_last_output_ = false;
    pnp_bias_yaw_deg_ = 0.0;
    pnp_bias_pitch_deg_ = 0.0;
  }

  if (tracking_mode || !visual_missing) {
    lost_auto_hold_active_ = false;
    lost_auto_hold_expired_logged_ = false;
    lost_auto_hold_start_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  } else if (has_tracked_target_auto_cmd_ &&
             (mode_ == ControlMode::LOST_RECOVER || mode_ == ControlMode::HOLD))
  {
    if (!lost_auto_hold_active_) {
      lost_auto_hold_active_ = true;
      lost_auto_hold_expired_logged_ = false;
      lost_auto_hold_start_ = now;
    }
  }

  double lost_auto_hold_age_s = 0.0;
  if (lost_auto_hold_active_ && lost_auto_hold_start_.nanoseconds() > 0) {
    lost_auto_hold_age_s = std::max(0.0, (now - lost_auto_hold_start_).seconds());
  }
  const bool lost_auto_hold_valid =
    lost_auto_hold_active_ &&
    lost_auto_hold_timeout_s_ > 0.0 &&
    lost_auto_hold_age_s <= lost_auto_hold_timeout_s_;
  const bool hold_last_auto_after_loss =
    lost_auto_hold_valid && has_hold_auto_cmd_;

  if (lost_auto_hold_active_ && !lost_auto_hold_valid && !lost_auto_hold_expired_logged_) {
    RCLCPP_INFO(
      this->get_logger(),
      "lost target auto hold expired after %.3f s, switching to %s recovery",
      lost_auto_hold_age_s,
      search_stage_mode_.c_str());
    lost_auto_hold_expired_logged_ = true;
  }

  if (hold_last_auto_after_loss &&
      mode_ == ControlMode::LOST_RECOVER &&
      search_stage_mode_ == "scan")
  {
    recover_scan_state.initialized = false;
  } else if (mode_changed && mode_ == ControlMode::LOST_RECOVER && search_stage_mode_ == "scan") {
    resetRecoverScanState(recover_scan_state, open_loop_yaw, open_loop_pitch, now);
  }

  const bool entering_track =
    tracking_mode &&
    !(prev_mode == ControlMode::FINE_TRACK || prev_mode == ControlMode::PNP_ONLY_TRACK);

  if (entering_track) {
    const double init_yaw =
      std::isfinite(last_auto_cmd_yaw_deg_) ? last_auto_cmd_yaw_deg_ : gimbal.yaw_deg;
    const double init_pitch =
      std::isfinite(last_auto_cmd_pitch_deg_) ? last_auto_cmd_pitch_deg_ : gimbal.pitch_deg;
    resetTrackPlannerState(planner_state, init_yaw, init_pitch, now);
  }

  if (mode_ == ControlMode::HOLD && !has_hold_auto_cmd_) {
    hold_auto_yaw_deg_ = open_loop_yaw;
    hold_auto_pitch_deg_ = open_loop_pitch;
    has_hold_auto_cmd_ = true;
  }

  if (tracking_mode &&
      (std::abs(planner_state.yaw.ref_pos_deg - gimbal.yaw_deg) > planner_params.track_feedback_reset_deg ||
       std::abs(planner_state.pitch.ref_pos_deg - gimbal.pitch_deg) > planner_params.track_feedback_reset_deg))
  {
    resetTrackPlannerState(planner_state, gimbal.yaw_deg, gimbal.pitch_deg, now);
    has_last_output_ = false;
  }

  double desired_yaw = open_loop_yaw;
  double desired_pitch = open_loop_pitch;

  double pixel_corr_yaw = 0.0;
  double pixel_corr_pitch = 0.0;
  double pixel_err_px = 0.0;
  double pixel_err_yaw_deg = 0.0;
  double pixel_err_pitch_deg = 0.0;
  double zone_scale = 0.0;
  double zone_step_max_deg = fine_single_step_max_abs_deg_;
  double filter_alpha = 1.0;
  std::string zone_name = "none";

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

  double pixel_full_inc_yaw_deg = 0.0;
  double pixel_full_inc_pitch_deg = 0.0;
  bool pixel_geom_valid = false;

  CommandAngles inc_pnp;
  inc_pnp.valid = false;
  inc_pnp.yaw_deg = 0.0;
  inc_pnp.pitch_deg = 0.0;

  Eigen::Vector3d target_cam_dbg(0.0, 0.0, 0.0);
  bool pnp_has_pose_dbg = false;
  Eigen::Vector3d target_from_laser(0.0, 0.0, 0.0);

  if (pnp_valid) {
    pnp_has_pose_dbg = true;
    target_cam_dbg = Eigen::Vector3d(obs.tx, obs.ty, obs.tz);
  }

  const bool inc_pnp_ok = computePnpIncrement(obs, inc_pnp, &target_from_laser);

  const bool pixel_ok = computePixelCorrection(
    active_laser_pixel_u, active_laser_pixel_v,
    pixel_corr_yaw, pixel_corr_pitch,
    pixel_err_px,
    pixel_err_yaw_deg, pixel_err_pitch_deg,
    zone_scale, zone_step_max_deg,
    filter_alpha, zone_name);

  if (pixel_ok && has_filtered_pixel_) {
    pixel_du = filtered_target_u_ - active_laser_pixel_u;
    pixel_dv = filtered_target_v_ - active_laser_pixel_v;

    target_yaw_ray_deg =
      rad2deg(std::atan2(filtered_target_u_ - camera_cx_, camera_fx_));
    laser_yaw_ray_deg =
      rad2deg(std::atan2(active_laser_pixel_u - camera_cx_, camera_fx_));

    target_pitch_ray_deg =
      rad2deg(std::atan2(filtered_target_v_ - camera_cy_, camera_fy_));
    laser_pitch_ray_deg =
      rad2deg(std::atan2(active_laser_pixel_v - camera_cy_, camera_fy_));

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

    pixel_full_inc_yaw_deg = -pixel_err_yaw_deg;
    pixel_full_inc_pitch_deg = -pixel_err_pitch_deg;
    pixel_geom_valid = true;
  }

  bool has_track_meas = false;
  double track_meas_yaw_deg = 0.0;
  double track_meas_pitch_deg = 0.0;
  std::string measurement_source = "none";
  bool pnp_consistent = false;

  double pixel_abs_yaw_deg = 0.0;
  double pixel_abs_pitch_deg = 0.0;
  if (tracking_mode && pixel_geom_valid) {
    pixel_abs_yaw_deg = gimbal.yaw_deg + pixel_full_inc_yaw_deg;
    pixel_abs_pitch_deg = gimbal.pitch_deg + pixel_full_inc_pitch_deg;
    track_meas_yaw_deg = pixel_abs_yaw_deg;
    track_meas_pitch_deg = pixel_abs_pitch_deg;
    has_track_meas = true;
    measurement_source = "pixel";
  }

  double pnp_abs_yaw_deg = 0.0;
  double pnp_abs_pitch_deg = 0.0;
  if (tracking_mode && inc_pnp_ok) {
    pnp_abs_yaw_deg = gimbal.yaw_deg + inc_pnp.yaw_deg;
    pnp_abs_pitch_deg = gimbal.pitch_deg + inc_pnp.pitch_deg;

    if (has_track_meas) {
      const double dy = pnp_abs_yaw_deg - pixel_abs_yaw_deg;
      const double dp = pnp_abs_pitch_deg - pixel_abs_pitch_deg;
      pnp_consistent =
        std::abs(dy) <= planner_params.pnp_consistency_deg &&
        std::abs(dp) <= planner_params.pnp_consistency_deg;

      if (enable_pnp_assist_ && pnp_consistent) {
        const double w = planner_params.pnp_blend_weight;
        const double fused_yaw =
          (1.0 - w) * track_meas_yaw_deg + w * pnp_abs_yaw_deg;
        const double fused_pitch =
          (1.0 - w) * track_meas_pitch_deg + w * pnp_abs_pitch_deg;

        pnp_bias_yaw_deg_ = fused_yaw - pixel_abs_yaw_deg;
        pnp_bias_pitch_deg_ = fused_pitch - pixel_abs_pitch_deg;

        track_meas_yaw_deg = fused_yaw;
        track_meas_pitch_deg = fused_pitch;
        measurement_source = "pixel+pnp";
      } else {
        pnp_bias_yaw_deg_ = 0.0;
        pnp_bias_pitch_deg_ = 0.0;
      }
    } else {
      has_track_meas = true;
      track_meas_yaw_deg = pnp_abs_yaw_deg;
      track_meas_pitch_deg = pnp_abs_pitch_deg;
      measurement_source = "pnp";
      pnp_bias_yaw_deg_ = 0.0;
      pnp_bias_pitch_deg_ = 0.0;
    }
  } else {
    pnp_bias_yaw_deg_ = 0.0;
    pnp_bias_pitch_deg_ = 0.0;
  }

  if (tracking_mode && pixel_geom_valid && inc_pnp_ok) {
    const double dy = pnp_abs_yaw_deg - pixel_abs_yaw_deg;
    const double dp = pnp_abs_pitch_deg - pixel_abs_pitch_deg;
    if (std::abs(dy) > planner_params.pnp_reject_deg ||
        std::abs(dp) > planner_params.pnp_reject_deg) {
      track_meas_yaw_deg = pixel_abs_yaw_deg;
      track_meas_pitch_deg = pixel_abs_pitch_deg;
      has_track_meas = true;
      measurement_source = "pixel";
      pnp_bias_yaw_deg_ = 0.0;
      pnp_bias_pitch_deg_ = 0.0;
      pnp_consistent = false;
    }
  }

  planner_state.last_measurement_source = measurement_source;
  planner_state.last_pnp_consistent = pnp_consistent;
  planner_state.last_track_meas_yaw_deg =
    has_track_meas ? track_meas_yaw_deg : planner_state.last_track_meas_yaw_deg;
  planner_state.last_track_meas_pitch_deg =
    has_track_meas ? track_meas_pitch_deg : planner_state.last_track_meas_pitch_deg;

  if (tracking_mode) {
    if (has_track_meas) {
      const rclcpp::Time meas_stamp = obs.received ? obs.rx_time : now;

      updateAxisMeasurement(
        planner_state.yaw,
        track_meas_yaw_deg,
        meas_stamp,
        planner_params.measurement_alpha,
        planner_params.velocity_alpha);

      updateAxisMeasurement(
        planner_state.pitch,
        track_meas_pitch_deg,
        meas_stamp,
        planner_params.measurement_alpha,
        planner_params.velocity_alpha);

      const double ang_speed_deg_s = std::hypot(
        planner_state.yaw.meas_vel_deg_s,
        planner_state.pitch.meas_vel_deg_s);

      const double prediction_delay_s =
        choosePredictionDelay(ang_speed_deg_s, planner_params);
      planner_state.last_prediction_delay_s = prediction_delay_s;

      const double pred_target_yaw_deg = predictAxisTarget(
        planner_state.yaw, now, prediction_delay_s, planner_params.prediction_max_lead_deg);
      const double pred_target_pitch_deg = predictAxisTarget(
        planner_state.pitch, now, prediction_delay_s, planner_params.prediction_max_lead_deg);

      planner_state.last_pred_target_yaw_deg = pred_target_yaw_deg;
      planner_state.last_pred_target_pitch_deg = pred_target_pitch_deg;

      stepAxisReference(
        planner_state.yaw,
        pred_target_yaw_deg,
        dt_s,
        fine_track_max_vel_deg_s_,
        fine_track_max_acc_deg_s2_);

      stepAxisReference(
        planner_state.pitch,
        pred_target_pitch_deg,
        dt_s,
        fine_track_max_vel_deg_s_,
        fine_track_max_acc_deg_s2_);

      desired_yaw = planner_state.yaw.ref_pos_deg;
      desired_pitch = planner_state.pitch.ref_pos_deg;

      command_vel_yaw_deg_s_ = planner_state.yaw.ref_vel_deg_s;
      command_vel_pitch_deg_s_ = planner_state.pitch.ref_vel_deg_s;
    } else {
      desired_yaw = planner_state.yaw.ref_pos_deg;
      desired_pitch = planner_state.pitch.ref_pos_deg;
      planner_state.last_prediction_delay_s = 0.0;
      planner_state.last_pred_target_yaw_deg = desired_yaw;
      planner_state.last_pred_target_pitch_deg = desired_pitch;
    }
  } else if (mode_ == ControlMode::LOST_RECOVER) {
    if (hold_last_auto_after_loss) {
      desired_yaw = hold_auto_yaw_deg_;
      desired_pitch = hold_auto_pitch_deg_;
    } else if (search_stage_mode_ == "scan") {
      if (!recover_scan_state.initialized) {
        resetRecoverScanState(recover_scan_state, open_loop_yaw, open_loop_pitch, now);
      }

      bool rough_fresh = false;
      rclcpp::Time rough_stamp(0, 0, RCL_ROS_TIME);
      if (has_rough && rough_stamp_ns > 0) {
        const double rough_age_s = (now.nanoseconds() - rough_stamp_ns) * 1e-9;
        rough_fresh = rough_age_s <= recover_scan_params.rough_timeout_s;
        if (rough_fresh) {
          rough_stamp = rclcpp::Time(rough_stamp_ns, RCL_ROS_TIME);
        }
      }

      if (rough_fresh) {
        updateAxisMeasurement(
          recover_scan_state.center_yaw,
          rough_yaw,
          rough_stamp,
          recover_scan_params.rough_measurement_alpha,
          recover_scan_params.rough_velocity_alpha);

        updateAxisMeasurement(
          recover_scan_state.center_pitch,
          rough_pitch,
          rough_stamp,
          recover_scan_params.rough_measurement_alpha,
          recover_scan_params.rough_velocity_alpha);
      }

      double center_target_yaw = recover_scan_state.center_yaw.ref_pos_deg;
      double center_target_pitch = recover_scan_state.center_pitch.ref_pos_deg;

      if (recover_scan_state.center_yaw.has_measurement) {
        center_target_yaw = predictAxisTarget(
          recover_scan_state.center_yaw,
          now,
          recover_scan_params.rough_prediction_delay_s,
          recover_scan_params.rough_prediction_max_lead_deg);
      }
      if (recover_scan_state.center_pitch.has_measurement) {
        center_target_pitch = predictAxisTarget(
          recover_scan_state.center_pitch,
          now,
          recover_scan_params.rough_prediction_delay_s,
          recover_scan_params.rough_prediction_max_lead_deg);
      }

      stepAxisReference(
        recover_scan_state.center_yaw,
        center_target_yaw,
        dt_s,
        recover_scan_params.center_max_vel_deg_s,
        recover_scan_params.center_max_acc_deg_s2);

      stepAxisReference(
        recover_scan_state.center_pitch,
        center_target_pitch,
        dt_s,
        recover_scan_params.center_max_vel_deg_s,
        recover_scan_params.center_max_acc_deg_s2);

      const double center_ref_yaw = recover_scan_state.center_yaw.ref_pos_deg;
      const double center_ref_pitch = recover_scan_state.center_pitch.ref_pos_deg;

      const bool has_scan_center =
        recover_scan_state.center_yaw.has_measurement &&
        recover_scan_state.center_pitch.has_measurement;
      const double env_target_yaw =
        has_scan_center ? recover_scan_params.yaw_env_max : 0.0;
      const double env_target_pitch =
        has_scan_center ? 1.0 : 0.0;

      const double env_up_step =
        dt_s / std::max(0.05, recover_scan_params.envelope_rise_s);
      const double env_down_step =
        dt_s / std::max(0.05, recover_scan_params.envelope_fall_s);

      if (env_target_yaw >= recover_scan_state.scan_env_yaw) {
        recover_scan_state.scan_env_yaw = moveToward(
          recover_scan_state.scan_env_yaw, env_target_yaw, env_up_step);
      } else {
        recover_scan_state.scan_env_yaw = moveToward(
          recover_scan_state.scan_env_yaw, env_target_yaw, env_down_step);
      }
      recover_scan_state.scan_env_yaw =
        clampValue(recover_scan_state.scan_env_yaw, 0.0, recover_scan_params.yaw_env_max);

      if (env_target_pitch >= recover_scan_state.scan_env_pitch) {
        recover_scan_state.scan_env_pitch = moveToward(
          recover_scan_state.scan_env_pitch, env_target_pitch, env_up_step);
      } else {
        recover_scan_state.scan_env_pitch = moveToward(
          recover_scan_state.scan_env_pitch, env_target_pitch, env_down_step);
      }
      recover_scan_state.scan_env_pitch =
        clampValue(recover_scan_state.scan_env_pitch, 0.0, 1.0);

      const double base_w = 2.0 * kPi / std::max(1e-6, scan_period_s_);
      const double wyaw = base_w * recover_scan_params.yaw_freq_scale;
      const double wpitch = base_w * recover_scan_params.pitch_freq_scale;

      recover_scan_state.scan_phase_yaw_rad =
        wrapAngleRad(recover_scan_state.scan_phase_yaw_rad + wyaw * dt_s);
      recover_scan_state.scan_phase_pitch_rad =
        wrapAngleRad(recover_scan_state.scan_phase_pitch_rad + wpitch * dt_s);

      const double yaw_wave =
        std::sin(recover_scan_state.scan_phase_yaw_rad + scan_phase_x_rad_);
      const double pitch_wave =
        std::sin(recover_scan_state.scan_phase_pitch_rad + scan_phase_y_rad_);

      const double scan_yaw =
        recover_scan_state.scan_env_yaw *
        (scan_window_half_yaw_deg_ * recover_scan_params.yaw_amp_scale) *
        yaw_wave;

      const double pitch_amp =
        (pitch_wave >= 0.0)
        ? (scan_window_half_pitch_deg_ * recover_scan_params.pitch_amp_up_scale)
        : (scan_window_half_pitch_deg_ * recover_scan_params.pitch_amp_down_scale);

      const double scan_pitch =
        recover_scan_state.scan_env_pitch *
        (recover_scan_params.pitch_up_bias_deg + pitch_wave * pitch_amp);

      desired_yaw = center_ref_yaw + scan_yaw;
      desired_pitch = center_ref_pitch + scan_pitch;

      recover_scan_state.last_center_target_yaw_deg = center_target_yaw;
      recover_scan_state.last_center_target_pitch_deg = center_target_pitch;
      recover_scan_state.last_center_ref_yaw_deg = center_ref_yaw;
      recover_scan_state.last_center_ref_pitch_deg = center_ref_pitch;
      recover_scan_state.last_scan_yaw_deg = scan_yaw;
      recover_scan_state.last_scan_pitch_deg = scan_pitch;
    } else {
      if (has_rough) {
        desired_yaw = rough_yaw;
        desired_pitch = rough_pitch;
      } else if (has_hold_auto_cmd_) {
        desired_yaw = hold_auto_yaw_deg_;
        desired_pitch = hold_auto_pitch_deg_;
      } else {
        desired_yaw = open_loop_yaw;
        desired_pitch = open_loop_pitch;
      }
    }
  } else {
    if (has_hold_auto_cmd_) {
      desired_yaw = hold_auto_yaw_deg_;
      desired_pitch = hold_auto_pitch_deg_;
    } else {
      desired_yaw = open_loop_yaw;
      desired_pitch = open_loop_pitch;
    }
  }

  double auto_yaw = desired_yaw;
  double auto_pitch = desired_pitch;

  const bool recover_scan_mode =
    mode_ == ControlMode::LOST_RECOVER && search_stage_mode_ == "scan";
  if (recover_scan_mode) {
    command_vel_yaw_deg_s_ = 0.0;
    command_vel_pitch_deg_s_ = 0.0;
  } else if (mode_ == ControlMode::LOST_RECOVER || mode_ == ControlMode::HOLD) {
    const double max_vel =
      (mode_ == ControlMode::LOST_RECOVER) ? recover_max_vel_deg_s_ : fine_track_max_vel_deg_s_;
    const double max_acc =
      (mode_ == ControlMode::LOST_RECOVER) ? recover_max_acc_deg_s2_ : fine_track_max_acc_deg_s2_;

    auto_yaw = computeVelocityLimitedAxisCommand(
      desired_yaw, open_loop_yaw, command_vel_yaw_deg_s_, dt_s, max_vel, max_acc, stop_error_deg_);
    auto_pitch = computeVelocityLimitedAxisCommand(
      desired_pitch, open_loop_pitch, command_vel_pitch_deg_s_, dt_s, max_vel, max_acc, stop_error_deg_);
  }

  if (enable_output_filter_) {
    double state_alpha = filter_params.hold_alpha;
    switch (mode_) {
      case ControlMode::FINE_TRACK:
        state_alpha = filter_params.track_alpha;
        break;
      case ControlMode::PNP_ONLY_TRACK:
        state_alpha = filter_params.pnp_only_alpha;
        break;
      case ControlMode::LOST_RECOVER:
        state_alpha =
          (search_stage_mode_ == "scan") ? filter_params.scan_alpha : filter_params.recover_alpha;
        break;
      case ControlMode::HOLD:
      default:
        state_alpha = filter_params.hold_alpha;
        break;
    }
    state_alpha = clampValue(state_alpha, 0.0, 1.0);

    if (state_alpha >= 0.999) {
      has_last_output_ = false;
    } else {
      if (!has_last_output_) {
        last_yaw_out_deg_ = auto_yaw;
        last_pitch_out_deg_ = auto_pitch;
        has_last_output_ = true;
      } else {
        last_yaw_out_deg_ = state_alpha * auto_yaw + (1.0 - state_alpha) * last_yaw_out_deg_;
        last_pitch_out_deg_ = state_alpha * auto_pitch + (1.0 - state_alpha) * last_pitch_out_deg_;
      }
      auto_yaw = last_yaw_out_deg_;
      auto_pitch = last_pitch_out_deg_;
    }
  } else {
    has_last_output_ = false;
  }

  last_auto_cmd_yaw_deg_ = auto_yaw;
  last_auto_cmd_pitch_deg_ = auto_pitch;
  has_auto_command_reference_ = true;

  if (tracking_mode) {
    hold_auto_yaw_deg_ = auto_yaw;
    hold_auto_pitch_deg_ = auto_pitch;
    has_hold_auto_cmd_ = true;
    has_tracked_target_auto_cmd_ = true;
  }

  double final_yaw = auto_yaw;
  double final_pitch = auto_pitch;

  const bool allow_manual_in_final =
    planner_params.manual_always_add ||
    ((mode_ == ControlMode::HOLD || mode_ == ControlMode::LOST_RECOVER) &&
     (search_stage_mode_ == "manual"));

  if (allow_manual_in_final && has_manual) {
    final_yaw += manual_yaw;
    final_pitch += manual_pitch;
  }

  publishCommand(final_yaw, final_pitch);
  last_mode_ = mode_;

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[PIXEL_RAW] valid=%s raw_uv=(%.2f, %.2f) filt_uv=(%.2f, %.2f) laser_uv=(%.2f, %.2f) "
      "du=%.2f dv=%.2f err_px=%.2f avg_alpha=%.3f zone=%s",
      (pixel_valid && has_filtered_pixel_) ? "true" : "false",
      obs.u, obs.v,
      has_filtered_pixel_ ? filtered_target_u_ : 0.0,
      has_filtered_pixel_ ? filtered_target_v_ : 0.0,
      active_laser_pixel_u, active_laser_pixel_v,
      pixel_du, pixel_dv,
      pixel_err_px, filter_alpha,
      zone_name.c_str());

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[PIXEL_SOLVE] "
      "target_ray=(yaw=%.4f pitch=%.4f) laser_ray=(yaw=%.4f pitch=%.4f) "
      "err_deg=(yaw=%.4f pitch=%.4f) "
      "full_inc=(yaw=%.4f pitch=%.4f) "
      "corr_pre=(yaw=%.4f pitch=%.4f) "
      "corr_deadzone=(yaw=%.4f pitch=%.4f) "
      "corr_clamp=(yaw=%.4f pitch=%.4f) "
      "tol=(yaw=%.4f pitch=%.4f) deadzone=%.4f scale=%.4f step_max=%.4f",
      target_yaw_ray_deg, target_pitch_ray_deg,
      laser_yaw_ray_deg, laser_pitch_ray_deg,
      pixel_err_yaw_deg, pixel_err_pitch_deg,
      pixel_full_inc_yaw_deg, pixel_full_inc_pitch_deg,
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
      "assist_bias=(yaw=%.4f pitch=%.4f) consistent=%s",
      inc_pnp_ok ? "true" : "false",
      pnp_has_pose_dbg ? "true" : "false",
      obs.tx, obs.ty, obs.tz,
      target_cam_dbg.x(), target_cam_dbg.y(), target_cam_dbg.z(),
      target_from_laser.x(), target_from_laser.y(), target_from_laser.z(),
      laser_offset_x_, laser_offset_y_, laser_offset_z_,
      inc_pnp.yaw_deg, inc_pnp.pitch_deg,
      pnp_bias_yaw_deg_, pnp_bias_pitch_deg_,
      planner_state.last_pnp_consistent ? "true" : "false");

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "[TRACK_PLAN] src=%s meas=(%.3f, %.3f) meas_vel=(%.3f, %.3f) "
      "delay=%.3f pred=(%.3f, %.3f) ref=(%.3f, %.3f) ref_vel=(%.3f, %.3f)",
      planner_state.last_measurement_source.c_str(),
      planner_state.last_track_meas_yaw_deg,
      planner_state.last_track_meas_pitch_deg,
      planner_state.yaw.meas_vel_deg_s,
      planner_state.pitch.meas_vel_deg_s,
      planner_state.last_prediction_delay_s,
      planner_state.last_pred_target_yaw_deg,
      planner_state.last_pred_target_pitch_deg,
      planner_state.yaw.ref_pos_deg,
      planner_state.pitch.ref_pos_deg,
      planner_state.yaw.ref_vel_deg_s,
      planner_state.pitch.ref_vel_deg_s);

    if (mode_ == ControlMode::LOST_RECOVER && search_stage_mode_ == "scan") {
      const bool rough_fresh_dbg =
        has_rough &&
        rough_stamp_ns > 0 &&
        ((now.nanoseconds() - rough_stamp_ns) * 1e-9 <= recover_scan_params.rough_timeout_s);

      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 100,
        "[RECOVER_SCAN] rough_fresh=%s center_target=(%.3f, %.3f) center_ref=(%.3f, %.3f) "
        "scan=(%.3f, %.3f) env=(yaw=%.3f pitch=%.3f) phase=(%.3f, %.3f)",
        rough_fresh_dbg ? "true" : "false",
        recover_scan_state.last_center_target_yaw_deg,
        recover_scan_state.last_center_target_pitch_deg,
        recover_scan_state.last_center_ref_yaw_deg,
        recover_scan_state.last_center_ref_pitch_deg,
        recover_scan_state.last_scan_yaw_deg,
        recover_scan_state.last_scan_pitch_deg,
        recover_scan_state.scan_env_yaw,
        recover_scan_state.scan_env_pitch,
        recover_scan_state.scan_phase_yaw_rad,
        recover_scan_state.scan_phase_pitch_rad);
    }

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 200,
      "mode=%s search_mode=%s target_fresh=%s has_target=%s pixel_valid=%s pnp_valid=%s has_rough=%s "
      "lost_hold=(%s %.2f/%.2f) "
      "gimbal=(%.3f, %.3f) hold_auto=(%.3f, %.3f) rough=(%.3f, %.3f) "
      "laser_ref=(%.1f, %.1f) dist=%.2f "
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
      hold_last_auto_after_loss ? "active" :
        (lost_auto_hold_active_ ? "expired" : "idle"),
      lost_auto_hold_age_s,
      lost_auto_hold_timeout_s_,
      gimbal.yaw_deg,
      gimbal.pitch_deg,
      has_hold_auto_cmd_ ? hold_auto_yaw_deg_ : 0.0,
      has_hold_auto_cmd_ ? hold_auto_pitch_deg_ : 0.0,
      has_rough ? rough_yaw : 0.0,
      has_rough ? rough_pitch : 0.0,
      active_laser_pixel_u,
      active_laser_pixel_v,
      obs.distance,
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
