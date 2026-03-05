#include "gimbal_aim_solver/gimbal_aim_solver_node.hpp"

#include <cmath>
#include <vector>
#include <std_msgs/msg/bool.hpp>

namespace gimbal_aim_solver
{

GimbalAimSolverNode::GimbalAimSolverNode(const rclcpp::NodeOptions& options)
: Node("gimbal_aim_solver_node", options)
{
  declareAndLoadParameters();

  uav_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
    uav_topic_, rclcpp::SensorDataQoS(),
    std::bind(&GimbalAimSolverNode::uavCallback, this, std::placeholders::_1));

  gimbal_angles_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    gimbal_angles_topic_, rclcpp::SensorDataQoS(),
    std::bind(&GimbalAimSolverNode::gimbalAnglesCallback, this, std::placeholders::_1));

  target_angles_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    target_angles_topic_, 10);

  gimbal_enable_pub_ = this->create_publisher<std_msgs::msg::Bool>(
  "/gimbal/enable", 10);

  RCLCPP_INFO(this->get_logger(), "gimbal_aim_solver_node started.");
  RCLCPP_INFO(this->get_logger(), "uav_topic: %s", uav_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_angles_topic: %s", gimbal_angles_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "target_angles_topic: %s", target_angles_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "target_side: %s", target_side_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_angle_order: %s", gimbal_angle_order_.c_str());
  RCLCPP_INFO(this->get_logger(), "use_increment_output: %s", use_increment_output_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "publish_debug: %s", publish_debug_ ? "true" : "false");
}

void GimbalAimSolverNode::declareAndLoadParameters()
{
  this->declare_parameter<std::string>("uav_topic", "/uav_positions");
  this->declare_parameter<std::string>("gimbal_angles_topic", "/gimbal/angles");
  this->declare_parameter<std::string>("target_angles_topic", "/gimbal/rough_command_angles");

  this->declare_parameter<std::string>("target_side", "red");
  this->declare_parameter<std::string>("gimbal_angle_order", "pitch_yaw");

  this->declare_parameter<bool>("publish_debug", true);
  this->declare_parameter<bool>("use_increment_output", false);

  this->declare_parameter<double>("max_uav_msg_age_sec", 0.5);
  this->declare_parameter<double>("max_gimbal_msg_age_sec", 0.5);

  this->declare_parameter<bool>("enable_output_filter", false);
  this->declare_parameter<double>("output_filter_alpha", 0.2);

  this->declare_parameter<std::vector<double>>(
    "world_rvec", std::vector<double>{1.3007738496052885, 1.0519496130346802, -1.0480104909702821});
  this->declare_parameter<std::vector<double>>(
    "world_tvec", std::vector<double>{-9.0641934014867864, 1.2351673073596281, 23.979868492980199});

  this->declare_parameter<std::vector<double>>(
    "camera_to_gimbal_xyz", std::vector<double>{-0.05, -0.20, 0.0});

  this->declare_parameter<double>("yaw_offset_deg", 0.0);
  this->declare_parameter<double>("pitch_offset_deg", 0.0);

  this->declare_parameter<double>("yaw_min_deg", -180.0);
  this->declare_parameter<double>("yaw_max_deg", 180.0);
  this->declare_parameter<double>("pitch_min_deg", -60.0);
  this->declare_parameter<double>("pitch_max_deg", 60.0);

  uav_topic_ = this->get_parameter("uav_topic").as_string();
  gimbal_angles_topic_ = this->get_parameter("gimbal_angles_topic").as_string();
  target_angles_topic_ = this->get_parameter("target_angles_topic").as_string();

  target_side_ = this->get_parameter("target_side").as_string();
  gimbal_angle_order_ = this->get_parameter("gimbal_angle_order").as_string();

  publish_debug_ = this->get_parameter("publish_debug").as_bool();
  use_increment_output_ = this->get_parameter("use_increment_output").as_bool();

  max_uav_msg_age_sec_ = this->get_parameter("max_uav_msg_age_sec").as_double();
  max_gimbal_msg_age_sec_ = this->get_parameter("max_gimbal_msg_age_sec").as_double();

  enable_output_filter_ = this->get_parameter("enable_output_filter").as_bool();
  output_filter_alpha_ = this->get_parameter("output_filter_alpha").as_double();

  const auto world_rvec = this->get_parameter("world_rvec").as_double_array();
  const auto world_tvec = this->get_parameter("world_tvec").as_double_array();

  const bool ok = extrinsic_transformer_.init(
    std::vector<double>(world_rvec.begin(), world_rvec.end()),
    std::vector<double>(world_tvec.begin(), world_tvec.end()));

  if (!ok) {
    RCLCPP_FATAL(this->get_logger(), "Failed to initialize extrinsic transformer.");
    throw std::runtime_error("Failed to initialize extrinsic transformer");
  }

  const auto camera_to_gimbal_xyz = this->get_parameter("camera_to_gimbal_xyz").as_double_array();
  if (camera_to_gimbal_xyz.size() != 3) {
    RCLCPP_FATAL(this->get_logger(), "camera_to_gimbal_xyz must have size 3.");
    throw std::runtime_error("camera_to_gimbal_xyz size error");
  }

  gimbal_geometry_.setCameraToGimbalTranslation(
    Eigen::Vector3d(camera_to_gimbal_xyz[0], camera_to_gimbal_xyz[1], camera_to_gimbal_xyz[2]));

  const double yaw_offset_deg = this->get_parameter("yaw_offset_deg").as_double();
  const double pitch_offset_deg = this->get_parameter("pitch_offset_deg").as_double();
  angle_solver_.setOffsets(yaw_offset_deg, pitch_offset_deg);

  const double yaw_min_deg = this->get_parameter("yaw_min_deg").as_double();
  const double yaw_max_deg = this->get_parameter("yaw_max_deg").as_double();
  const double pitch_min_deg = this->get_parameter("pitch_min_deg").as_double();
  const double pitch_max_deg = this->get_parameter("pitch_max_deg").as_double();

  angle_solver_.setLimits(yaw_min_deg, yaw_max_deg, pitch_min_deg, pitch_max_deg);
}

bool GimbalAimSolverNode::isFinitePoint(const Eigen::Vector3d& p) const
{
  return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

bool GimbalAimSolverNode::selectTargetWorldPoint(
  const geometry_msgs::msg::PoseArray::SharedPtr& msg,
  Eigen::Vector3d& p_world)
{
  if (msg->poses.size() < 2) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "uav_positions poses.size() < 2");
    return false;
  }

  const Eigen::Vector3d red_p(
    msg->poses[1].position.x,
    msg->poses[1].position.y,
    msg->poses[1].position.z);

  const Eigen::Vector3d blue_p(
    msg->poses[0].position.x,
    msg->poses[0].position.y,
    msg->poses[0].position.z);

  const bool red_valid = isFinitePoint(red_p);
  const bool blue_valid = isFinitePoint(blue_p);

  if (target_side_ == "red") {
    if (!red_valid) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "target_side=red but red position invalid. "
        "red=(%.3f, %.3f, %.3f), blue_valid=%s",
        red_p.x(), red_p.y(), red_p.z(),
        blue_valid ? "true" : "false");
      return false;
    }

    p_world = red_p;
    return true;
  }

  if (target_side_ == "blue") {
    if (!blue_valid) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "target_side=blue but blue position invalid. "
        "blue=(%.3f, %.3f, %.3f), red_valid=%s",
        blue_p.x(), blue_p.y(), blue_p.z(),
        red_valid ? "true" : "false");
      return false;
    }

    p_world = blue_p;
    return true;
  }

  RCLCPP_WARN_THROTTLE(
    this->get_logger(), *this->get_clock(), 1000,
    "target_side must be 'red' or 'blue', got: %s",
    target_side_.c_str());
  return false;
}

void GimbalAimSolverNode::publishTargetAngles(double yaw_deg, double pitch_deg) const
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(2);
  msg.data[0] = yaw_deg;
  msg.data[1] = pitch_deg;
  target_angles_pub_->publish(msg);

  std_msgs::msg::Bool enable_msg;
  enable_msg.data = true;
  gimbal_enable_pub_->publish(enable_msg);
}

void GimbalAimSolverNode::gimbalAnglesCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    RCLCPP_WARN(this->get_logger(), "/gimbal/angles data.size() < 2");
    return;
  }

  if (gimbal_angle_order_ == "pitch_yaw") {
    current_gimbal_state_.pitch_deg = msg->data[0];
    current_gimbal_state_.yaw_deg = msg->data[1];
  } else if (gimbal_angle_order_ == "yaw_pitch") {
    current_gimbal_state_.yaw_deg = msg->data[0];
    current_gimbal_state_.pitch_deg = msg->data[1];
  } else {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Invalid gimbal_angle_order: %s", gimbal_angle_order_.c_str());
    return;
  }

  if (!std::isfinite(current_gimbal_state_.yaw_deg) ||
      !std::isfinite(current_gimbal_state_.pitch_deg)) {
    RCLCPP_WARN(this->get_logger(), "/gimbal/angles contains NaN/Inf");
    return;
  }

  current_gimbal_state_.valid = true;
  last_gimbal_msg_time_ = this->now();
  has_gimbal_msg_time_ = true;
}

void GimbalAimSolverNode::uavCallback(
  const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  const rclcpp::Time now_time = this->now();

  if ((now_time - msg->header.stamp).seconds() > max_uav_msg_age_sec_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "/uav_positions too old: %.3f sec",
      (now_time - msg->header.stamp).seconds());
    return;
  }

  Eigen::Vector3d p_world;
  if (!selectTargetWorldPoint(msg, p_world)) {
    return;
  }

  const Eigen::Vector3d p_camera = extrinsic_transformer_.worldToCamera(p_world);
  if (!isFinitePoint(p_camera)) {
    RCLCPP_WARN(this->get_logger(), "camera point has NaN/Inf");
    return;
  }

  const Eigen::Vector3d p_gimbal = gimbal_geometry_.cameraToGimbal(p_camera);
  if (!isFinitePoint(p_gimbal)) {
    RCLCPP_WARN(this->get_logger(), "gimbal point has NaN/Inf");
    return;
  }

  TargetAngles target = angle_solver_.solve(p_gimbal);
  if (!target.valid) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "target invalid, likely behind camera/gimbal. p_gimbal=(%.3f, %.3f, %.3f)",
      p_gimbal.x(), p_gimbal.y(), p_gimbal.z());
    return;
  }

  double out_yaw_deg = target.yaw_deg;
  double out_pitch_deg = target.pitch_deg;

  if (use_increment_output_) {
    if (!current_gimbal_state_.valid || !has_gimbal_msg_time_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "increment mode enabled, but no valid current gimbal state.");
      return;
    }

    if ((now_time - last_gimbal_msg_time_).seconds() > max_gimbal_msg_age_sec_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "current gimbal state too old: %.3f sec",
        (now_time - last_gimbal_msg_time_).seconds());
      return;
    }

    out_yaw_deg = target.yaw_deg - current_gimbal_state_.yaw_deg;
    out_pitch_deg = target.pitch_deg - current_gimbal_state_.pitch_deg;
  }

  if (enable_output_filter_) {
    const double a = output_filter_alpha_;
    if (!has_last_output_) {
      last_output_yaw_deg_ = out_yaw_deg;
      last_output_pitch_deg_ = out_pitch_deg;
      has_last_output_ = true;
    } else {
      last_output_yaw_deg_ = a * out_yaw_deg + (1.0 - a) * last_output_yaw_deg_;
      last_output_pitch_deg_ = a * out_pitch_deg + (1.0 - a) * last_output_pitch_deg_;
    }
    out_yaw_deg = last_output_yaw_deg_;
    out_pitch_deg = last_output_pitch_deg_;
  }

  publishTargetAngles(out_yaw_deg, out_pitch_deg);

  if (publish_debug_) {
    if (use_increment_output_) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 200,
        "[%s] world=(%.3f, %.3f, %.3f), camera=(%.3f, %.3f, %.3f), gimbal=(%.3f, %.3f, %.3f), "
        "target_abs=(yaw=%.2f, pitch=%.2f), current=(yaw=%.2f, pitch=%.2f), out_inc=(yaw=%.2f, pitch=%.2f)",
        target_side_.c_str(),
        p_world.x(), p_world.y(), p_world.z(),
        p_camera.x(), p_camera.y(), p_camera.z(),
        p_gimbal.x(), p_gimbal.y(), p_gimbal.z(),
        target.yaw_deg, target.pitch_deg,
        current_gimbal_state_.yaw_deg, current_gimbal_state_.pitch_deg,
        out_yaw_deg, out_pitch_deg);
    } else {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 200,
        "[%s] world=(%.3f, %.3f, %.3f), camera=(%.3f, %.3f, %.3f), gimbal=(%.3f, %.3f, %.3f), "
        "out_abs=(yaw=%.2f, pitch=%.2f)",
        target_side_.c_str(),
        p_world.x(), p_world.y(), p_world.z(),
        p_camera.x(), p_camera.y(), p_camera.z(),
        p_gimbal.x(), p_gimbal.y(), p_gimbal.z(),
        out_yaw_deg, out_pitch_deg);
    }
  }
}

}  // namespace gimbal_aim_solver