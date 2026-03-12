#include "laser_aim_controller/laser_aim_controller_node.hpp"

#include <cmath>
#include <vector>

namespace laser_aim_controller
{

LaserAimControllerNode::LaserAimControllerNode(const rclcpp::NodeOptions& options)
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

  RCLCPP_INFO(this->get_logger(), "laser_aim_controller_node started.");
  RCLCPP_INFO(this->get_logger(), "target_topic: %s", target_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_angles_topic: %s", gimbal_angles_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "gimbal_quaternion_topic: %s", gimbal_quaternion_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "rough_command_topic: %s", rough_command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "manual_command_topic: %s", manual_command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "command_topic: %s", command_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "output_mode: %s", output_mode_.c_str());
  RCLCPP_INFO(this->get_logger(), "fallback_to_rough_when_no_target: %s",
              fallback_to_rough_when_no_target_ ? "true" : "false");
}

void LaserAimControllerNode::declareAndLoadParameters()
{
  this->declare_parameter<std::string>("target_topic", "/auto_aim/target");
  this->declare_parameter<std::string>("gimbal_angles_topic", "/gimbal/angles");
  this->declare_parameter<std::string>("gimbal_quaternion_topic", "/gimbal/quaternion");
  this->declare_parameter<std::string>("rough_command_topic", "/gimbal/rough_command_angles");
  this->declare_parameter<std::string>("manual_command_topic", "/gimbal/command_angle_manual");
  this->declare_parameter<std::string>("command_topic", "/gimbal/command_angles");

  this->declare_parameter<std::string>("gimbal_angle_order", "pitch_yaw");
  this->declare_parameter<std::string>("output_mode", "increment");

  this->declare_parameter<bool>("publish_debug", true);
  this->declare_parameter<bool>("enable_output_filter", true);
  this->declare_parameter<bool>("fallback_to_rough_when_no_target", true);

  this->declare_parameter<double>("output_filter_alpha", 0.3);
  this->declare_parameter<double>("min_valid_z_m", 0.05);

  this->declare_parameter<std::vector<double>>(
    "camera_offset_xyz", std::vector<double>{0.0, 0.0, 0.0});
  this->declare_parameter<std::vector<double>>(
    "laser_offset_xyz", std::vector<double>{0.0, -0.05, 0.0});

  this->declare_parameter<double>("yaw_inc_min_deg", -10.0);
  this->declare_parameter<double>("yaw_inc_max_deg", 10.0);
  this->declare_parameter<double>("pitch_inc_min_deg", -10.0);
  this->declare_parameter<double>("pitch_inc_max_deg", 10.0);

  target_topic_ = this->get_parameter("target_topic").as_string();
  gimbal_angles_topic_ = this->get_parameter("gimbal_angles_topic").as_string();
  gimbal_quaternion_topic_ = this->get_parameter("gimbal_quaternion_topic").as_string();
  rough_command_topic_ = this->get_parameter("rough_command_topic").as_string();
  manual_command_topic_ = this->get_parameter("manual_command_topic").as_string();
  command_topic_ = this->get_parameter("command_topic").as_string();

  gimbal_angle_order_ = this->get_parameter("gimbal_angle_order").as_string();
  output_mode_ = this->get_parameter("output_mode").as_string();

  publish_debug_ = this->get_parameter("publish_debug").as_bool();
  enable_output_filter_ = this->get_parameter("enable_output_filter").as_bool();
  fallback_to_rough_when_no_target_ =
    this->get_parameter("fallback_to_rough_when_no_target").as_bool();

  output_filter_alpha_ = this->get_parameter("output_filter_alpha").as_double();
  min_valid_z_m_ = this->get_parameter("min_valid_z_m").as_double();

  const auto cam_offset = this->get_parameter("camera_offset_xyz").as_double_array();
  const auto laser_offset = this->get_parameter("laser_offset_xyz").as_double_array();

  if (cam_offset.size() != 3 || laser_offset.size() != 3) {
    throw std::runtime_error("camera_offset_xyz and laser_offset_xyz must both have size 3");
  }

  laser_geometry_.setCameraOffset(Eigen::Vector3d(cam_offset[0], cam_offset[1], cam_offset[2]));
  laser_geometry_.setLaserOffset(Eigen::Vector3d(laser_offset[0], laser_offset[1], laser_offset[2]));

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

void LaserAimControllerNode::gimbalAnglesCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }

  if (gimbal_angle_order_ == "pitch_yaw") {
    current_gimbal_.pitch_deg = msg->data[0];
    current_gimbal_.yaw_deg = msg->data[1];
  } else if (gimbal_angle_order_ == "yaw_pitch") {
    current_gimbal_.yaw_deg = msg->data[0];
    current_gimbal_.pitch_deg = msg->data[1];
  } else {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Invalid gimbal_angle_order: %s", gimbal_angle_order_.c_str());
    return;
  }

  current_gimbal_.valid =
    std::isfinite(current_gimbal_.yaw_deg) &&
    std::isfinite(current_gimbal_.pitch_deg);
}

void LaserAimControllerNode::quaternionCallback(
  const geometry_msgs::msg::QuaternionStamped::SharedPtr)
{
  // 第一版保留接口，不参与主控制链
}

void LaserAimControllerNode::roughCommandCallback(
  const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }

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

  manual_yaw_deg_ = msg->data[0];
  manual_pitch_deg_ = msg->data[1];
  has_manual_cmd_ = std::isfinite(manual_yaw_deg_) && std::isfinite(manual_pitch_deg_);

  if (!has_manual_cmd_) {
    manual_yaw_deg_ = 0.0;
    manual_pitch_deg_ = 0.0;
  }
}

void LaserAimControllerNode::applyManualOffset(double& yaw_deg, double& pitch_deg) const
{
  if (!has_manual_cmd_) {
    return;
  }

  yaw_deg += manual_yaw_deg_;
  pitch_deg += manual_pitch_deg_;
}

void LaserAimControllerNode::publishCommand(double yaw_deg, double pitch_deg)
{
  applyManualOffset(yaw_deg, pitch_deg);

  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(2);
  msg.data[0] = yaw_deg;
  msg.data[1] = pitch_deg;
  cmd_pub_->publish(msg);
}

void LaserAimControllerNode::publishRoughIfAvailable()
{
  if (!fallback_to_rough_when_no_target_) {
    return;
  }

  if (!has_rough_cmd_) {
    return;
  }

  double out_yaw = rough_yaw_deg_;
  double out_pitch = rough_pitch_deg_;
  applyManualOffset(out_yaw, out_pitch);
  publishCommand(rough_yaw_deg_, rough_pitch_deg_);

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 500,
      "no valid fine target, fallback to rough command: base=(yaw=%.2f pitch=%.2f), "
      "manual=(yaw=%.2f pitch=%.2f), final=(yaw=%.2f pitch=%.2f)",
      rough_yaw_deg_, rough_pitch_deg_,
      has_manual_cmd_ ? manual_yaw_deg_ : 0.0,
      has_manual_cmd_ ? manual_pitch_deg_ : 0.0,
      out_yaw, out_pitch);
  }
}

void LaserAimControllerNode::targetCallback(
  const auto_aim_fast::msg::AimTarget::SharedPtr msg)
{
  if (!msg->has_target || !msg->has_pose) {
    publishRoughIfAvailable();
    return;
  }

  if (!isFinite3(msg->tx, msg->ty, msg->tz)) {
    publishRoughIfAvailable();
    return;
  }

  if (msg->tz <= min_valid_z_m_) {
    publishRoughIfAvailable();
    return;
  }

  const Eigen::Vector3d target_cam(msg->tx, msg->ty, msg->tz);
  const Eigen::Vector3d target_from_laser = laser_geometry_.targetFromLaser(target_cam);

  CommandAngles inc = angle_solver_.solveIncrement(target_from_laser);
  if (!inc.valid) {
    publishRoughIfAvailable();
    return;
  }

  double out_yaw = inc.yaw_deg;
  double out_pitch = inc.pitch_deg;

  if (output_mode_ == "absolute") {
    if (!current_gimbal_.valid) {
      publishRoughIfAvailable();
      return;
    }
    out_yaw = current_gimbal_.yaw_deg + inc.yaw_deg;
    out_pitch = current_gimbal_.pitch_deg + inc.pitch_deg;
  } else if (output_mode_ != "increment") {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Invalid output_mode: %s, expected 'increment' or 'absolute'. Using increment.",
      output_mode_.c_str());
  }

  if (enable_output_filter_) {
    const double a = output_filter_alpha_;

    if (!has_last_output_) {
      last_yaw_out_deg_ = out_yaw;
      last_pitch_out_deg_ = out_pitch;
      has_last_output_ = true;
    } else {
      last_yaw_out_deg_ = a * out_yaw + (1.0 - a) * last_yaw_out_deg_;
      last_pitch_out_deg_ = a * out_pitch + (1.0 - a) * last_pitch_out_deg_;
    }

    out_yaw = last_yaw_out_deg_;
    out_pitch = last_pitch_out_deg_;
  }

  const double base_yaw = out_yaw;
  const double base_pitch = out_pitch;

  applyManualOffset(out_yaw, out_pitch);
  publishCommand(base_yaw, base_pitch);

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "fine target: target_cam=(%.3f, %.3f, %.3f), target_from_laser=(%.3f, %.3f, %.3f), "
      "inc=(yaw=%.2f, pitch=%.2f), base=(yaw=%.2f, pitch=%.2f), "
      "manual=(yaw=%.2f, pitch=%.2f), final=(yaw=%.2f, pitch=%.2f)",
      msg->tx, msg->ty, msg->tz,
      target_from_laser.x(), target_from_laser.y(), target_from_laser.z(),
      inc.yaw_deg, inc.pitch_deg,
      base_yaw, base_pitch,
      has_manual_cmd_ ? manual_yaw_deg_ : 0.0,
      has_manual_cmd_ ? manual_pitch_deg_ : 0.0,
      out_yaw, out_pitch);
  }
}

}  // namespace laser_aim_controller