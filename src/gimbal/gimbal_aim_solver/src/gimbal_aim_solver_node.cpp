#include "gimbal_aim_solver/gimbal_aim_solver_node.hpp"

#include <cmath>
#include <vector>
#include <std_msgs/msg/bool.hpp>

#include "radar_decision_comm/common/team_identity.hpp"

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

  referee_identity_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
    referee_identity_topic_, 10,
    std::bind(&GimbalAimSolverNode::refereeIdentityCallback, this, std::placeholders::_1));

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
  this->declare_parameter<std::string>("referee_identity_topic", "/radar/referee/robot_identity");

  this->declare_parameter<std::string>("target_side", "red");
  this->declare_parameter<std::string>("gimbal_angle_order", "yaw_pitch");
  this->declare_parameter<bool>("use_referee_identity_target_side", true);

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

  // 新增：激光发射点相对于云台解算原点，在“云台本体系”下的安装偏置
  this->declare_parameter<std::vector<double>>(
    "laser_offset_xyz", std::vector<double>{0.0, 0.0, 0.0});

  this->declare_parameter<double>("yaw_offset_deg", 0.0);
  this->declare_parameter<double>("pitch_offset_deg", 0.0);

  this->declare_parameter<double>("yaw_min_deg", -180.0);
  this->declare_parameter<double>("yaw_max_deg", 180.0);
  this->declare_parameter<double>("pitch_min_deg", -60.0);
  this->declare_parameter<double>("pitch_max_deg", 60.0);

  uav_topic_ = this->get_parameter("uav_topic").as_string();
  gimbal_angles_topic_ = this->get_parameter("gimbal_angles_topic").as_string();
  target_angles_topic_ = this->get_parameter("target_angles_topic").as_string();
  referee_identity_topic_ = this->get_parameter("referee_identity_topic").as_string();

  target_side_ = this->get_parameter("target_side").as_string();
  gimbal_angle_order_ = this->get_parameter("gimbal_angle_order").as_string();
  use_referee_identity_target_side_ =
    this->get_parameter("use_referee_identity_target_side").as_bool();

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

  const auto laser_offset_xyz = this->get_parameter("laser_offset_xyz").as_double_array();
  if (laser_offset_xyz.size() != 3) {
    RCLCPP_FATAL(this->get_logger(), "laser_offset_xyz must have size 3.");
    throw std::runtime_error("laser_offset_xyz size error");
  }

  laser_offset_xyz_ = Eigen::Vector3d(
    laser_offset_xyz[0],
    laser_offset_xyz[1],
    laser_offset_xyz[2]);

  const double yaw_offset_deg = this->get_parameter("yaw_offset_deg").as_double();
  const double pitch_offset_deg = this->get_parameter("pitch_offset_deg").as_double();
  angle_solver_.setOffsets(yaw_offset_deg, pitch_offset_deg);

  const double yaw_min_deg = this->get_parameter("yaw_min_deg").as_double();
  const double yaw_max_deg = this->get_parameter("yaw_max_deg").as_double();
  const double pitch_min_deg = this->get_parameter("pitch_min_deg").as_double();
  const double pitch_max_deg = this->get_parameter("pitch_max_deg").as_double();

  angle_solver_.setLimits(yaw_min_deg, yaw_max_deg, pitch_min_deg, pitch_max_deg);
}

void GimbalAimSolverNode::refereeIdentityCallback(
  const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
{
  if (!use_referee_identity_target_side_ || !msg || !msg->valid) {
    return;
  }

  const std::string next_target_side =
    radar_decision_comm::team_identity::enemy_target_side_from_self_color(msg->self_color);
  if (next_target_side == target_side_) {
    return;
  }

  target_side_ = next_target_side;
  RCLCPP_INFO(
    this->get_logger(),
    "gimbal_aim_solver switched target_side=%s from referee robot_id=%u",
    target_side_.c_str(),
    msg->robot_id);
}

bool GimbalAimSolverNode::isFinitePoint(const Eigen::Vector3d& p) const
{
  return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

Eigen::Matrix3d GimbalAimSolverNode::rotationFromSolvedYawPitchDeg(
  double yaw_deg, double pitch_deg) const
{
  // 与 AngleSolver::solve() 保持一致的坐标约定：
  // x = right, y = down, z = forward
  // yaw = atan2(-x, z)     => 正yaw朝左
  // pitch = -atan2(y, ...) => 正pitch朝上
  //
  // 因此从“云台本体系”旋到“静止初始云台系”的旋转可写为：
  // R = Ry(-yaw) * Rx(pitch)

  const double yaw_rad = yaw_deg * M_PI / 180.0;
  const double pitch_rad = pitch_deg * M_PI / 180.0;

  const Eigen::AngleAxisd yaw_rot(-yaw_rad, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd pitch_rot(pitch_rad, Eigen::Vector3d::UnitX());

  return (yaw_rot * pitch_rot).toRotationMatrix();
}

bool GimbalAimSolverNode::selectTargetWorldPoint(
  const geometry_msgs::msg::PoseArray::SharedPtr& msg,
  Eigen::Vector3d& p_world,
  bool& target_observed)
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

  const bool red_observed = msg->poses[1].orientation.w > 0.5;
  const bool blue_observed = msg->poses[0].orientation.w > 0.5;

  const auto is_zero_point = [](const Eigen::Vector3d& p) {
    constexpr double kZeroPointEps = 1e-6;
    return std::abs(p.x()) <= kZeroPointEps &&
           std::abs(p.y()) <= kZeroPointEps &&
           std::abs(p.z()) <= kZeroPointEps;
  };

  const bool red_valid = isFinitePoint(red_p);
  const bool blue_valid = isFinitePoint(blue_p);
  const bool red_zero = is_zero_point(red_p);
  const bool blue_zero = is_zero_point(blue_p);

  if (target_side_ == "red") {
    if (!red_valid || red_zero) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "target_side=red but red position invalid. "
        "red=(%.3f, %.3f, %.3f), red_zero=%s, blue_valid=%s, blue_zero=%s",
        red_p.x(), red_p.y(), red_p.z(),
        red_zero ? "true" : "false",
        blue_valid ? "true" : "false",
        blue_zero ? "true" : "false");
      return false;
    }

    p_world = red_p;
    target_observed = red_observed;
    return true;
  }

  if (target_side_ == "blue") {
    if (!blue_valid || blue_zero) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "target_side=blue but blue position invalid. "
        "blue=(%.3f, %.3f, %.3f), blue_zero=%s, red_valid=%s, red_zero=%s",
        blue_p.x(), blue_p.y(), blue_p.z(),
        blue_zero ? "true" : "false",
        red_valid ? "true" : "false",
        red_zero ? "true" : "false");
      return false;
    }

    p_world = blue_p;
    target_observed = blue_observed;
    return true;
  }

  RCLCPP_WARN_THROTTLE(
    this->get_logger(), *this->get_clock(), 1000,
    "target_side must be 'red' or 'blue', got: %s",
    target_side_.c_str());
  return false;
}

void GimbalAimSolverNode::publishTargetAngles(
  double yaw_deg,
  double pitch_deg,
  bool target_observed) const
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(3);
  msg.data[0] = yaw_deg;
  msg.data[1] = pitch_deg;
  msg.data[2] = target_observed ? 1.0 : 0.0;
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
  bool target_observed = false;
  if (!selectTargetWorldPoint(msg, p_world, target_observed)) {
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

  // 第一次粗解：原始“前向轴对准目标”
  TargetAngles raw_target = angle_solver_.solve(p_gimbal);
  if (!raw_target.valid) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "raw target invalid, likely behind camera/gimbal. p_gimbal=(%.3f, %.3f, %.3f)",
      p_gimbal.x(), p_gimbal.y(), p_gimbal.z());
    return;
  }

  // 把激光安装偏置从“云台本体系”旋到“静止初始云台系”
  const Eigen::Matrix3d R_gimbal = rotationFromSolvedYawPitchDeg(
    raw_target.yaw_deg, raw_target.pitch_deg);
  const Eigen::Vector3d laser_offset_in_static_gimbal = R_gimbal * laser_offset_xyz_;

  // 关键补偿：让“激光发射点 -> 目标”的向量进入解算
  const Eigen::Vector3d p_from_laser = p_gimbal - laser_offset_in_static_gimbal;
  if (!isFinitePoint(p_from_laser)) {
    RCLCPP_WARN(this->get_logger(), "laser-relative point has NaN/Inf");
    return;
  }

  TargetAngles target = angle_solver_.solve(p_from_laser);
  if (!target.valid) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "laser-corrected target invalid. p_from_laser=(%.3f, %.3f, %.3f)",
      p_from_laser.x(), p_from_laser.y(), p_from_laser.z());
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

  publishTargetAngles(out_yaw_deg, out_pitch_deg, target_observed);

  if (publish_debug_) {
    if (use_increment_output_) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 200,
        "[%s] world=(%.3f, %.3f, %.3f), camera=(%.3f, %.3f, %.3f), gimbal=(%.3f, %.3f, %.3f), "
        "laser_rel=(%.3f, %.3f, %.3f), raw_abs=(yaw=%.2f, pitch=%.2f), "
        "target_abs=(yaw=%.2f, pitch=%.2f), current=(yaw=%.2f, pitch=%.2f), "
        "out_inc=(yaw=%.2f, pitch=%.2f), observed=%s",
        target_side_.c_str(),
        p_world.x(), p_world.y(), p_world.z(),
        p_camera.x(), p_camera.y(), p_camera.z(),
        p_gimbal.x(), p_gimbal.y(), p_gimbal.z(),
        p_from_laser.x(), p_from_laser.y(), p_from_laser.z(),
        raw_target.yaw_deg, raw_target.pitch_deg,
        target.yaw_deg, target.pitch_deg,
        current_gimbal_state_.yaw_deg, current_gimbal_state_.pitch_deg,
        out_yaw_deg, out_pitch_deg,
        target_observed ? "true" : "false");
    } else {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 200,
        "[%s] world=(%.3f, %.3f, %.3f), camera=(%.3f, %.3f, %.3f), gimbal=(%.3f, %.3f, %.3f), "
        "laser_rel=(%.3f, %.3f, %.3f), raw_abs=(yaw=%.2f, pitch=%.2f), "
        "out_abs=(yaw=%.2f, pitch=%.2f), observed=%s",
        target_side_.c_str(),
        p_world.x(), p_world.y(), p_world.z(),
        p_camera.x(), p_camera.y(), p_camera.z(),
        p_gimbal.x(), p_gimbal.y(), p_gimbal.z(),
        p_from_laser.x(), p_from_laser.y(), p_from_laser.z(),
        raw_target.yaw_deg, raw_target.pitch_deg,
        out_yaw_deg, out_pitch_deg,
        target_observed ? "true" : "false");
    }
  }
}

}  // namespace gimbal_aim_solver
