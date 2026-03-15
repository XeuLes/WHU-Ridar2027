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
  this->declare_parameter<std::string>("gimbal_angles_topic", "/gimbal/angles_true");
  this->declare_parameter<std::string>("gimbal_quaternion_topic", "/gimbal/quaternion_true");
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

  this->declare_parameter<double>("fine_stack_max_abs_deg", 0.5);

  // 增量选择与像素法配置
  this->declare_parameter<std::string>("increment_mode", "auto"); // auto | pnp | pixel
  this->declare_parameter<double>("pnp_exceed_threshold_deg", 5.0);

  this->declare_parameter<double>("camera_fx", 1.0);
  this->declare_parameter<double>("camera_fy", 1.0);
  this->declare_parameter<double>("camera_cx", 0.0);
  this->declare_parameter<double>("camera_cy", 0.0);

  this->declare_parameter<std::vector<double>>("laser_pixel_uv", std::vector<double>{0.0, 0.0});

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

  fine_stack_max_abs_deg_ = this->get_parameter("fine_stack_max_abs_deg").as_double();

  increment_mode_ = this->get_parameter("increment_mode").as_string();
  pnp_exceed_threshold_deg_ = this->get_parameter("pnp_exceed_threshold_deg").as_double();

  camera_fx_ = this->get_parameter("camera_fx").as_double();
  camera_fy_ = this->get_parameter("camera_fy").as_double();
  camera_cx_ = this->get_parameter("camera_cx").as_double();
  camera_cy_ = this->get_parameter("camera_cy").as_double();

  const auto laser_uv = this->get_parameter("laser_pixel_uv").as_double_array();
  if (laser_uv.size() >= 2) {
    laser_pixel_u_ = laser_uv[0];
    laser_pixel_v_ = laser_uv[1];
  }

  if (cam_offset.size() != 3 || laser_offset.size() != 3) {
    throw std::runtime_error("camera_offset_xyz and laser_offset_xyz must both have size 3");
  }

  laser_geometry_.setCameraOffset(Eigen::Vector3d(cam_offset[0], cam_offset[1], cam_offset[2]));
  laser_geometry_.setLaserOffset(Eigen::Vector3d(laser_offset[0], laser_offset[1], laser_offset[2]));

  // 保存激光物理偏移到成员变量，供像素法按帧投影使用
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

  double final_yaw = rough_yaw_deg_;
  double final_pitch = rough_pitch_deg_;
  applyManualOffset(final_yaw, final_pitch);

  publishCommand(rough_yaw_deg_, rough_pitch_deg_);

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 500,
      "no valid fine target, fallback to rough command: base=(yaw=%.2f pitch=%.2f), "
      "manual=(yaw=%.2f pitch=%.2f), final=(yaw=%.2f pitch=%.2f)",
      rough_yaw_deg_, rough_pitch_deg_,
      has_manual_cmd_ ? manual_yaw_deg_ : 0.0,
      has_manual_cmd_ ? manual_pitch_deg_ : 0.0,
      final_yaw, final_pitch);
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

  // 先用 PnP 计算增量
  CommandAngles inc_pnp = angle_solver_.solveIncrement(target_from_laser);

  // 像素法计算增量（基于激光点像素与灯条中心像素距离）
  auto computePixelIncrement = [&](CommandAngles & out_inc) -> bool {
    // 像素法优先使用 PnP 的相机坐标 (tx,ty,tz) 投影为图像像素作为灯条中心，回退到 ROI 中心
    double target_u = 0.0, target_v = 0.0;
    bool have_target_pixel = false;
    if (msg->has_pose && msg->tz != 0.0) {
      // 使用相机坐标通过内参投影到像素平面（首选）
      if (camera_fx_ <= 0.0 || camera_fy_ <= 0.0) return false;
      target_u = camera_fx_ * static_cast<double>(msg->tx) / static_cast<double>(msg->tz) + camera_cx_;
      target_v = camera_fy_ * static_cast<double>(msg->ty) / static_cast<double>(msg->tz) + camera_cy_;
      have_target_pixel = true;
    } else if (msg->roi_w > 0 && msg->roi_h > 0) {
      // 回退：使用 ROI 中心作为灯条中心像素
      target_u = static_cast<double>(msg->roi_x) + static_cast<double>(msg->roi_w) * 0.5;
      target_v = static_cast<double>(msg->roi_y) + static_cast<double>(msg->roi_h) * 0.5;
      have_target_pixel = true;
    }

    if (!have_target_pixel) return false;
    if (camera_fx_ == 0.0 || camera_fy_ == 0.0) return false;

    // 优先使用基于实际深度的激光像素投影 (深度视差)
    double laser_u = laser_pixel_u_;
    double laser_v = laser_pixel_v_;
    if (msg->tz > min_valid_z_m_) {
      // laser_offset_x_/y_ 是激光相对于相机的物理偏移 (米)
      laser_u = camera_cx_ + camera_fx_ * (laser_offset_x_ / static_cast<double>(msg->tz));
      laser_v = camera_cy_ + camera_fy_ * (laser_offset_y_ / static_cast<double>(msg->tz));
    }

    const double du = target_u - laser_u;
    const double dv = target_v - laser_v;
    const double rad2deg = 57.29577951308232;

    out_inc.valid = true;
    out_inc.yaw_deg = std::atan2(du, camera_fx_) * rad2deg;
    out_inc.pitch_deg = std::atan2(dv, camera_fy_) * rad2deg;
    return true;
  };

  // 根据参数选择使用哪种方式
  CommandAngles inc_used;
  bool used_pixel_method = false;
  if (increment_mode_ == "pnp") {
    if (!inc_pnp.valid) {
      publishRoughIfAvailable();
      return;
    }
    inc_used = inc_pnp;
  } else if (increment_mode_ == "pixel") {
    CommandAngles inc_pixel;
    if (!computePixelIncrement(inc_pixel)) {
      publishRoughIfAvailable();
      return;
    }
    inc_used = inc_pixel;
    used_pixel_method = true;
  } else { // auto
    bool pnp_exceeds = !inc_pnp.valid ||
      (std::abs(inc_pnp.yaw_deg) > pnp_exceed_threshold_deg_) ||
      (std::abs(inc_pnp.pitch_deg) > pnp_exceed_threshold_deg_);

    if (pnp_exceeds) {
      CommandAngles inc_pixel;
      if (computePixelIncrement(inc_pixel)) {
        inc_used = inc_pixel;
        used_pixel_method = true;
      } else if (inc_pnp.valid) {
        inc_used = inc_pnp; // 回退到 PnP
      } else {
        publishRoughIfAvailable();
        return;
      }
    } else {
      // 使用 PnP
      inc_used = inc_pnp;
    }
  }

  double base_yaw = 0.0;
  double base_pitch = 0.0;

  bool fine_within_stack_limit =
    std::abs(inc_used.yaw_deg) <= fine_stack_max_abs_deg_ &&
    std::abs(inc_used.pitch_deg) <= fine_stack_max_abs_deg_;

  bool stacked_fine_on_rough = false;

  // 规则：
  // 1) 有粗瞄时，只有细瞄两个轴都不超过阈值，才叠加到粗瞄上
  // 2) 只要任一轴超过阈值，就不叠加细瞄，直接用粗瞄
  // 3) 没有粗瞄时，退化为原先细瞄输出
  if (has_rough_cmd_) {
    base_yaw = rough_yaw_deg_;
    base_pitch = rough_pitch_deg_;

    if (fine_within_stack_limit) {
      base_yaw += inc_used.yaw_deg;
      base_pitch += inc_used.pitch_deg;
      stacked_fine_on_rough = true;
    }
  } else {
    // 没有粗瞄时，保留原有细瞄独立输出逻辑
    base_yaw = inc_used.yaw_deg;
    base_pitch = inc_used.pitch_deg;

    if (output_mode_ == "absolute") {
      if (!current_gimbal_.valid) {
        publishRoughIfAvailable();
        return;
      }
  base_yaw = current_gimbal_.yaw_deg + inc_used.yaw_deg;
  base_pitch = current_gimbal_.pitch_deg + inc_used.pitch_deg;
    } else if (output_mode_ != "increment") {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Invalid output_mode: %s, expected 'increment' or 'absolute'. Using increment.",
        output_mode_.c_str());
    }
  }

  if (enable_output_filter_) {
    const double a = output_filter_alpha_;

    if (!has_last_output_) {
      last_yaw_out_deg_ = base_yaw;
      last_pitch_out_deg_ = base_pitch;
      has_last_output_ = true;
    } else {
      last_yaw_out_deg_ = a * base_yaw + (1.0 - a) * last_yaw_out_deg_;
      last_pitch_out_deg_ = a * base_pitch + (1.0 - a) * last_pitch_out_deg_;
    }

    base_yaw = last_yaw_out_deg_;
    base_pitch = last_pitch_out_deg_;
  }

  double final_yaw = base_yaw;
  double final_pitch = base_pitch;
  applyManualOffset(final_yaw, final_pitch);

  publishCommand(base_yaw, base_pitch);

  if (publish_debug_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 100,
      "fine target: target_cam=(%.3f, %.3f, %.3f), target_from_laser=(%.3f, %.3f, %.3f), "
      "inc=(yaw=%.2f, pitch=%.2f), rough=(yaw=%.2f, pitch=%.2f), "
      "stack_limit=%.2f, stacked=%s, base=(yaw=%.2f, pitch=%.2f), "
      "manual=(yaw=%.2f, pitch=%.2f), final=(yaw=%.2f, pitch=%.2f)",
      msg->tx, msg->ty, msg->tz,
      target_from_laser.x(), target_from_laser.y(), target_from_laser.z(),
  inc_used.yaw_deg, inc_used.pitch_deg,
      has_rough_cmd_ ? rough_yaw_deg_ : 0.0,
      has_rough_cmd_ ? rough_pitch_deg_ : 0.0,
      fine_stack_max_abs_deg_,
      stacked_fine_on_rough ? "true" : "false",
      base_yaw, base_pitch,
      has_manual_cmd_ ? manual_yaw_deg_ : 0.0,
      has_manual_cmd_ ? manual_pitch_deg_ : 0.0,
      final_yaw, final_pitch);

    if (publish_debug_) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 100,
        "increment source: %s",
        used_pixel_method ? "pixel" : "pnp");
    }
  }
}

}  // namespace laser_aim_controller