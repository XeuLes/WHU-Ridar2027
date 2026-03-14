#pragma once

#include <string>

#include <Eigen/Core>

#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "auto_aim_fast/msg/aim_target.hpp"
#include "laser_aim_controller/angle_solver.hpp"
#include "laser_aim_controller/laser_geometry.hpp"
#include "laser_aim_controller/types.hpp"

namespace laser_aim_controller
{

class LaserAimControllerNode : public rclcpp::Node
{
public:
  explicit LaserAimControllerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void declareAndLoadParameters();

  void targetCallback(const auto_aim_fast::msg::AimTarget::SharedPtr msg);
  void gimbalAnglesCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void quaternionCallback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg);
  void roughCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  void publishCommand(double yaw_deg, double pitch_deg);
  void publishRoughIfAvailable();
  bool isFinite3(double x, double y, double z) const;

private:

  // 新增函数声明
  void manualCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void applyManualOffset(double& yaw_deg, double& pitch_deg) const;

  // 新增订阅器
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manual_cmd_sub_;

  // 新增topic
  std::string manual_command_topic_;

  // 新增状态
  double manual_yaw_deg_{0.0};
  double manual_pitch_deg_{0.0};
  bool has_manual_cmd_{false};

  rclcpp::Subscription<auto_aim_fast::msg::AimTarget>::SharedPtr target_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr gimbal_angles_sub_;
  rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr quaternion_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rough_cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;

  std::string target_topic_;
  std::string gimbal_angles_topic_;
  std::string gimbal_quaternion_topic_;
  std::string rough_command_topic_;
  std::string command_topic_;

  std::string gimbal_angle_order_;
  std::string output_mode_;

  bool publish_debug_{true};
  bool enable_output_filter_{false};
  bool fallback_to_rough_when_no_target_{true};

  double output_filter_alpha_{0.3};
  double min_valid_z_m_{0.05};

  LaserGeometry laser_geometry_;
  AngleSolver angle_solver_;
  GimbalState current_gimbal_;

  bool has_last_output_{false};
  double last_yaw_out_deg_{0.0};
  double last_pitch_out_deg_{0.0};

  bool has_rough_cmd_{false};
  double rough_yaw_deg_{0.0};
  double rough_pitch_deg_{0.0};

  double fine_stack_max_abs_deg_{0.5};
};

}  // namespace laser_aim_controller