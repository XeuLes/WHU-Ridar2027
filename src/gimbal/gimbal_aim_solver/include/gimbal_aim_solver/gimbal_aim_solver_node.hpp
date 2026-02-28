#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "gimbal_aim_solver/angle_solver.hpp"
#include "gimbal_aim_solver/extrinsic_transformer.hpp"
#include "gimbal_aim_solver/gimbal_geometry.hpp"
#include "gimbal_aim_solver/types.hpp"

namespace gimbal_aim_solver
{

class GimbalAimSolverNode : public rclcpp::Node
{
public:
  explicit GimbalAimSolverNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void declareAndLoadParameters();

  void uavCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void gimbalAnglesCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  bool selectTargetWorldPoint(const geometry_msgs::msg::PoseArray::SharedPtr& msg,
                              Eigen::Vector3d& p_world) ;

  bool isFinitePoint(const Eigen::Vector3d& p) const;

  void publishTargetAngles(double yaw_deg, double pitch_deg) const;

private:
  int red_pose_index_{0};
  int blue_pose_index_{1};

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr uav_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr gimbal_angles_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_angles_pub_;

  ExtrinsicTransformer extrinsic_transformer_;
  GimbalGeometry gimbal_geometry_;
  AngleSolver angle_solver_;

  GimbalState current_gimbal_state_;

  std::string uav_topic_;
  std::string gimbal_angles_topic_;
  std::string target_angles_topic_;

  std::string target_side_;
  std::string gimbal_angle_order_;

  bool publish_debug_{true};

  // 当前角是否参与输出
  bool use_increment_output_{false};

  // 用于消息 freshness 判断
  double max_uav_msg_age_sec_{0.5};
  double max_gimbal_msg_age_sec_{0.5};

  rclcpp::Time last_gimbal_msg_time_{0, 0, RCL_ROS_TIME};
  bool has_gimbal_msg_time_{false};

  // 平滑
  bool enable_output_filter_{false};
  double output_filter_alpha_{0.2};
  mutable bool has_last_output_{false};
  mutable double last_output_yaw_deg_{0.0};
  mutable double last_output_pitch_deg_{0.0};
};

}  // namespace gimbal_aim_solver