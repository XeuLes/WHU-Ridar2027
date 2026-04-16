#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "auto_aim_fast/legacy/detector.hpp"
#include "auto_aim_fast/pipeline/pipeline.hpp"
#include "auto_aim_fast/msg/aim_target.hpp"
#include "auto_aim_fast/msg/debug_stats.hpp"
#include "auto_aim_fast/vision/pairing.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"

namespace auto_aim_fast
{
class AutoAimNode final : public rclcpp::Node
{
public:
  explicit AutoAimNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~AutoAimNode() override;

private:
  void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  void onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void onRefereeIdentity(const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg);
  void onPublish();
  void onReport();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr sub_identity_;

  rclcpp::Publisher<auto_aim_fast::msg::AimTarget>::SharedPtr pub_target_;
  rclcpp::Publisher<auto_aim_fast::msg::DebugStats>::SharedPtr pub_stats_;

  // debug image publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_binary_image_;

  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr report_timer_;

  pipeline::Pipeline pipeline_;
  bool camera_set_{false};
  bool use_referee_identity_color_{true};
  std::string referee_identity_topic_{"/radar/referee/robot_identity"};
  SimpleLightDetector::Params detector_params_;
  vision::YoloDetectorParams yolo_params_;

  // Predictor state
  bool pred_inited_{false};
  int64_t last_stamp_ns_{0};
  float last_yaw_{0.0f};
  float last_pitch_{0.0f};
  float yaw_rate_{0.0f};
  float pitch_rate_{0.0f};
};
}  // namespace auto_aim_fast
