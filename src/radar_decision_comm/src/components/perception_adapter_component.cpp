#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/msg/tracked_robot_array.hpp"

namespace radar_decision_comm
{

class PerceptionAdapterComponent : public rclcpp::Node
{
public:
  explicit PerceptionAdapterComponent(const rclcpp::NodeOptions & options)
  : Node("perception_adapter", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    input_enemy_tracks_topic_ = this->declare_parameter<std::string>(
      "input_enemy_tracks_topic", "/radar/perception/enemy_tracks");
    input_our_tracks_topic_ = this->declare_parameter<std::string>(
      "input_our_tracks_topic", "/radar/perception/our_tracks");
    output_enemy_tracks_topic_ = this->declare_parameter<std::string>(
      "output_enemy_tracks_topic", "/radar/internal/perception/enemy_tracks");
    output_our_tracks_topic_ = this->declare_parameter<std::string>(
      "output_our_tracks_topic", "/radar/internal/perception/our_tracks");

    enemy_pub_ = this->create_publisher<radar_decision_comm::msg::TrackedRobotArray>(
      output_enemy_tracks_topic_, 10);
    our_pub_ = this->create_publisher<radar_decision_comm::msg::TrackedRobotArray>(
      output_our_tracks_topic_, 10);

    enemy_sub_ = this->create_subscription<radar_decision_comm::msg::TrackedRobotArray>(
      input_enemy_tracks_topic_, 10,
      [this](const radar_decision_comm::msg::TrackedRobotArray::SharedPtr msg)
      {
        enemy_pub_->publish(*msg);
      });

    our_sub_ = this->create_subscription<radar_decision_comm::msg::TrackedRobotArray>(
      input_our_tracks_topic_, 10,
      [this](const radar_decision_comm::msg::TrackedRobotArray::SharedPtr msg)
      {
        our_pub_->publish(*msg);
      });

    RCLCPP_INFO(
      get_logger(),
      "PerceptionAdapterComponent started. enabled=%s enemy_in=%s enemy_out=%s our_in=%s our_out=%s",
      enabled_ ? "true" : "false",
      input_enemy_tracks_topic_.c_str(),
      output_enemy_tracks_topic_.c_str(),
      input_our_tracks_topic_.c_str(),
      output_our_tracks_topic_.c_str());
  }

private:
  bool enabled_{true};
  std::string input_enemy_tracks_topic_;
  std::string input_our_tracks_topic_;
  std::string output_enemy_tracks_topic_;
  std::string output_our_tracks_topic_;

  rclcpp::Publisher<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr enemy_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr our_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr enemy_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr our_sub_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::PerceptionAdapterComponent)
