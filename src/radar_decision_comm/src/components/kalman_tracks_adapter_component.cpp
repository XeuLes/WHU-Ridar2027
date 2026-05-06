#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/header.hpp>

#include <vision_interface/msg/detect_result.hpp>

#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
#include "radar_decision_comm/msg/tracked_robot.hpp"
#include "radar_decision_comm/msg/tracked_robot_array.hpp"

namespace radar_decision_comm
{

namespace
{

struct TeamRobotIds
{
  std::array<uint16_t, 6> ids{};
};

TeamRobotIds ids_for_team(const std::string & color)
{
  if (color == "Red" || color == "red") {
    return TeamRobotIds{{1U, 2U, 3U, 4U, 6U, 7U}};
  }
  return TeamRobotIds{{101U, 102U, 103U, 104U, 106U, 107U}};
}

bool slot_valid(float x, float y)
{
  return (x != 0.0F) || (y != 0.0F);
}

}  // namespace

class KalmanTracksAdapterComponent : public rclcpp::Node
{
public:
  explicit KalmanTracksAdapterComponent(const rclcpp::NodeOptions & options)
  : Node("kalman_tracks_adapter", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    our_color_ = this->declare_parameter<std::string>("our_color", "Red");
    fixed_confidence_ = static_cast<float>(this->declare_parameter<double>("fixed_confidence", 1.0));
    direct_source_mask_ = static_cast<uint32_t>(this->declare_parameter<int>("direct_source_mask", 0x01));
    input_detect_topic_ = this->declare_parameter<std::string>("input_detect_topic", "/kalman_detect");
    output_our_tracks_topic_ = this->declare_parameter<std::string>(
      "output_our_tracks_topic", "/radar/perception/our_tracks");
    output_enemy_tracks_topic_ = this->declare_parameter<std::string>(
      "output_enemy_tracks_topic", "/radar/perception/enemy_tracks");
    use_referee_identity_color_ = this->declare_parameter<bool>("use_referee_identity_color", true);
    identity_topic_ = this->declare_parameter<std::string>(
      "identity_topic", "/radar/referee/robot_identity");

    our_pub_ = this->create_publisher<radar_decision_comm::msg::TrackedRobotArray>(
      output_our_tracks_topic_, 10);
    enemy_pub_ = this->create_publisher<radar_decision_comm::msg::TrackedRobotArray>(
      output_enemy_tracks_topic_, 10);

    detect_sub_ = this->create_subscription<vision_interface::msg::DetectResult>(
      input_detect_topic_, 10,
      [this](const vision_interface::msg::DetectResult::SharedPtr msg)
      {
        if (!enabled_ || !msg) {
          return;
        }
        publish_tracks(*msg);
      });

    identity_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      identity_topic_, 10,
      [this](const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
      {
        if (!use_referee_identity_color_ || !msg || !msg->valid) {
          return;
        }
        our_color_ = team_identity::my_color_from_self_color(msg->self_color);
      });

    RCLCPP_INFO(
      this->get_logger(),
      "KalmanTracksAdapterComponent started. enabled=%s our_color=%s input=%s our_out=%s enemy_out=%s",
      enabled_ ? "true" : "false",
      our_color_.c_str(),
      input_detect_topic_.c_str(),
      output_our_tracks_topic_.c_str(),
      output_enemy_tracks_topic_.c_str());
  }

private:
  radar_decision_comm::msg::TrackedRobot make_robot(
    uint16_t robot_id,
    float x,
    float y,
    const std_msgs::msg::Header & header) const
  {
    radar_decision_comm::msg::TrackedRobot robot;
    robot.header = header;
    robot.robot_id = robot_id;
    robot.track_id = static_cast<uint32_t>(robot_id);
    robot.is_valid = true;
    robot.confidence = fixed_confidence_;
    robot.field_position.x = static_cast<double>(x);
    robot.field_position.y = static_cast<double>(y);
    robot.field_position.z = 0.0;
    robot.source_mask = direct_source_mask_;
    return robot;
  }

  void append_team_tracks(
    radar_decision_comm::msg::TrackedRobotArray & out,
    const TeamRobotIds & ids,
    const std::array<float, 7> & xs,
    const std::array<float, 7> & ys,
    const std_msgs::msg::Header & header) const
  {
    out.header = header;
    out.robots.clear();
    out.robots.reserve(ids.ids.size());

    for (std::size_t i = 0; i < ids.ids.size(); ++i) {
      const float x = xs[i];
      const float y = ys[i];
      if (!slot_valid(x, y)) {
        continue;
      }
      out.robots.push_back(make_robot(ids.ids[i], x, y, header));
    }
  }

  void publish_tracks(const vision_interface::msg::DetectResult & msg)
  {
    const TeamRobotIds red_ids = ids_for_team("Red");
    const TeamRobotIds blue_ids = ids_for_team("Blue");

    const std::array<float, 7> blue_xs = msg.blue_x;
    const std::array<float, 7> blue_ys = msg.blue_y;
    const std::array<float, 7> red_xs = msg.red_x;
    const std::array<float, 7> red_ys = msg.red_y;

    radar_decision_comm::msg::TrackedRobotArray our_tracks;
    radar_decision_comm::msg::TrackedRobotArray enemy_tracks;

    const bool our_is_red = (our_color_ == "Red" || our_color_ == "red");
    if (our_is_red) {
      append_team_tracks(our_tracks, red_ids, red_xs, red_ys, msg.header);
      append_team_tracks(enemy_tracks, blue_ids, blue_xs, blue_ys, msg.header);
    } else {
      append_team_tracks(our_tracks, blue_ids, blue_xs, blue_ys, msg.header);
      append_team_tracks(enemy_tracks, red_ids, red_xs, red_ys, msg.header);
    }

    our_pub_->publish(our_tracks);
    enemy_pub_->publish(enemy_tracks);
  }

  bool enabled_{true};
  std::string our_color_{"Red"};
  bool use_referee_identity_color_{true};
  float fixed_confidence_{1.0F};
  uint32_t direct_source_mask_{0x01U};
  std::string input_detect_topic_{"/kalman_detect"};
  std::string output_our_tracks_topic_{"/radar/perception/our_tracks"};
  std::string output_enemy_tracks_topic_{"/radar/perception/enemy_tracks"};
  std::string identity_topic_{"/radar/referee/robot_identity"};

  rclcpp::Publisher<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr our_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr enemy_pub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr detect_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr identity_sub_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::KalmanTracksAdapterComponent)
