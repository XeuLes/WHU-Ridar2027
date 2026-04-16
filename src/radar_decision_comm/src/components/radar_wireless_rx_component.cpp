#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/string.hpp>

#include "radar_decision_comm/codec/wireless_packet_parser.hpp"
#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/msg/enemy_world_state.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
#include "radar_decision_comm/msg/wireless_enemy_ammo.hpp"
#include "radar_decision_comm/msg/wireless_enemy_buffs.hpp"
#include "radar_decision_comm/msg/wireless_enemy_hp.hpp"
#include "radar_decision_comm/msg/wireless_enemy_macro_state.hpp"
#include "radar_decision_comm/msg/wireless_interference_key.hpp"

namespace radar_decision_comm
{

class RadarWirelessRxComponent : public rclcpp::Node
{
public:
  explicit RadarWirelessRxComponent(const rclcpp::NodeOptions & options)
  : Node("radar_wireless_rx", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    input_pdu_topic_ = this->declare_parameter<std::string>("input_pdu_topic", "/radar/all_pdu");
    input_queue_depth_ = this->declare_parameter<int>("input_queue_depth", 50);
    use_referee_identity_color_ = this->declare_parameter<bool>("use_referee_identity_color", true);
    identity_topic_ = this->declare_parameter<std::string>(
      "identity_topic", "/radar/referee/robot_identity");

    parser_.set_my_color(my_color_);

    enemy_world_pub_ = this->create_publisher<radar_decision_comm::msg::EnemyWorldState>(
      "/radar/wireless/enemy_world_state", 10);
    enemy_hp_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyHp>(
      "/radar/wireless/enemy_hp", 10);
    enemy_ammo_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyAmmo>(
      "/radar/wireless/enemy_ammo", 10);
    macro_state_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyMacroState>(
      "/radar/wireless/enemy_macro_state", 10);
    enemy_buffs_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyBuffs>(
      "/radar/wireless/enemy_buffs", 10);
    interference_key_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessInterferenceKey>(
      "/radar/wireless/interference_key", 10);

    pdu_sub_ = this->create_subscription<std_msgs::msg::String>(
      input_pdu_topic_, input_queue_depth_,
      [this](const std_msgs::msg::String::SharedPtr msg)
      {
        if (!enabled_) {
          return;
        }
        handle_pdu_message(msg->data);
      });

    identity_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      identity_topic_, 10,
      [this](const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
      {
        if (!use_referee_identity_color_ || !msg || !msg->valid) {
          return;
        }
        const std::string next_color = team_identity::my_color_from_self_color(msg->self_color);
        if (next_color != my_color_) {
          my_color_ = next_color;
          parser_.set_my_color(my_color_);
          RCLCPP_INFO(
            this->get_logger(),
            "RadarWirelessRxComponent switched my_color=%s from referee robot_id=%u",
            my_color_.c_str(),
            msg->robot_id);
        }
      });

    RCLCPP_INFO(
      get_logger(),
      "RadarWirelessRxComponent started. enabled=%s color=%s input_pdu_topic=%s queue_depth=%d",
      enabled_ ? "true" : "false",
      my_color_.c_str(),
      input_pdu_topic_.c_str(),
      input_queue_depth_);
  }

private:
  void publish_parsed(const WirelessParseResult & result)
  {
    if (result.has_enemy_world_state) {
      enemy_world_pub_->publish(result.enemy_world_state);
    }
    if (result.has_enemy_hp) {
      enemy_hp_pub_->publish(result.enemy_hp);
    }
    if (result.has_enemy_ammo) {
      enemy_ammo_pub_->publish(result.enemy_ammo);
    }
    if (result.has_macro_state) {
      macro_state_pub_->publish(result.macro_state);
    }
    if (result.has_enemy_buffs) {
      enemy_buffs_pub_->publish(result.enemy_buffs);
    }
    if (result.has_interference_key) {
      interference_key_pub_->publish(result.interference_key);
    }
  }

  void handle_pdu_message(const std::string & json_text)
  {
    uint16_t cmd_id = 0;
    std::string source_type;
    const auto result = parser_.parse_bridge_json(json_text, this->now(), &cmd_id, &source_type);

    if (!result.warning.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Wireless bridge parse warning for source=%s cmd=0x%04x: %s",
        source_type.c_str(),
        cmd_id,
        result.warning.c_str());
    }

    if (
      result.has_enemy_world_state || result.has_enemy_hp || result.has_enemy_ammo ||
      result.has_macro_state || result.has_enemy_buffs || result.has_interference_key)
    {
      publish_parsed(result);

      static int log_count = 0;
      if ((log_count++ % 10) == 0) {
        RCLCPP_INFO(
          this->get_logger(),
          "Wireless PDU decoded from bridge: source=%s cmd=0x%04x",
          source_type.c_str(),
          cmd_id);
      }
    } else if (result.warning.empty()) {
      RCLCPP_INFO(
        this->get_logger(),
        "Wireless bridge PDU ignored: source=%s cmd=0x%04x",
        source_type.c_str(),
        cmd_id);
    }
  }

  bool enabled_{true};
  std::string my_color_{"Red"};
  bool use_referee_identity_color_{true};
  std::string input_pdu_topic_{"/radar/all_pdu"};
  std::string identity_topic_{"/radar/referee/robot_identity"};
  int input_queue_depth_{50};

  WirelessPacketParser parser_;

  rclcpp::Publisher<radar_decision_comm::msg::EnemyWorldState>::SharedPtr enemy_world_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyHp>::SharedPtr enemy_hp_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyAmmo>::SharedPtr enemy_ammo_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyMacroState>::SharedPtr macro_state_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyBuffs>::SharedPtr enemy_buffs_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessInterferenceKey>::SharedPtr interference_key_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pdu_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr identity_sub_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::RadarWirelessRxComponent)
