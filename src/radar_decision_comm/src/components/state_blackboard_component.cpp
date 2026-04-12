#include <chrono>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/msg/battle_blackboard.hpp"
#include "radar_decision_comm/msg/dart_status.hpp"
#include "radar_decision_comm/msg/enemy_world_state.hpp"
#include "radar_decision_comm/msg/radar_decision_sync.hpp"
#include "radar_decision_comm/msg/radar_mark_progress.hpp"
#include "radar_decision_comm/msg/referee_enemy_hp.hpp"
#include "radar_decision_comm/msg/referee_game_state.hpp"
#include "radar_decision_comm/msg/tracked_robot_array.hpp"
#include "radar_decision_comm/msg/wireless_enemy_ammo.hpp"
#include "radar_decision_comm/msg/wireless_enemy_buffs.hpp"
#include "radar_decision_comm/msg/wireless_enemy_hp.hpp"
#include "radar_decision_comm/msg/wireless_enemy_macro_state.hpp"
#include "radar_decision_comm/msg/wireless_interference_key.hpp"
#include "radar_decision_comm/srv/reset_state.hpp"

namespace radar_decision_comm
{

static std::chrono::milliseconds hz_to_period_ms(double hz)
{
  if (hz <= 0.0) {
    return std::chrono::milliseconds(1000);
  }
  auto ms = static_cast<int>(1000.0 / hz);
  if (ms < 1) {
    ms = 1;
  }
  return std::chrono::milliseconds(ms);
}

class StateBlackboardComponent : public rclcpp::Node
{
public:
  explicit StateBlackboardComponent(const rclcpp::NodeOptions & options)
  : Node("state_blackboard", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    state_ttl_ms_ = this->declare_parameter<int>("state_ttl_ms", 500);
    enemy_track_ttl_ms_ = this->declare_parameter<int>("enemy_track_ttl_ms", 300);
    wireless_track_ttl_ms_ = this->declare_parameter<int>("wireless_track_ttl_ms", 500);
    our_track_ttl_ms_ = this->declare_parameter<int>("our_track_ttl_ms", 500);
    referee_state_ttl_ms_ = this->declare_parameter<int>("referee_state_ttl_ms", 1500);
    publish_blackboard_hz_ = this->declare_parameter<double>("publish_blackboard_hz", 30.0);

    blackboard_pub_ = this->create_publisher<radar_decision_comm::msg::BattleBlackboard>(
      "/radar/fusion/battle_blackboard", 10);

    game_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeGameState>(
      "/radar/referee/game_status", 10,
      [this](const radar_decision_comm::msg::RefereeGameState::SharedPtr msg)
      {
        last_game_state_ = *msg;
        last_referee_update_ = this->now();
      });

    hp_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeEnemyHp>(
      "/radar/referee/enemy_hp", 10,
      [this](const radar_decision_comm::msg::RefereeEnemyHp::SharedPtr msg)
      {
        last_enemy_hp_ = *msg;
        last_referee_update_ = this->now();
      });

    dart_sub_ = this->create_subscription<radar_decision_comm::msg::DartStatus>(
      "/radar/referee/dart_status", 10,
      [this](const radar_decision_comm::msg::DartStatus::SharedPtr msg)
      {
        last_dart_status_ = *msg;
        last_referee_update_ = this->now();
      });

    mark_sub_ = this->create_subscription<radar_decision_comm::msg::RadarMarkProgress>(
      "/radar/referee/mark_progress", 10,
      [this](const radar_decision_comm::msg::RadarMarkProgress::SharedPtr msg)
      {
        last_mark_progress_ = *msg;
        last_referee_update_ = this->now();
      });

    sync_sub_ = this->create_subscription<radar_decision_comm::msg::RadarDecisionSync>(
      "/radar/referee/radar_decision_sync", 10,
      [this](const radar_decision_comm::msg::RadarDecisionSync::SharedPtr msg)
      {
        last_radar_sync_ = *msg;
        last_referee_update_ = this->now();
      });

    perception_sub_ = this->create_subscription<radar_decision_comm::msg::TrackedRobotArray>(
      "/radar/internal/perception/enemy_tracks", 10,
      [this](const radar_decision_comm::msg::TrackedRobotArray::SharedPtr msg)
      {
        (void)msg;
        last_perception_update_ = this->now();
      });

    enemy_state_sub_ = this->create_subscription<radar_decision_comm::msg::EnemyWorldState>(
      "/radar/fusion/enemy_world_state", 10,
      [this](const radar_decision_comm::msg::EnemyWorldState::SharedPtr msg)
      {
        last_enemy_state_ = *msg;
      });

    our_state_sub_ = this->create_subscription<radar_decision_comm::msg::TrackedRobotArray>(
      "/radar/internal/perception/our_tracks", 10,
      [this](const radar_decision_comm::msg::TrackedRobotArray::SharedPtr msg)
      {
        last_our_state_ = *msg;
        last_our_update_ = this->now();
      });

    wireless_hp_sub_ = this->create_subscription<radar_decision_comm::msg::WirelessEnemyHp>(
      "/radar/wireless/enemy_hp", 10,
      [this](const radar_decision_comm::msg::WirelessEnemyHp::SharedPtr msg)
      {
        last_wireless_enemy_hp_ = *msg;
        last_wireless_update_ = this->now();
      });

    wireless_ammo_sub_ = this->create_subscription<radar_decision_comm::msg::WirelessEnemyAmmo>(
      "/radar/wireless/enemy_ammo", 10,
      [this](const radar_decision_comm::msg::WirelessEnemyAmmo::SharedPtr msg)
      {
        last_wireless_enemy_ammo_ = *msg;
        last_wireless_update_ = this->now();
      });

    wireless_macro_sub_ = this->create_subscription<radar_decision_comm::msg::WirelessEnemyMacroState>(
      "/radar/wireless/enemy_macro_state", 10,
      [this](const radar_decision_comm::msg::WirelessEnemyMacroState::SharedPtr msg)
      {
        last_wireless_macro_state_ = *msg;
        last_wireless_update_ = this->now();
      });

    wireless_buffs_sub_ = this->create_subscription<radar_decision_comm::msg::WirelessEnemyBuffs>(
      "/radar/wireless/enemy_buffs", 10,
      [this](const radar_decision_comm::msg::WirelessEnemyBuffs::SharedPtr msg)
      {
        last_wireless_enemy_buffs_ = *msg;
        last_wireless_update_ = this->now();
      });

    wireless_key_sub_ = this->create_subscription<radar_decision_comm::msg::WirelessInterferenceKey>(
      "/radar/wireless/interference_key", 10,
      [this](const radar_decision_comm::msg::WirelessInterferenceKey::SharedPtr msg)
      {
        last_wireless_interference_key_ = *msg;
        last_wireless_update_ = this->now();
      });

    reset_state_srv_ = this->create_service<radar_decision_comm::srv::ResetState>(
      "~/reset_state",
      [this](
        const std::shared_ptr<radar_decision_comm::srv::ResetState::Request> request,
        std::shared_ptr<radar_decision_comm::srv::ResetState::Response> response)
      {
        (void)request;
        last_game_state_ = radar_decision_comm::msg::RefereeGameState();
        last_enemy_hp_ = radar_decision_comm::msg::RefereeEnemyHp();
        last_dart_status_ = radar_decision_comm::msg::DartStatus();
        last_mark_progress_ = radar_decision_comm::msg::RadarMarkProgress();
        last_radar_sync_ = radar_decision_comm::msg::RadarDecisionSync();
        last_enemy_state_ = radar_decision_comm::msg::EnemyWorldState();
        last_our_state_ = radar_decision_comm::msg::TrackedRobotArray();
        last_wireless_enemy_hp_ = radar_decision_comm::msg::WirelessEnemyHp();
        last_wireless_enemy_ammo_ = radar_decision_comm::msg::WirelessEnemyAmmo();
        last_wireless_macro_state_ = radar_decision_comm::msg::WirelessEnemyMacroState();
        last_wireless_enemy_buffs_ = radar_decision_comm::msg::WirelessEnemyBuffs();
        last_wireless_interference_key_ = radar_decision_comm::msg::WirelessInterferenceKey();
        last_referee_update_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_wireless_update_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_perception_update_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_our_update_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        response->success = true;
        response->message = "blackboard state reset";
      });

    timer_ = this->create_wall_timer(
      hz_to_period_ms(publish_blackboard_hz_),
      [this]()
      {
        radar_decision_comm::msg::BattleBlackboard msg;
        msg.header.stamp = this->now();
        msg.game_state = last_game_state_;
        msg.enemy_hp = last_enemy_hp_;
        msg.mark_progress = last_mark_progress_;
        msg.radar_sync = last_radar_sync_;
        msg.dart_status = last_dart_status_;
        msg.enemy_state = last_enemy_state_;
        msg.our_state = last_our_state_;
        msg.wireless_enemy_hp = last_wireless_enemy_hp_;
        msg.wireless_enemy_ammo = last_wireless_enemy_ammo_;
        msg.wireless_macro_state = last_wireless_macro_state_;
        msg.wireless_enemy_buffs = last_wireless_enemy_buffs_;
        msg.wireless_interference_key = last_wireless_interference_key_;

        msg.referee_last_update = last_referee_update_;
        msg.wireless_last_update = last_wireless_update_;
        msg.perception_last_update = last_perception_update_;
        msg.our_last_update = last_our_update_;

        msg.referee_stale = is_stale(last_referee_update_, referee_state_ttl_ms_);
        msg.wireless_stale = is_stale(last_wireless_update_, wireless_track_ttl_ms_);
        msg.perception_stale = is_stale(last_perception_update_, enemy_track_ttl_ms_);
        msg.our_stale = is_stale(last_our_update_, our_track_ttl_ms_);

        blackboard_pub_->publish(msg);
      });

    RCLCPP_INFO(
      get_logger(),
      "StateBlackboardComponent started. enabled=%s publish_blackboard_hz=%.1f",
      enabled_ ? "true" : "false",
      publish_blackboard_hz_);
  }

private:
  bool is_stale(const rclcpp::Time & last, int ttl_ms) const
  {
    if (last.nanoseconds() == 0) {
      return true;
    }
    const auto age_ns = (this->now() - last).nanoseconds();
    return age_ns > static_cast<int64_t>(ttl_ms) * 1000000LL;
  }

  bool enabled_{true};
  int state_ttl_ms_{500};
  int enemy_track_ttl_ms_{300};
  int wireless_track_ttl_ms_{500};
  int our_track_ttl_ms_{500};
  int referee_state_ttl_ms_{1500};
  double publish_blackboard_hz_{30.0};

  radar_decision_comm::msg::RefereeGameState last_game_state_;
  radar_decision_comm::msg::RefereeEnemyHp last_enemy_hp_;
  radar_decision_comm::msg::DartStatus last_dart_status_;
  radar_decision_comm::msg::RadarMarkProgress last_mark_progress_;
  radar_decision_comm::msg::RadarDecisionSync last_radar_sync_;
  radar_decision_comm::msg::EnemyWorldState last_enemy_state_;
  radar_decision_comm::msg::TrackedRobotArray last_our_state_;
  radar_decision_comm::msg::WirelessEnemyHp last_wireless_enemy_hp_;
  radar_decision_comm::msg::WirelessEnemyAmmo last_wireless_enemy_ammo_;
  radar_decision_comm::msg::WirelessEnemyMacroState last_wireless_macro_state_;
  radar_decision_comm::msg::WirelessEnemyBuffs last_wireless_enemy_buffs_;
  radar_decision_comm::msg::WirelessInterferenceKey last_wireless_interference_key_;

  rclcpp::Time last_referee_update_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_wireless_update_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_perception_update_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_our_update_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<radar_decision_comm::msg::BattleBlackboard>::SharedPtr blackboard_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeGameState>::SharedPtr game_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeEnemyHp>::SharedPtr hp_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::DartStatus>::SharedPtr dart_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RadarMarkProgress>::SharedPtr mark_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RadarDecisionSync>::SharedPtr sync_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr perception_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::EnemyWorldState>::SharedPtr enemy_state_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr our_state_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::WirelessEnemyHp>::SharedPtr wireless_hp_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::WirelessEnemyAmmo>::SharedPtr wireless_ammo_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::WirelessEnemyMacroState>::SharedPtr wireless_macro_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::WirelessEnemyBuffs>::SharedPtr wireless_buffs_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::WirelessInterferenceKey>::SharedPtr wireless_key_sub_;
  rclcpp::Service<radar_decision_comm::srv::ResetState>::SharedPtr reset_state_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::StateBlackboardComponent)
