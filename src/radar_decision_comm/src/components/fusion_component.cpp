#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/msg/enemy_world_state.hpp"
#include "radar_decision_comm/msg/tracked_robot.hpp"
#include "radar_decision_comm/msg/tracked_robot_array.hpp"

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

class FusionComponent : public rclcpp::Node
{
public:
  explicit FusionComponent(const rclcpp::NodeOptions & options)
  : Node("fusion_engine", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    fusion_rate_hz_ = this->declare_parameter<double>("fusion_rate_hz", 50.0);
    direct_obs_weight_ = this->declare_parameter<double>("direct_obs_weight", 1.0);
    wireless_obs_weight_ = this->declare_parameter<double>("wireless_obs_weight", 0.8);
    topo_rule_weight_ = this->declare_parameter<double>("topo_rule_weight", 0.5);
    stale_decay_ms_ = this->declare_parameter<int>("stale_decay_ms", 300);
    enable_kalman_slot_ = this->declare_parameter<bool>("enable_kalman_slot", false);
    enable_topo_slot_ = this->declare_parameter<bool>("enable_topo_slot", true);
    output_enemy_world_state_topic_ = this->declare_parameter<std::string>(
      "output_enemy_world_state_topic", "/radar/fusion/enemy_world_state");

    enemy_world_pub_ = this->create_publisher<radar_decision_comm::msg::EnemyWorldState>(
      output_enemy_world_state_topic_, 10);

    enemy_tracks_sub_ = this->create_subscription<radar_decision_comm::msg::TrackedRobotArray>(
      "/radar/internal/perception/enemy_tracks", 10,
      [this](const radar_decision_comm::msg::TrackedRobotArray::SharedPtr msg) { last_enemy_tracks_ = *msg; });

    wireless_enemy_sub_ = this->create_subscription<radar_decision_comm::msg::EnemyWorldState>(
      "/radar/wireless/enemy_world_state", 10,
      [this](const radar_decision_comm::msg::EnemyWorldState::SharedPtr msg) { last_wireless_enemy_state_ = *msg; });

    timer_ = this->create_wall_timer(
      hz_to_period_ms(fusion_rate_hz_),
      [this]() { publish_fused_state(); });

    RCLCPP_INFO(
      get_logger(),
      "FusionComponent started. enabled=%s fusion_rate_hz=%.1f weights(direct=%.2f wireless=%.2f topo=%.2f) kalman=%s topo=%s",
      enabled_ ? "true" : "false",
      fusion_rate_hz_,
      direct_obs_weight_,
      wireless_obs_weight_,
      topo_rule_weight_,
      enable_kalman_slot_ ? "true" : "false",
      enable_topo_slot_ ? "true" : "false");
  }

private:
  void publish_fused_state()
  {
    radar_decision_comm::msg::EnemyWorldState out;
    out.header.stamp = this->now();
    out.has_direct_observation = !last_enemy_tracks_.robots.empty();
    out.has_wireless_observation =
      !last_wireless_enemy_state_.robots.empty() || last_wireless_enemy_state_.has_wireless_observation;
    out.has_rule_prediction = false;

    std::unordered_map<uint16_t, std::size_t> robot_index;

    for (const auto & robot : last_enemy_tracks_.robots) {
      auto fused = robot;
      fused.source_mask |= 0x01U;
      fused.confidence = std::max(fused.confidence, 0.90F);
      robot_index[fused.robot_id] = out.robots.size();
      out.robots.push_back(fused);
    }

    for (const auto & robot : last_wireless_enemy_state_.robots) {
      const auto it = robot_index.find(robot.robot_id);
      if (it == robot_index.end()) {
        auto fused = robot;
        fused.source_mask |= 0x04U;
        fused.confidence = std::max(fused.confidence, 0.60F);
        robot_index[fused.robot_id] = out.robots.size();
        out.robots.push_back(fused);
      } else {
        auto & fused = out.robots[it->second];
        fused.source_mask |= 0x04U;
        fused.confidence = std::max(fused.confidence, 0.95F);
      }
    }

    enemy_world_pub_->publish(out);

    static int log_count = 0;
    if ((log_count++ % 20) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Fusion published enemy_world_state: perception=%zu wireless=%zu fused=%zu",
        last_enemy_tracks_.robots.size(),
        last_wireless_enemy_state_.robots.size(),
        out.robots.size());
    }
  }

  bool enabled_{true};
  bool enable_kalman_slot_{false};
  bool enable_topo_slot_{true};

  int stale_decay_ms_{300};

  double fusion_rate_hz_{50.0};
  double direct_obs_weight_{1.0};
  double wireless_obs_weight_{0.8};
  double topo_rule_weight_{0.5};

  std::string output_enemy_world_state_topic_;

  radar_decision_comm::msg::TrackedRobotArray last_enemy_tracks_;
  radar_decision_comm::msg::EnemyWorldState last_wireless_enemy_state_;

  rclcpp::Publisher<radar_decision_comm::msg::EnemyWorldState>::SharedPtr enemy_world_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::TrackedRobotArray>::SharedPtr enemy_tracks_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::EnemyWorldState>::SharedPtr wireless_enemy_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::FusionComponent)
