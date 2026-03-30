#pragma once

#include <cstdint>
#include <string>

#include "radar_decision_comm/msg/battle_blackboard.hpp"
#include "radar_decision_comm/msg/tracked_robot.hpp"

namespace radar_decision_comm
{

struct TargetSelectorConfig
{
  std::string my_color{"Red"};
  int enemy_hero_id{101};
  int low_hp_alert_threshold{120};
  int low_ammo_threshold{10};
  double buff_bonus_score{8.0};
  double macro_bonus_score{3.0};
};

struct TargetSelectorResult
{
  bool has_target{false};
  radar_decision_comm::msg::TrackedRobot primary_target;
  double threat_score{0.0};
  std::string reason;
};

class TargetSelector
{
public:
  static TargetSelectorResult select(
    const radar_decision_comm::msg::BattleBlackboard & bb,
    const TargetSelectorConfig & cfg);

private:
  static int enemy_slot_from_robot_id(const std::string & my_color, uint16_t robot_id);
  static uint16_t wireless_hp_for_slot(
    const radar_decision_comm::msg::BattleBlackboard & bb,
    int slot);
  static uint16_t wireless_ammo_for_slot(
    const radar_decision_comm::msg::BattleBlackboard & bb,
    int slot);
  static double base_threat_for_slot(int slot);
};

}  // namespace radar_decision_comm
