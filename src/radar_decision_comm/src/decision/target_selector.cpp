#include "radar_decision_comm/decision/target_selector.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace radar_decision_comm
{

int TargetSelector::enemy_slot_from_robot_id(const std::string & my_color, uint16_t robot_id)
{
  if (my_color == "Red") {
    switch (robot_id) {
      case 101: return 0;
      case 102: return 1;
      case 103: return 2;
      case 104: return 3;
      case 105: return 4;
      case 107: return 5;
      default: return -1;
    }
  }

  switch (robot_id) {
    case 1: return 0;
    case 2: return 1;
    case 3: return 2;
    case 4: return 3;
    case 5: return 4;
    case 7: return 5;
    default: return -1;
  }
}

uint16_t TargetSelector::wireless_hp_for_slot(
  const radar_decision_comm::msg::BattleBlackboard & bb,
  int slot)
{
  switch (slot) {
    case 0: return bb.wireless_enemy_hp.hero;
    case 1: return bb.wireless_enemy_hp.engineer;
    case 2: return bb.wireless_enemy_hp.infantry_3;
    case 3: return bb.wireless_enemy_hp.infantry_4;
    case 4: return bb.wireless_enemy_hp.infantry_5;
    case 5: return bb.wireless_enemy_hp.sentinel;
    default: return 0;
  }
}

uint16_t TargetSelector::wireless_ammo_for_slot(
  const radar_decision_comm::msg::BattleBlackboard & bb,
  int slot)
{
  switch (slot) {
    case 0: return bb.wireless_enemy_ammo.hero;
    case 1: return 0;
    case 2: return bb.wireless_enemy_ammo.infantry_3;
    case 3: return bb.wireless_enemy_ammo.infantry_4;
    case 4: return 0;
    case 5: return bb.wireless_enemy_ammo.sentinel;
    default: return 0;
  }
}

double TargetSelector::base_threat_for_slot(int slot)
{
  switch (slot) {
    case 0: return 100.0;  // hero
    case 5: return 80.0;   // sentinel
    case 2: return 65.0;
    case 3: return 60.0;
    case 4: return 55.0;
    case 1: return 45.0;   // engineer
    default: return 30.0;
  }
}

TargetSelectorResult TargetSelector::select(
  const radar_decision_comm::msg::BattleBlackboard & bb,
  const TargetSelectorConfig & cfg)
{
  TargetSelectorResult out;
  if (bb.enemy_state.robots.empty()) {
    out.reason = "no_enemy";
    return out;
  }

  const bool wireless_valid = !bb.wireless_stale;
  const bool referee_valid = !bb.referee_stale;
  const bool buffs_valid = wireless_valid && !bb.wireless_enemy_buffs.raw_bytes.empty();
  const bool macro_valid = wireless_valid && !bb.wireless_macro_state.raw_bytes.empty();

  for (const auto & robot : bb.enemy_state.robots) {
    const int slot = enemy_slot_from_robot_id(cfg.my_color, robot.robot_id);
    double score = base_threat_for_slot(slot);

    if (wireless_valid && slot >= 0) {
      const auto hp = wireless_hp_for_slot(bb, slot);
      if (hp > 0) {
        score += (300.0 - std::min<double>(hp, 300.0)) * 0.20;
      }

      const auto ammo = wireless_ammo_for_slot(bb, slot);
      if (ammo > 0 && ammo <= static_cast<uint16_t>(cfg.low_ammo_threshold)) {
        score += 6.0;
      }
    }

    if (referee_valid && slot == 0 && bb.mark_progress.hero >= 100) {
      score += 10.0;
    }

    if (buffs_valid) {
      score += cfg.buff_bonus_score;
    }

    if (macro_valid) {
      score += cfg.macro_bonus_score;
    }

    if (robot.source_mask & 0x01U) {
      score += 4.0;
    }
    if (robot.source_mask & 0x04U) {
      score += 2.0;
    }

    if (!out.has_target || score > out.threat_score) {
      out.has_target = true;
      out.primary_target = robot;
      out.threat_score = score;

      std::ostringstream oss;
      oss << "slot=" << slot
          << "|score=" << score
          << "|src=" << robot.source_mask;

      if (wireless_valid && slot >= 0) {
        oss << "|hp=" << wireless_hp_for_slot(bb, slot)
            << "|ammo=" << wireless_ammo_for_slot(bb, slot);
      }

      if (buffs_valid) {
        oss << "|buff=1";
      }
      if (macro_valid) {
        oss << "|macro=1";
      }

      out.reason = oss.str();
    }
  }

  return out;
}

}  // namespace radar_decision_comm
