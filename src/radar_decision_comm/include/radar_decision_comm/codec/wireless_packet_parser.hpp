#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/time.hpp>

#include "radar_decision_comm/msg/enemy_world_state.hpp"
#include "radar_decision_comm/msg/wireless_enemy_ammo.hpp"
#include "radar_decision_comm/msg/wireless_enemy_buffs.hpp"
#include "radar_decision_comm/msg/wireless_enemy_hp.hpp"
#include "radar_decision_comm/msg/wireless_enemy_macro_state.hpp"
#include "radar_decision_comm/msg/wireless_interference_key.hpp"

namespace radar_decision_comm
{

struct WirelessParseResult
{
  bool has_enemy_world_state{false};
  bool has_enemy_hp{false};
  bool has_enemy_ammo{false};
  bool has_macro_state{false};
  bool has_enemy_buffs{false};
  bool has_interference_key{false};

  radar_decision_comm::msg::EnemyWorldState enemy_world_state;
  radar_decision_comm::msg::WirelessEnemyHp enemy_hp;
  radar_decision_comm::msg::WirelessEnemyAmmo enemy_ammo;
  radar_decision_comm::msg::WirelessEnemyMacroState macro_state;
  radar_decision_comm::msg::WirelessEnemyBuffs enemy_buffs;
  radar_decision_comm::msg::WirelessInterferenceKey interference_key;

  std::string warning;
};

class WirelessPacketParser
{
public:
  WirelessPacketParser() = default;
  ~WirelessPacketParser() = default;

  void set_my_color(const std::string & my_color);

  WirelessParseResult parse(
    uint16_t cmd_id,
    const std::vector<uint8_t> & payload,
    const rclcpp::Time & stamp) const;

private:
  std::string my_color_{"Red"};
};

}  // namespace radar_decision_comm
