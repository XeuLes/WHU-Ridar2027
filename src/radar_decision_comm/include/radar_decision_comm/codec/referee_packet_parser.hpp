#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/time.hpp>

#include "radar_decision_comm/msg/dart_status.hpp"
#include "radar_decision_comm/msg/radar_decision_sync.hpp"
#include "radar_decision_comm/msg/radar_mark_progress.hpp"
#include "radar_decision_comm/msg/referee_enemy_hp.hpp"
#include "radar_decision_comm/msg/referee_game_state.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"

namespace radar_decision_comm
{

struct RefereeParseResult
{
  bool has_game_state{false};
  bool has_enemy_hp{false};
  bool has_dart_status{false};
  bool has_mark_progress{false};
  bool has_radar_sync{false};
  bool has_robot_identity{false};

  radar_decision_comm::msg::RefereeGameState game_state;
  radar_decision_comm::msg::RefereeEnemyHp enemy_hp;
  radar_decision_comm::msg::DartStatus dart_status;
  radar_decision_comm::msg::RadarMarkProgress mark_progress;
  radar_decision_comm::msg::RadarDecisionSync radar_sync;
  radar_decision_comm::msg::RefereeRobotIdentity robot_identity;

  std::string warning;
};

class RefereePacketParser
{
public:
  RefereePacketParser() = default;
  ~RefereePacketParser() = default;

  void set_my_color(const std::string & my_color);
  void set_legacy_mark_progress(bool enabled);
  void set_legacy_dart_bits(bool enabled);

  RefereeParseResult parse(
    uint16_t cmd_id,
    const std::vector<uint8_t> & payload,
    const rclcpp::Time & stamp) const;

private:
  std::string my_color_{"Red"};
  bool legacy_mark_progress_{false};
  bool legacy_dart_bits_{false};
};

}  // namespace radar_decision_comm
