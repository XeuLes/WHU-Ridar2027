#include "radar_decision_comm/codec/referee_packet_parser.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "radar_decision_comm/common/team_identity.hpp"

namespace radar_decision_comm
{

namespace
{

uint16_t le_u16(const std::vector<uint8_t> & data, std::size_t offset)
{
  if (offset + 1 >= data.size()) {
    return 0;
  }
  return static_cast<uint16_t>(data[offset]) |
         (static_cast<uint16_t>(data[offset + 1]) << 8U);
}

uint64_t le_u64(const std::vector<uint8_t> & data, std::size_t offset)
{
  if (offset + 7 >= data.size()) {
    return 0;
  }

  uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= (static_cast<uint64_t>(data[offset + i]) << (8U * i));
  }
  return value;
}

uint8_t bit_flag(uint16_t value, unsigned bit)
{
  return static_cast<uint8_t>(((value >> bit) & 0x01U) != 0U ? 1U : 0U);
}

}  // namespace

void RefereePacketParser::set_my_color(const std::string & my_color)
{
  my_color_ = my_color;
}

void RefereePacketParser::set_legacy_mark_progress(bool enabled)
{
  legacy_mark_progress_ = enabled;
}

void RefereePacketParser::set_legacy_dart_bits(bool enabled)
{
  legacy_dart_bits_ = enabled;
}

RefereeParseResult RefereePacketParser::parse(
  uint16_t cmd_id,
  const std::vector<uint8_t> & payload,
  const rclcpp::Time & stamp) const
{
  RefereeParseResult out;

  switch (cmd_id) {
    case 0x0001: {
      if (payload.size() != 11) {
        out.warning = "0x0001 payload length mismatch";
        return out;
      }

      out.has_game_state = true;
      out.game_state.header.stamp = stamp;

      const uint8_t first = payload[0];
      out.game_state.game_type = static_cast<uint8_t>(first & 0x0F);
      out.game_state.game_progress = static_cast<uint8_t>((first >> 4U) & 0x0F);
      out.game_state.stage_remain_time = le_u16(payload, 1);

      const uint64_t sync_unix = le_u64(payload, 3);
      const uint64_t max_i32 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
      out.game_state.sync_time.sec = static_cast<int32_t>(std::min(sync_unix, max_i32));
      out.game_state.sync_time.nanosec = 0;
      break;
    }

    case 0x0003: {
      if (payload.size() != 16) {
        out.warning = "0x0003 payload length mismatch";
        return out;
      }

      out.has_enemy_hp = true;
      out.enemy_hp.header.stamp = stamp;
      out.enemy_hp.hero = le_u16(payload, 0);
      out.enemy_hp.engineer = le_u16(payload, 2);
      out.enemy_hp.infantry_3 = le_u16(payload, 4);
      out.enemy_hp.infantry_4 = le_u16(payload, 6);
      out.enemy_hp.reserved = le_u16(payload, 8);
      out.enemy_hp.sentinel = le_u16(payload, 10);
      out.enemy_hp.outpost = le_u16(payload, 12);
      out.enemy_hp.base = le_u16(payload, 14);
      break;
    }

    case 0x0105: {
      if (payload.size() != 3) {
        out.warning = "0x0105 payload length mismatch";
        return out;
      }

      out.has_dart_status = true;
      out.dart_status.header.stamp = stamp;
      out.dart_status.dart_remaining_time = payload[0];

      const uint16_t dart_info = le_u16(payload, 1);
      if (legacy_dart_bits_) {
        out.dart_status.dart_hit_target = static_cast<uint8_t>(dart_info & 0x03U);
        out.dart_status.dart_hit_count = static_cast<uint8_t>((dart_info >> 2U) & 0x07U);
        out.dart_status.selected_target = static_cast<uint8_t>((dart_info >> 5U) & 0x03U);
      } else {
        out.dart_status.dart_hit_target = static_cast<uint8_t>(dart_info & 0x07U);
        out.dart_status.dart_hit_count = static_cast<uint8_t>((dart_info >> 3U) & 0x07U);
        out.dart_status.selected_target = static_cast<uint8_t>((dart_info >> 6U) & 0x07U);
      }
      break;
    }

    case 0x0201: {
      if (payload.size() != 13) {
        out.warning = "0x0201 payload length mismatch";
        return out;
      }

      const uint16_t robot_id = static_cast<uint16_t>(payload[0]);
      const int self_color = team_identity::self_color_code_from_robot_id(robot_id);
      if (self_color < 0) {
        out.warning = "0x0201 robot_id invalid for color derivation";
        return out;
      }

      out.has_robot_identity = true;
      out.robot_identity.header.stamp = stamp;
      out.robot_identity.valid = true;
      out.robot_identity.robot_id = robot_id;
      out.robot_identity.self_color = static_cast<int8_t>(self_color);
      break;
    }

    case 0x020C: {
      if (payload.size() == 2) {
        const uint16_t mark_bits = le_u16(payload, 0);
        out.has_mark_progress = true;
        out.mark_progress.header.stamp = stamp;
        out.mark_progress.hero = bit_flag(mark_bits, 0);
        out.mark_progress.engineer = bit_flag(mark_bits, 1);
        out.mark_progress.infantry_3 = bit_flag(mark_bits, 2);
        out.mark_progress.infantry_4 = bit_flag(mark_bits, 3);
        out.mark_progress.aerial_6 = bit_flag(mark_bits, 4);
        out.mark_progress.sentinel = bit_flag(mark_bits, 5);
        out.mark_progress.ally_hero = bit_flag(mark_bits, 6);
        out.mark_progress.ally_engineer = bit_flag(mark_bits, 7);
        out.mark_progress.ally_infantry_3 = bit_flag(mark_bits, 8);
        out.mark_progress.ally_infantry_4 = bit_flag(mark_bits, 9);
        out.mark_progress.ally_aerial_6 = bit_flag(mark_bits, 10);
        out.mark_progress.ally_sentinel = bit_flag(mark_bits, 11);
        out.mark_progress.raw_bits = mark_bits;
      } else if (legacy_mark_progress_ && payload.size() == 6) {
        out.has_mark_progress = true;
        out.mark_progress.header.stamp = stamp;
        out.mark_progress.hero = payload[0];
        out.mark_progress.engineer = payload[1];
        out.mark_progress.infantry_3 = payload[2];
        out.mark_progress.infantry_4 = payload[3];
        out.mark_progress.aerial_6 = payload[4];
        out.mark_progress.sentinel = payload[5];
        out.warning = "0x020C parsed with legacy 6-byte mark progress layout";
      } else {
        out.warning = "0x020C payload length mismatch";
      }
      break;
    }

    case 0x020E: {
      if (payload.size() != 1) {
        out.warning = "0x020E payload length mismatch";
        return out;
      }

      const uint8_t radar_info = payload[0];
      out.has_radar_sync = true;
      out.radar_sync.header.stamp = stamp;
      out.radar_sync.double_effect_chance = static_cast<uint8_t>(radar_info & 0x03U);
      out.radar_sync.is_double_effect_active = ((radar_info >> 2U) & 0x01U) != 0U;
      out.radar_sync.self_key_level = static_cast<uint8_t>((radar_info >> 3U) & 0x03U);
      out.radar_sync.can_change_key = ((radar_info >> 5U) & 0x01U) != 0U;
      break;
    }

    default:
      break;
  }

  return out;
}

}  // namespace radar_decision_comm
