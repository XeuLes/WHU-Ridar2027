#include "radar_decision_comm/codec/referee_packet_parser.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

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
      if (payload.size() < 11) {
        out.warning = "0x0001 payload too short";
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
      if (payload.size() >= 32) {
        // legacy/old project compatible: red block + blue block
        const std::size_t base = (my_color_ == "Red") ? 16 : 0;

        out.has_enemy_hp = true;
        out.enemy_hp.header.stamp = stamp;
        out.enemy_hp.hero = le_u16(payload, base + 0);
        out.enemy_hp.engineer = le_u16(payload, base + 2);
        out.enemy_hp.infantry_3 = le_u16(payload, base + 4);
        out.enemy_hp.infantry_4 = le_u16(payload, base + 6);
        out.enemy_hp.infantry_5 = le_u16(payload, base + 8);
        out.enemy_hp.sentinel = le_u16(payload, base + 10);
      } else if (payload.size() >= 16) {
        // 2026 current protocol snippet: ally-only compact format
        out.has_enemy_hp = true;
        out.enemy_hp.header.stamp = stamp;
        out.enemy_hp.hero = le_u16(payload, 0);
        out.enemy_hp.engineer = le_u16(payload, 2);
        out.enemy_hp.infantry_3 = le_u16(payload, 4);
        out.enemy_hp.infantry_4 = le_u16(payload, 6);
        out.enemy_hp.infantry_5 = 0;
        out.enemy_hp.sentinel = le_u16(payload, 10);
        out.warning = "0x0003 parsed with 16-byte ally-only layout; infantry_5 set to 0";
      } else {
        out.warning = "0x0003 payload too short";
      }
      break;
    }

    case 0x0105: {
      if (payload.size() < 3) {
        out.warning = "0x0105 payload too short";
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

    case 0x020C: {
      if (payload.size() >= 6) {
        out.has_mark_progress = true;
        out.mark_progress.header.stamp = stamp;
        out.mark_progress.hero = payload[0];
        out.mark_progress.engineer = payload[1];
        out.mark_progress.infantry_3 = payload[2];
        out.mark_progress.infantry_4 = payload[3];
        out.mark_progress.infantry_5 = payload[4];
        out.mark_progress.sentinel = payload[5];
      } else if (payload.size() == 2) {
        if (legacy_mark_progress_) {
          out.warning = "0x020C received 2-byte compact format but legacy 6-byte parsing is enabled";
        } else {
          out.warning = "0x020C 2-byte compact format reserved; field mapping not implemented yet";
        }
      } else {
        out.warning = "0x020C payload too short";
      }
      break;
    }

    case 0x020E: {
      if (payload.empty()) {
        out.warning = "0x020E payload too short";
        return out;
      }

      const uint8_t radar_info = payload[0];
      out.has_radar_sync = true;
      out.radar_sync.header.stamp = stamp;
      out.radar_sync.double_effect_chance = static_cast<uint8_t>(radar_info & 0x03U);
      out.radar_sync.is_double_effect_active = ((radar_info >> 2U) & 0x01U) != 0U;
      out.radar_sync.enemy_key_level = static_cast<uint8_t>((radar_info >> 3U) & 0x03U);
      out.radar_sync.can_change_key = ((radar_info >> 5U) & 0x01U) != 0U;
      break;
    }

    default:
      break;
  }

  return out;
}

}  // namespace radar_decision_comm
