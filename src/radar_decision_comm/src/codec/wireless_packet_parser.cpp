#include "radar_decision_comm/codec/wireless_packet_parser.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "radar_decision_comm/msg/tracked_robot.hpp"

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

std::array<uint16_t, 6> enemy_ids_for_color(const std::string & my_color)
{
  if (my_color == "Red") {
    return {101, 102, 103, 104, 105, 107};
  }
  return {1, 2, 3, 4, 5, 7};
}

template<typename MsgT>
void fill_six_u16_msg(
  MsgT & msg,
  const std::vector<uint8_t> & payload,
  const rclcpp::Time & stamp)
{
  msg.header.stamp = stamp;
  msg.hero = le_u16(payload, 0);
  msg.engineer = le_u16(payload, 2);
  msg.infantry_3 = le_u16(payload, 4);
  msg.infantry_4 = le_u16(payload, 6);
  msg.infantry_5 = le_u16(payload, 8);
  msg.sentinel = le_u16(payload, 10);
}

}  // namespace

void WirelessPacketParser::set_my_color(const std::string & my_color)
{
  my_color_ = my_color;
}

WirelessParseResult WirelessPacketParser::parse(
  uint16_t cmd_id,
  const std::vector<uint8_t> & payload,
  const rclcpp::Time & stamp) const
{
  WirelessParseResult out;

  switch (cmd_id) {
    case 0x0A01: {
      if (payload.size() < 24) {
        out.warning = "0x0A01 payload too short";
        return out;
      }

      out.has_enemy_world_state = true;
      out.enemy_world_state.header.stamp = stamp;
      out.enemy_world_state.has_direct_observation = false;
      out.enemy_world_state.has_wireless_observation = true;
      out.enemy_world_state.has_rule_prediction = false;

      const auto ids = enemy_ids_for_color(my_color_);
      for (std::size_t i = 0; i < 6; ++i) {
        const std::size_t off = i * 4;
        const uint16_t x_cm = le_u16(payload, off + 0);
        const uint16_t y_cm = le_u16(payload, off + 2);

        if (x_cm == 0 && y_cm == 0) {
          continue;
        }

        radar_decision_comm::msg::TrackedRobot robot;
        robot.header.stamp = stamp;
        robot.robot_id = ids[i];
        robot.track_id = static_cast<uint32_t>(i + 1);
        robot.is_valid = true;
        robot.confidence = 1.0F;
        robot.image_u = 0.0F;
        robot.image_v = 0.0F;
        robot.camera_position.x = 0.0;
        robot.camera_position.y = 0.0;
        robot.camera_position.z = 0.0;
        robot.field_position.x = static_cast<double>(x_cm) / 100.0;
        robot.field_position.y = static_cast<double>(y_cm) / 100.0;
        robot.field_position.z = 0.0;
        robot.velocity.x = 0.0;
        robot.velocity.y = 0.0;
        robot.velocity.z = 0.0;
        robot.source_mask = 0x04U;
        out.enemy_world_state.robots.push_back(robot);
      }
      break;
    }

    case 0x0A02: {
      if (payload.size() < 12) {
        out.warning = "0x0A02 payload too short";
        return out;
      }
      out.has_enemy_hp = true;
      fill_six_u16_msg(out.enemy_hp, payload, stamp);
      break;
    }

    case 0x0A03: {
      if (payload.size() < 12) {
        out.warning = "0x0A03 payload too short";
        return out;
      }
      out.has_enemy_ammo = true;
      fill_six_u16_msg(out.enemy_ammo, payload, stamp);
      break;
    }

    case 0x0A04: {
      out.has_macro_state = true;
      out.macro_state.header.stamp = stamp;
      out.macro_state.raw_bytes = payload;
      break;
    }

    case 0x0A05: {
      out.has_enemy_buffs = true;
      out.enemy_buffs.header.stamp = stamp;
      out.enemy_buffs.raw_bytes = payload;
      break;
    }

    case 0x0A06: {
      out.has_interference_key = true;
      out.interference_key.header.stamp = stamp;
      out.interference_key.raw_bytes = payload;
      break;
    }

    default:
      break;
  }

  return out;
}

}  // namespace radar_decision_comm
