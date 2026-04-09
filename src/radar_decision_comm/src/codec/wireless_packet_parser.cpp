#include "radar_decision_comm/codec/wireless_packet_parser.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

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

void fill_five_u16_ammo_msg(
  radar_decision_comm::msg::WirelessEnemyAmmo & msg,
  const std::vector<uint8_t> & payload,
  const rclcpp::Time & stamp)
{
  msg.header.stamp = stamp;
  msg.hero = le_u16(payload, 0);
  msg.engineer = 0;
  msg.infantry_3 = le_u16(payload, 2);
  msg.infantry_4 = le_u16(payload, 4);
  msg.aerial_6 = le_u16(payload, 6);
  msg.infantry_5 = 0;
  msg.sentinel = le_u16(payload, 8);
}

bool find_json_value_start(const std::string & json, const std::string & key, std::size_t & pos)
{
  const std::string pattern = "\"" + key + "\"";
  const auto key_pos = json.find(pattern);
  if (key_pos == std::string::npos) {
    return false;
  }

  auto colon_pos = json.find(':', key_pos + pattern.size());
  if (colon_pos == std::string::npos) {
    return false;
  }

  pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  return pos < json.size();
}

bool extract_json_string(const std::string & json, const std::string & key, std::string & out)
{
  std::size_t pos = 0;
  if (!find_json_value_start(json, key, pos) || json[pos] != '"') {
    return false;
  }

  ++pos;
  std::string value;
  bool escape = false;
  for (; pos < json.size(); ++pos) {
    const char ch = json[pos];
    if (escape) {
      value.push_back(ch);
      escape = false;
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '"') {
      out = value;
      return true;
    }
    value.push_back(ch);
  }

  return false;
}

bool extract_json_int64(const std::string & json, const std::string & key, int64_t & out)
{
  std::size_t pos = 0;
  if (!find_json_value_start(json, key, pos)) {
    return false;
  }

  const auto start = pos;
  if (json[pos] == '-') {
    ++pos;
  }
  while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos == start || (pos == start + 1 && json[start] == '-')) {
    return false;
  }

  char * end = nullptr;
  const auto parsed = std::strtoll(json.substr(start, pos - start).c_str(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }

  out = static_cast<int64_t>(parsed);
  return true;
}

bool hex_char_to_nibble(char ch, uint8_t & out)
{
  if (ch >= '0' && ch <= '9') {
    out = static_cast<uint8_t>(ch - '0');
    return true;
  }
  if (ch >= 'a' && ch <= 'f') {
    out = static_cast<uint8_t>(10 + ch - 'a');
    return true;
  }
  if (ch >= 'A' && ch <= 'F') {
    out = static_cast<uint8_t>(10 + ch - 'A');
    return true;
  }
  return false;
}

bool hex_to_bytes(const std::string & hex, std::vector<uint8_t> & out)
{
  out.clear();
  if (hex.size() % 2 != 0) {
    return false;
  }

  out.reserve(hex.size() / 2);
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    uint8_t hi = 0;
    uint8_t lo = 0;
    if (!hex_char_to_nibble(hex[i], hi) || !hex_char_to_nibble(hex[i + 1], lo)) {
      out.clear();
      return false;
    }
    out.push_back(static_cast<uint8_t>((hi << 4U) | lo));
  }

  return true;
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
      if (payload.size() < 10) {
        out.warning = "0x0A03 payload too short";
        return out;
      }
      out.has_enemy_ammo = true;
      fill_five_u16_ammo_msg(out.enemy_ammo, payload, stamp);
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

WirelessBridgePacket WirelessPacketParser::parse_bridge_packet(
  const std::string & json_text,
  const rclcpp::Time & default_stamp) const
{
  WirelessBridgePacket out;
  out.stamp = default_stamp;

  int64_t cmd_id = 0;
  if (!extract_json_int64(json_text, "cmd_id", cmd_id)) {
    out.warning = "bridge json missing cmd_id";
    return out;
  }

  if (cmd_id < 0 || cmd_id > std::numeric_limits<uint16_t>::max()) {
    out.warning = "bridge json cmd_id out of range";
    return out;
  }

  std::string data_hex;
  if (!extract_json_string(json_text, "data_hex", data_hex)) {
    out.warning = "bridge json missing data_hex";
    return out;
  }

  if (!hex_to_bytes(data_hex, out.payload)) {
    out.warning = "bridge json data_hex invalid";
    return out;
  }

  std::string source_type;
  if (extract_json_string(json_text, "source_type", source_type)) {
    out.source_type = source_type;
  }

  int64_t stamp_ns = 0;
  if (extract_json_int64(json_text, "stamp_ns", stamp_ns) && stamp_ns > 0) {
    out.stamp = rclcpp::Time(stamp_ns, RCL_ROS_TIME);
  }

  out.cmd_id = static_cast<uint16_t>(cmd_id);
  out.valid = true;
  return out;
}

WirelessParseResult WirelessPacketParser::parse_bridge_json(
  const std::string & json_text,
  const rclcpp::Time & default_stamp,
  uint16_t * out_cmd_id,
  std::string * out_source_type) const
{
  auto packet = parse_bridge_packet(json_text, default_stamp);
  WirelessParseResult out;

  if (!packet.valid) {
    out.warning = packet.warning;
    return out;
  }

  if (out_cmd_id != nullptr) {
    *out_cmd_id = packet.cmd_id;
  }
  if (out_source_type != nullptr) {
    *out_source_type = packet.source_type;
  }

  out = parse(packet.cmd_id, packet.payload, packet.stamp);
  return out;
}

}  // namespace radar_decision_comm
