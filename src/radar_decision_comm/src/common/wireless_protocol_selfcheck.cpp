#include <iostream>

#include "radar_decision_comm/codec/wireless_packet_parser.hpp"

int main()
{
  radar_decision_comm::WirelessPacketParser parser;

  parser.set_my_color("Red");

  auto make_pdu_json = [](
    const char * source_type,
    uint16_t cmd_id,
    const std::string & data_hex,
    int64_t stamp_ns)
    {
      return std::string("{") +
             "\"source_type\":\"" + source_type + "\"," +
             "\"cmd_id\":" + std::to_string(cmd_id) + "," +
             "\"seq\":1," +
             "\"packet_len\":0," +
             "\"data_len\":" + std::to_string(data_hex.size() / 2) + "," +
             "\"cmd_name\":\"selfcheck\"," +
             "\"data_hex\":\"" + data_hex + "\"," +
             "\"frame_hex\":\"\"," +
             "\"stamp_ns\":" + std::to_string(stamp_ns) + "," +
             "\"msg_index\":1," +
             "\"crc32\":0"
             "}";
    };

  // 0x0A01
  const auto parsed_pos = parser.parse_bridge_json(
    make_pdu_json("info", 0x0A01, "3a026c0200000000000000000000000000000000d0072003", 123456789),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_pos.has_enemy_world_state || parsed_pos.enemy_world_state.robots.size() != 2) {
    std::cerr << "wireless selfcheck failed: 0x0A01 parse mismatch" << std::endl;
    return 2;
  }

  // 0x0A02
  const auto parsed_hp = parser.parse_bridge_json(
    make_pdu_json("info", 0x0A02, "640050003c00320028007800", 123456790),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_hp.has_enemy_hp || parsed_hp.enemy_hp.hero != 100 || parsed_hp.enemy_hp.sentinel != 120) {
    std::cerr << "wireless selfcheck failed: 0x0A02 parse mismatch" << std::endl;
    return 4;
  }

  // 0x0A03
  const auto parsed_ammo = parser.parse_bridge_json(
    make_pdu_json("info", 0x0A03, "1e0014000a0008000600", 123456791),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_ammo.has_enemy_ammo || parsed_ammo.enemy_ammo.hero != 30 ||
    parsed_ammo.enemy_ammo.infantry_3 != 20 ||
    parsed_ammo.enemy_ammo.infantry_4 != 10 || parsed_ammo.enemy_ammo.aerial_6 != 8 ||
    parsed_ammo.enemy_ammo.sentinel != 6)
  {
    std::cerr << "wireless selfcheck failed: 0x0A03 parse mismatch" << std::endl;
    return 6;
  }

  // 0x0A04
  const auto parsed_macro = parser.parse_bridge_json(
    make_pdu_json("info", 0x0A04, "6400c80003020000", 123456792),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_macro.has_macro_state || parsed_macro.macro_state.raw_bytes.size() != 8 ||
    parsed_macro.macro_state.coins_remaining != 100 ||
    parsed_macro.macro_state.coins_total != 200 ||
    parsed_macro.macro_state.status_bits != 0x00000203U)
  {
    std::cerr << "wireless selfcheck failed: 0x0A04 parse mismatch" << std::endl;
    return 8;
  }

  // 0x0A05
  const auto parsed_buffs = parser.parse_bridge_json(
    make_pdu_json(
      "info",
      0x0A05,
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223",
      123456793),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_buffs.has_enemy_buffs || parsed_buffs.enemy_buffs.raw_bytes.size() != 36) {
    std::cerr << "wireless selfcheck failed: 0x0A05 parse mismatch" << std::endl;
    return 10;
  }

  // 0x0A06
  const auto parsed_key = parser.parse_bridge_json(
    make_pdu_json("jam", 0x0A06, "414243313233", 123456794),
    rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_key.has_interference_key || parsed_key.interference_key.raw_bytes.size() != 6 ||
    parsed_key.interference_key.key[0] != static_cast<uint8_t>('A') ||
    parsed_key.interference_key.key[5] != static_cast<uint8_t>('3'))
  {
    std::cerr << "wireless selfcheck failed: 0x0A06 parse mismatch" << std::endl;
    return 12;
  }

  const auto & hero = parsed_pos.enemy_world_state.robots[0];
  const auto & sentinel = parsed_pos.enemy_world_state.robots[1];

  std::cout << "wireless selfcheck ok" << std::endl;
  std::cout << "robots=" << parsed_pos.enemy_world_state.robots.size() << std::endl;
  std::cout << "hero_id=" << hero.robot_id << " hero_xy=("
            << hero.field_position.x << "," << hero.field_position.y << ")" << std::endl;
  std::cout << "sentinel_id=" << sentinel.robot_id << " sentinel_xy=("
            << sentinel.field_position.x << "," << sentinel.field_position.y << ")" << std::endl;
  std::cout << "hp_hero=" << parsed_hp.enemy_hp.hero
            << " hp_sentinel=" << parsed_hp.enemy_hp.sentinel << std::endl;
  std::cout << "ammo_hero=" << parsed_ammo.enemy_ammo.hero
            << " ammo_infantry3=" << parsed_ammo.enemy_ammo.infantry_3
            << " ammo_infantry4=" << parsed_ammo.enemy_ammo.infantry_4
            << " ammo_aerial6=" << parsed_ammo.enemy_ammo.aerial_6
            << " ammo_sentinel=" << parsed_ammo.enemy_ammo.sentinel << std::endl;
  std::cout << "macro_bytes=" << parsed_macro.macro_state.raw_bytes.size()
            << " buffs_bytes=" << parsed_buffs.enemy_buffs.raw_bytes.size()
            << " key_bytes=" << parsed_key.interference_key.raw_bytes.size() << std::endl;

  return 0;
}
