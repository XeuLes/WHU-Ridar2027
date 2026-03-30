#include <cmath>
#include <iostream>
#include <vector>

#include "radar_decision_comm/codec/referee_decoder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"
#include "radar_decision_comm/codec/wireless_packet_parser.hpp"

int main()
{
  radar_decision_comm::RefereeEncoder encoder;
  radar_decision_comm::RefereeDecoder decoder;
  radar_decision_comm::WirelessPacketParser parser;

  parser.set_my_color("Red");
  decoder.configure(true, true);

  // 0x0A01
  std::vector<uint8_t> pos_payload(24, 0);
  auto put16 = [&pos_payload](std::size_t off, uint16_t v) {
    pos_payload[off] = static_cast<uint8_t>(v & 0xFFU);
    pos_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
  };
  put16(0, 1850);
  put16(2, 620);
  put16(20, 2000);
  put16(22, 800);

  auto frame_pos = encoder.encode_frame(0x0A01, pos_payload);
  decoder.push_bytes(frame_pos);
  auto decoded_pos = decoder.try_decode();
  if (!decoded_pos.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A01 decoder nullopt" << std::endl;
    return 1;
  }
  auto parsed_pos = parser.parse(decoded_pos->cmd_id, decoded_pos->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_pos.has_enemy_world_state || parsed_pos.enemy_world_state.robots.size() != 2) {
    std::cerr << "wireless selfcheck failed: 0x0A01 parse mismatch" << std::endl;
    return 2;
  }

  // 0x0A02
  std::vector<uint8_t> hp_payload(12, 0);
  auto put16hp = [&hp_payload](std::size_t off, uint16_t v) {
    hp_payload[off] = static_cast<uint8_t>(v & 0xFFU);
    hp_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
  };
  put16hp(0, 100);
  put16hp(2, 80);
  put16hp(4, 60);
  put16hp(6, 50);
  put16hp(8, 40);
  put16hp(10, 120);
  auto frame_hp = encoder.encode_frame(0x0A02, hp_payload);
  decoder.push_bytes(frame_hp);
  auto decoded_hp = decoder.try_decode();
  if (!decoded_hp.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A02 decoder nullopt" << std::endl;
    return 3;
  }
  auto parsed_hp = parser.parse(decoded_hp->cmd_id, decoded_hp->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_hp.has_enemy_hp || parsed_hp.enemy_hp.hero != 100 || parsed_hp.enemy_hp.sentinel != 120) {
    std::cerr << "wireless selfcheck failed: 0x0A02 parse mismatch" << std::endl;
    return 4;
  }

  // 0x0A03
  std::vector<uint8_t> ammo_payload(12, 0);
  auto put16ammo = [&ammo_payload](std::size_t off, uint16_t v) {
    ammo_payload[off] = static_cast<uint8_t>(v & 0xFFU);
    ammo_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
  };
  put16ammo(0, 30);
  put16ammo(2, 20);
  put16ammo(4, 10);
  put16ammo(6, 8);
  put16ammo(8, 6);
  put16ammo(10, 50);
  auto frame_ammo = encoder.encode_frame(0x0A03, ammo_payload);
  decoder.push_bytes(frame_ammo);
  auto decoded_ammo = decoder.try_decode();
  if (!decoded_ammo.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A03 decoder nullopt" << std::endl;
    return 5;
  }
  auto parsed_ammo = parser.parse(decoded_ammo->cmd_id, decoded_ammo->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_ammo.has_enemy_ammo || parsed_ammo.enemy_ammo.hero != 30 || parsed_ammo.enemy_ammo.sentinel != 50) {
    std::cerr << "wireless selfcheck failed: 0x0A03 parse mismatch" << std::endl;
    return 6;
  }

  // 0x0A04
  auto frame_macro = encoder.encode_frame(0x0A04, std::vector<uint8_t>{1, 2, 3, 4});
  decoder.push_bytes(frame_macro);
  auto decoded_macro = decoder.try_decode();
  if (!decoded_macro.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A04 decoder nullopt" << std::endl;
    return 7;
  }
  auto parsed_macro = parser.parse(decoded_macro->cmd_id, decoded_macro->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_macro.has_macro_state || parsed_macro.macro_state.raw_bytes.size() != 4) {
    std::cerr << "wireless selfcheck failed: 0x0A04 parse mismatch" << std::endl;
    return 8;
  }

  // 0x0A05
  auto frame_buffs = encoder.encode_frame(0x0A05, std::vector<uint8_t>{5, 6, 7, 8});
  decoder.push_bytes(frame_buffs);
  auto decoded_buffs = decoder.try_decode();
  if (!decoded_buffs.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A05 decoder nullopt" << std::endl;
    return 9;
  }
  auto parsed_buffs = parser.parse(decoded_buffs->cmd_id, decoded_buffs->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_buffs.has_enemy_buffs || parsed_buffs.enemy_buffs.raw_bytes.size() != 4) {
    std::cerr << "wireless selfcheck failed: 0x0A05 parse mismatch" << std::endl;
    return 10;
  }

  // 0x0A06
  auto frame_key = encoder.encode_frame(0x0A06, std::vector<uint8_t>{9, 10, 11, 12});
  decoder.push_bytes(frame_key);
  auto decoded_key = decoder.try_decode();
  if (!decoded_key.has_value()) {
    std::cerr << "wireless selfcheck failed: 0x0A06 decoder nullopt" << std::endl;
    return 11;
  }
  auto parsed_key = parser.parse(decoded_key->cmd_id, decoded_key->payload, rclcpp::Time(0, 0, RCL_ROS_TIME));
  if (!parsed_key.has_interference_key || parsed_key.interference_key.raw_bytes.size() != 4) {
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
            << " ammo_sentinel=" << parsed_ammo.enemy_ammo.sentinel << std::endl;
  std::cout << "macro_bytes=" << parsed_macro.macro_state.raw_bytes.size()
            << " buffs_bytes=" << parsed_buffs.enemy_buffs.raw_bytes.size()
            << " key_bytes=" << parsed_key.interference_key.raw_bytes.size() << std::endl;

  return 0;
}
