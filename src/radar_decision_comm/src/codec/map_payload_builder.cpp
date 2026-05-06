#include "radar_decision_comm/codec/map_payload_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace radar_decision_comm
{

int MapPayloadBuilder::opponent_slot_from_robot_id(const std::string & my_color, uint16_t robot_id)
{
  if (my_color == "Red") {
    switch (robot_id) {
      case 101: return 0;
      case 102: return 1;
      case 103: return 2;
      case 104: return 3;
      case 106: return 4;
      case 107: return 5;
      default: return -1;
    }
  }

  switch (robot_id) {
    case 1: return 0;
    case 2: return 1;
    case 3: return 2;
    case 4: return 3;
    case 6: return 4;
    case 7: return 5;
    default: return -1;
  }
}

int MapPayloadBuilder::ally_slot_from_robot_id(const std::string & my_color, uint16_t robot_id)
{
  if (my_color == "Red") {
    switch (robot_id) {
      case 1: return 0;
      case 2: return 1;
      case 3: return 2;
      case 4: return 3;
      case 6: return 4;
      case 7: return 5;
      default: return -1;
    }
  }

  switch (robot_id) {
    case 101: return 0;
    case 102: return 1;
    case 103: return 2;
    case 104: return 3;
    case 106: return 4;
    case 107: return 5;
    default: return -1;
  }
}

uint16_t MapPayloadBuilder::meter_to_cm_u16(double value_m)
{
  const double cm = value_m * 100.0;
  const double clamped = std::clamp(cm, 0.0, 65535.0);
  return static_cast<uint16_t>(std::lround(clamped));
}

void MapPayloadBuilder::append_u16_le(std::vector<uint8_t> & out, uint16_t value)
{
  out.push_back(static_cast<uint8_t>(value & 0xFFU));
  out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

MapPayloadBuilder::PackedSlots MapPayloadBuilder::build_slots(
  const radar_decision_comm::msg::MapPublishRequest & msg,
  const std::string & my_color)
{
  PackedSlots slots{};

  for (const auto & robot : msg.robots) {
    auto * targets = &slots.opponent;
    int slot = opponent_slot_from_robot_id(my_color, robot.robot_id);
    if (slot < 0) {
      slot = ally_slot_from_robot_id(my_color, robot.robot_id);
      targets = &slots.ally;
    }
    if (slot < 0) {
      continue;
    }

    auto & target = (*targets)[static_cast<std::size_t>(slot)];
    target.x_cm = meter_to_cm_u16(robot.field_position.x);
    target.y_cm = meter_to_cm_u16(robot.field_position.y);
    target.valid = true;
  }

  return slots;
}

std::vector<uint8_t> MapPayloadBuilder::build_legacy24(
  const PackedSlots & slots)
{
  std::vector<uint8_t> payload;
  payload.reserve(24);

  for (const auto & target : slots.opponent) {
    append_u16_le(payload, target.x_cm);
    append_u16_le(payload, target.y_cm);
  }

  return payload;
}

std::vector<uint8_t> MapPayloadBuilder::build_rm2026_48(
  const PackedSlots & slots)
{
  std::vector<uint8_t> payload;
  payload.reserve(48);

  const auto append_targets = [&payload](const std::array<PackedTarget, 6> & targets) {
      for (const auto & target : targets) {
        MapPayloadBuilder::append_u16_le(payload, target.x_cm);
        MapPayloadBuilder::append_u16_le(payload, target.y_cm);
      }
    };

  append_targets(slots.opponent);
  append_targets(slots.ally);

  return payload;
}

MapPayloadBuildResult MapPayloadBuilder::build_payload(
  const radar_decision_comm::msg::MapPublishRequest & msg,
  const std::string & my_color,
  const std::string & layout)
{
  MapPayloadBuildResult out;
  const auto slots = build_slots(msg, my_color);

  if (layout == "legacy24") {
    out.payload = build_legacy24(slots);
    out.layout_used = "legacy24";
    return out;
  }

  if (
    layout == "rm2026" || layout == "rm2026_48" || layout == "proto48_operational" ||
    layout == "proto48_candidate" || layout == "proto48_stub")
  {
    out.payload = build_rm2026_48(slots);
    out.layout_used = "rm2026_48";
    return out;
  }

  out.payload = build_rm2026_48(slots);
  out.layout_used = "rm2026_48";
  out.used_fallback = true;
  return out;
}

}  // namespace radar_decision_comm
