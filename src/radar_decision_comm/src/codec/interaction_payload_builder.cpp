#include "radar_decision_comm/codec/interaction_payload_builder.hpp"

#include <cstring>

namespace radar_decision_comm
{

void InteractionPayloadBuilder::append_u16_le(std::vector<uint8_t> & out, uint16_t value)
{
  out.push_back(static_cast<uint8_t>(value & 0xFFU));
  out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void InteractionPayloadBuilder::append_float_le(std::vector<uint8_t> & out, float value)
{
  uint8_t bytes[sizeof(float)];
  std::memcpy(bytes, &value, sizeof(float));
  out.insert(out.end(), bytes, bytes + sizeof(float));
}

std::vector<uint8_t> InteractionPayloadBuilder::build_sentry_alert_angle_payload(
  uint16_t robot_id,
  float distance,
  uint16_t quadrant)
{
  std::vector<uint8_t> payload;
  payload.reserve(8);
  append_u16_le(payload, robot_id);
  append_float_le(payload, distance);
  append_u16_le(payload, quadrant);
  return payload;
}

std::vector<uint8_t> InteractionPayloadBuilder::build_sentry_field_state_payload(
  const std::array<EnemyFieldStateSlot, 6> & slots)
{
  std::vector<uint8_t> payload;
  payload.reserve(6 * (1 + 4 + 4));
  for (const auto & slot : slots) {
    payload.push_back(slot.valid);
    append_float_le(payload, slot.x);
    append_float_le(payload, slot.y);
  }
  return payload;
}

std::vector<uint8_t> InteractionPayloadBuilder::build_hero_alert_flag_payload(bool is_alert)
{
  return std::vector<uint8_t>{static_cast<uint8_t>(is_alert ? 0xFFU : 0x00U)};
}

std::vector<uint8_t> InteractionPayloadBuilder::build_double_effect_forward_payload(uint8_t times)
{
  return std::vector<uint8_t>{times};
}

}  // namespace radar_decision_comm
