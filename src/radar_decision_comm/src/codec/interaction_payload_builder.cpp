#include "radar_decision_comm/codec/interaction_payload_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace radar_decision_comm
{

void InteractionPayloadBuilder::append_u16_le(std::vector<uint8_t> & out, uint16_t value)
{
  out.push_back(static_cast<uint8_t>(value & 0xFFU));
  out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void InteractionPayloadBuilder::append_i16_le(std::vector<uint8_t> & out, int16_t value)
{
  append_u16_le(out, static_cast<uint16_t>(value));
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

std::vector<uint8_t> InteractionPayloadBuilder::build_hero_lob_assist_payload(
  uint8_t flags,
  double yaw_field_cw_deg,
  double pitch_deg,
  uint16_t age_ms,
  double impact_error_m,
  double splash_radius_m,
  uint8_t state,
  uint8_t sequence_tag)
{
  const auto clamp_i16 = [](double value) {
      const auto rounded = static_cast<int>(std::lround(value));
      return static_cast<int16_t>(std::clamp(rounded, -32768, 32767));
    };
  const auto clamp_u16 = [](double value) {
      const auto rounded = static_cast<int>(std::lround(value));
      return static_cast<uint16_t>(std::clamp(rounded, 0, 65535));
    };

  std::vector<uint8_t> payload;
  payload.reserve(14);
  payload.push_back(1U);
  payload.push_back(flags);
  append_i16_le(payload, clamp_i16(yaw_field_cw_deg * 100.0));
  append_i16_le(payload, clamp_i16(pitch_deg * 100.0));
  append_u16_le(payload, age_ms);
  append_u16_le(payload, clamp_u16(impact_error_m * 100.0));
  append_u16_le(payload, clamp_u16(splash_radius_m * 100.0));
  payload.push_back(state);
  payload.push_back(sequence_tag);
  return payload;
}

}  // namespace radar_decision_comm
