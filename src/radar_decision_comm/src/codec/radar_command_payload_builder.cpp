#include "radar_decision_comm/codec/radar_command_payload_builder.hpp"

namespace radar_decision_comm
{

void RadarCommandPayloadBuilder::append_u16_le(std::vector<uint8_t> & out, uint16_t value)
{
  out.push_back(static_cast<uint8_t>(value & 0xFFU));
  out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

std::vector<uint8_t> RadarCommandPayloadBuilder::build_radar_double_effect_times_payload(uint16_t times)
{
  std::vector<uint8_t> payload;
  payload.reserve(2);
  append_u16_le(payload, times);
  return payload;
}

}  // namespace radar_decision_comm
