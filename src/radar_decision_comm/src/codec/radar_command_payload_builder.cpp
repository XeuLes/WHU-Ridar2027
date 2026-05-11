#include "radar_decision_comm/codec/radar_command_payload_builder.hpp"

namespace radar_decision_comm
{

std::vector<uint8_t> RadarCommandPayloadBuilder::build_radar_double_effect_times_payload(uint8_t times)
{
  return build_radar_decision_payload(times, 0U, std::array<uint8_t, 6>{});
}

std::vector<uint8_t> RadarCommandPayloadBuilder::build_radar_decision_payload(
  uint8_t radar_cmd,
  uint8_t password_cmd,
  const std::array<uint8_t, 6> & password)
{
  std::vector<uint8_t> payload;
  payload.reserve(8);
  payload.push_back(radar_cmd);
  payload.push_back(password_cmd);
  payload.insert(payload.end(), password.begin(), password.end());
  return payload;
}

}  // namespace radar_decision_comm
