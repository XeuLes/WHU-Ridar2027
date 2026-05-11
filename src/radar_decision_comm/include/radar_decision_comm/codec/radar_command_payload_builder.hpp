#pragma once

#include <cstdint>
#include <array>
#include <vector>

namespace radar_decision_comm
{

class RadarCommandPayloadBuilder
{
public:
  static std::vector<uint8_t> build_radar_double_effect_times_payload(uint8_t times);
  static std::vector<uint8_t> build_radar_decision_payload(
    uint8_t radar_cmd,
    uint8_t password_cmd,
    const std::array<uint8_t, 6> & password);
};

}  // namespace radar_decision_comm
