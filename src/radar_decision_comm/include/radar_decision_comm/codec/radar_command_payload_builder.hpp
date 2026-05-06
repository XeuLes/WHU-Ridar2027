#pragma once

#include <cstdint>
#include <vector>

namespace radar_decision_comm
{

class RadarCommandPayloadBuilder
{
public:
  static std::vector<uint8_t> build_radar_double_effect_times_payload(uint8_t times);
};

}  // namespace radar_decision_comm
