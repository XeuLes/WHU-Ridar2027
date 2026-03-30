#pragma once

#include <cstdint>
#include <vector>

namespace radar_decision_comm
{

class RadarCommandPayloadBuilder
{
public:
  static std::vector<uint8_t> build_radar_double_effect_times_payload(uint16_t times);

private:
  static void append_u16_le(std::vector<uint8_t> & out, uint16_t value);
};

}  // namespace radar_decision_comm
