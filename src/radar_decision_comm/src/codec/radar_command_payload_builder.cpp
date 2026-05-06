#include "radar_decision_comm/codec/radar_command_payload_builder.hpp"

namespace radar_decision_comm
{

std::vector<uint8_t> RadarCommandPayloadBuilder::build_radar_double_effect_times_payload(uint8_t times)
{
  return std::vector<uint8_t>{times};
}

}  // namespace radar_decision_comm
