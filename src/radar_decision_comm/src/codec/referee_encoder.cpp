#include "radar_decision_comm/codec/referee_encoder.hpp"

namespace radar_decision_comm
{

std::vector<uint8_t> RefereeEncoder::encode_frame(uint16_t cmd_id, const std::vector<uint8_t> & payload)
{
  return sender_.build_frame(cmd_id, payload);
}

void RefereeEncoder::set_seq(uint8_t seq)
{
  sender_.set_seq(seq);
}

uint8_t RefereeEncoder::next_seq() const
{
  return sender_.next_seq();
}

}  // namespace radar_decision_comm
