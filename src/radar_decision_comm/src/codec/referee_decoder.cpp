#include "radar_decision_comm/codec/referee_decoder.hpp"

namespace radar_decision_comm
{

void RefereeDecoder::configure(bool enable_crc8_check, bool enable_crc16_check)
{
  receiver_.configure(enable_crc8_check, enable_crc16_check);
}

void RefereeDecoder::push_bytes(const std::vector<uint8_t> & bytes)
{
  receiver_.push_bytes(bytes);
}

std::optional<DecodedFrame> RefereeDecoder::try_decode()
{
  return receiver_.try_pop_frame();
}

}  // namespace radar_decision_comm
