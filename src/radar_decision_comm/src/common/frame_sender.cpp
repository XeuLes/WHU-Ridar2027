#include "radar_decision_comm/io/frame_sender.hpp"

#include "radar_decision_comm/common/crc.hpp"

namespace radar_decision_comm
{

std::vector<uint8_t> FrameSender::build_frame(uint16_t cmd_id, const std::vector<uint8_t> & payload)
{
  std::vector<uint8_t> frame;
  frame.reserve(5 + 2 + payload.size() + 2);

  const uint16_t data_length = static_cast<uint16_t>(payload.size());

  frame.push_back(0xA5);
  frame.push_back(static_cast<uint8_t>(data_length & 0xFFU));
  frame.push_back(static_cast<uint8_t>((data_length >> 8U) & 0xFFU));
  frame.push_back(next_seq_);
  frame.push_back(0x00);

  frame[4] = radar_decision_comm::crc::crc8(frame.data(), 4);

  frame.push_back(static_cast<uint8_t>(cmd_id & 0xFFU));
  frame.push_back(static_cast<uint8_t>((cmd_id >> 8U) & 0xFFU));

  frame.insert(frame.end(), payload.begin(), payload.end());

  const uint16_t crc16_value = radar_decision_comm::crc::crc16(frame.data(), frame.size());
  frame.push_back(static_cast<uint8_t>(crc16_value & 0xFFU));
  frame.push_back(static_cast<uint8_t>((crc16_value >> 8U) & 0xFFU));

  ++next_seq_;
  return frame;
}

void FrameSender::set_seq(uint8_t seq)
{
  next_seq_ = seq;
}

uint8_t FrameSender::next_seq() const
{
  return next_seq_;
}

}  // namespace radar_decision_comm
