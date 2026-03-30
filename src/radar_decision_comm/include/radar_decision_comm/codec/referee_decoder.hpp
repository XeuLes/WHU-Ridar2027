#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "radar_decision_comm/io/frame_receiver.hpp"

namespace radar_decision_comm
{

class RefereeDecoder
{
public:
  RefereeDecoder() = default;
  ~RefereeDecoder() = default;

  void configure(bool enable_crc8_check, bool enable_crc16_check);
  void push_bytes(const std::vector<uint8_t> & bytes);
  std::optional<DecodedFrame> try_decode();

private:
  FrameReceiver receiver_;
};

}  // namespace radar_decision_comm
