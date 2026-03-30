#pragma once

#include <cstdint>
#include <vector>

#include "radar_decision_comm/io/frame_sender.hpp"

namespace radar_decision_comm
{

class RefereeEncoder
{
public:
  RefereeEncoder() = default;
  ~RefereeEncoder() = default;

  std::vector<uint8_t> encode_frame(uint16_t cmd_id, const std::vector<uint8_t> & payload);
  void set_seq(uint8_t seq);
  uint8_t next_seq() const;

private:
  FrameSender sender_;
};

}  // namespace radar_decision_comm
