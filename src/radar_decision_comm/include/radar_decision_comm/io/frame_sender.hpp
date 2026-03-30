#pragma once

#include <cstdint>
#include <vector>

namespace radar_decision_comm
{

class FrameSender
{
public:
  FrameSender() = default;
  ~FrameSender() = default;

  std::vector<uint8_t> build_frame(uint16_t cmd_id, const std::vector<uint8_t> & payload);
  void set_seq(uint8_t seq);
  uint8_t next_seq() const;

private:
  uint8_t next_seq_{0};
};

}  // namespace radar_decision_comm
