#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace radar_decision_comm
{

struct DecodedFrame
{
  uint8_t seq{0};
  uint16_t cmd_id{0};
  std::vector<uint8_t> payload;
  std::vector<uint8_t> raw_frame;
};

class FrameReceiver
{
public:
  FrameReceiver() = default;
  ~FrameReceiver() = default;

  void configure(bool enable_crc8_check, bool enable_crc16_check);
  void push_bytes(const std::vector<uint8_t> & bytes);
  std::optional<DecodedFrame> try_pop_frame();
  void clear();

private:
  bool enable_crc8_check_{true};
  bool enable_crc16_check_{true};
  std::vector<uint8_t> buffer_;
};

}  // namespace radar_decision_comm
