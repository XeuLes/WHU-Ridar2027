#include "radar_decision_comm/io/frame_receiver.hpp"

#include <algorithm>

#include "radar_decision_comm/common/crc.hpp"

namespace radar_decision_comm
{

void FrameReceiver::configure(bool enable_crc8_check, bool enable_crc16_check)
{
  enable_crc8_check_ = enable_crc8_check;
  enable_crc16_check_ = enable_crc16_check;
}

void FrameReceiver::push_bytes(const std::vector<uint8_t> & bytes)
{
  buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
}

void FrameReceiver::clear()
{
  buffer_.clear();
}

std::optional<DecodedFrame> FrameReceiver::try_pop_frame()
{
  while (true) {
    if (buffer_.empty()) {
      return std::nullopt;
    }

    auto sof_it = std::find(buffer_.begin(), buffer_.end(), static_cast<uint8_t>(0xA5));
    if (sof_it == buffer_.end()) {
      buffer_.clear();
      return std::nullopt;
    }

    if (sof_it != buffer_.begin()) {
      buffer_.erase(buffer_.begin(), sof_it);
    }

    if (buffer_.size() < 5) {
      return std::nullopt;
    }

    if (enable_crc8_check_) {
      const uint8_t header_crc = radar_decision_comm::crc::crc8(buffer_.data(), 4);
      if (header_crc != buffer_[4]) {
        buffer_.erase(buffer_.begin());
        continue;
      }
    }

    const uint16_t data_length =
      static_cast<uint16_t>(buffer_[1]) |
      (static_cast<uint16_t>(buffer_[2]) << 8U);

    const std::size_t total_length = 5 + 2 + static_cast<std::size_t>(data_length) + 2;
    if (buffer_.size() < total_length) {
      return std::nullopt;
    }

    if (enable_crc16_check_) {
      const uint16_t expected_crc16 =
        static_cast<uint16_t>(buffer_[total_length - 2]) |
        (static_cast<uint16_t>(buffer_[total_length - 1]) << 8U);
      const uint16_t actual_crc16 =
        radar_decision_comm::crc::crc16(buffer_.data(), total_length - 2);

      if (expected_crc16 != actual_crc16) {
        buffer_.erase(buffer_.begin());
        continue;
      }
    }

    DecodedFrame out;
    out.seq = buffer_[3];
    out.cmd_id =
      static_cast<uint16_t>(buffer_[5]) |
      (static_cast<uint16_t>(buffer_[6]) << 8U);
    out.payload.assign(buffer_.begin() + 7, buffer_.begin() + 7 + data_length);
    out.raw_frame.assign(buffer_.begin(), buffer_.begin() + total_length);

    buffer_.erase(buffer_.begin(), buffer_.begin() + total_length);
    return out;
  }
}

}  // namespace radar_decision_comm
