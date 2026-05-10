#include <iomanip>
#include <iostream>
#include <vector>

#include "radar_decision_comm/codec/referee_decoder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"

int main()
{
  radar_decision_comm::RefereeEncoder encoder;
  radar_decision_comm::RefereeDecoder decoder;

  const uint16_t cmd_id = 0x0305;
  std::vector<uint8_t> payload(48, 0);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i);
  }

  auto frame = encoder.encode_frame(cmd_id, payload);
  decoder.push_bytes(frame);
  auto decoded = decoder.try_decode();

  if (!decoded.has_value()) {
    std::cerr << "selfcheck failed: decoder returned nullopt" << std::endl;
    return 1;
  }

  if (decoded->cmd_id != cmd_id) {
    std::cerr << "selfcheck failed: cmd_id mismatch" << std::endl;
    return 2;
  }
  if (
    decoded->raw_frame.size() < 7 || decoded->raw_frame[5] != 0x05 ||
    decoded->raw_frame[6] != 0x03)
  {
    std::cerr << "selfcheck failed: cmd_id is not encoded little-endian" << std::endl;
    return 5;
  }

  if (decoded->payload != payload) {
    std::cerr << "selfcheck failed: payload mismatch" << std::endl;
    return 3;
  }
  if (decoded->raw_frame.size() != 57) {
    std::cerr << "selfcheck failed: 0x0305 frame length mismatch" << std::endl;
    return 4;
  }

  std::cout << "selfcheck ok" << std::endl;
  std::cout << "seq=" << static_cast<int>(decoded->seq) << std::endl;
  std::cout << "cmd_id=0x" << std::hex << std::setw(4) << std::setfill('0') << decoded->cmd_id << std::dec << std::endl;
  std::cout << "payload_size=" << decoded->payload.size() << std::endl;
  std::cout << "frame_bytes=" << decoded->raw_frame.size() << std::endl;
  std::cout << "payload=";
  for (auto b : decoded->payload) {
    std::cout << " 0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
  }
  std::cout << std::dec << std::endl;

  return 0;
}
