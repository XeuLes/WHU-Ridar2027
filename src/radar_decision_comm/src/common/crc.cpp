#include "radar_decision_comm/common/crc.hpp"

namespace radar_decision_comm::crc
{

uint8_t crc8(const uint8_t * data, std::size_t length, uint8_t init)
{
  uint8_t crc = init;
  for (std::size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 0x01U) {
        crc = static_cast<uint8_t>((crc >> 1U) ^ 0x8CU);
      } else {
        crc = static_cast<uint8_t>(crc >> 1U);
      }
    }
  }
  return crc;
}

uint16_t crc16(const uint8_t * data, std::size_t length, uint16_t init)
{
  uint16_t crc = init;
  for (std::size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 0x0001U) {
        crc = static_cast<uint16_t>((crc >> 1U) ^ 0x8408U);
      } else {
        crc = static_cast<uint16_t>(crc >> 1U);
      }
    }
  }
  return crc;
}

}  // namespace radar_decision_comm::crc
