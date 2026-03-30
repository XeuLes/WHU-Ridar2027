#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace radar_decision_comm::crc
{

uint8_t crc8(const uint8_t * data, std::size_t length, uint8_t init = 0xFF);
uint16_t crc16(const uint8_t * data, std::size_t length, uint16_t init = 0xFFFF);

inline uint8_t crc8(const std::vector<uint8_t> & data, uint8_t init = 0xFF)
{
  return crc8(data.data(), data.size(), init);
}

inline uint16_t crc16(const std::vector<uint8_t> & data, uint16_t init = 0xFFFF)
{
  return crc16(data.data(), data.size(), init);
}

}  // namespace radar_decision_comm::crc
