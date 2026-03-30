#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace radar_decision_comm
{

enum class SourceType : uint8_t
{
  kUnknown = 0,
  kPerception = 1,
  kReferee = 2,
  kWireless = 3,
  kClient = 4,
  kRule = 5
};

struct ByteBuffer
{
  std::vector<uint8_t> data;
};

}  // namespace radar_decision_comm
