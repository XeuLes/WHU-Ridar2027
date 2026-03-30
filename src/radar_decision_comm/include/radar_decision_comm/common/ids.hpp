#pragma once

#include <cstdint>

namespace radar_decision_comm
{

struct RobotIds
{
  static constexpr uint16_t kRedRadar = 9;
  static constexpr uint16_t kBlueRadar = 109;
  static constexpr uint16_t kRedHero = 1;
  static constexpr uint16_t kBlueHero = 101;
  static constexpr uint16_t kRedSentinel = 7;
  static constexpr uint16_t kBlueSentinel = 107;
};

}  // namespace radar_decision_comm
