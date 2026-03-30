#pragma once

#include <chrono>

namespace radar_decision_comm
{

using SteadyTimePoint = std::chrono::steady_clock::time_point;
using SystemTimePoint = std::chrono::system_clock::time_point;

}  // namespace radar_decision_comm
