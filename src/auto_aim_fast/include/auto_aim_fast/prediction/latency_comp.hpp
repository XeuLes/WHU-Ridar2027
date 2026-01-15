#pragma once
#include <cstdint>

namespace auto_aim_fast::prediction
{
class LatencyComp
{
public:
  LatencyComp() = default;
  // Estimate total latency (ns). Placeholder constant for now.
  int64_t estimateLatencyNs() const { return 0; }
};
}  // namespace auto_aim_fast::prediction
