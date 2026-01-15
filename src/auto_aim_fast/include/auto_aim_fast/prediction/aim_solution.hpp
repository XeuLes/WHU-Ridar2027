#pragma once
#include "auto_aim_fast/tracking/target_tracker.hpp"
#include "auto_aim_fast/prediction/latency_comp.hpp"
#include "auto_aim_fast/prediction/ballistic.hpp"

namespace auto_aim_fast::prediction
{
struct AimSolution
{
  bool ok{false};
  float yaw{0}, pitch{0};
  float confidence{0};
};

class AimSolver
{
public:
  AimSolver() = default;

  AimSolution solve(const auto_aim_fast::tracking::TrackOutput & trk) const;

private:
  LatencyComp latency_;
  Ballistic ballistic_;
};
}  // namespace auto_aim_fast::prediction
