#include "auto_aim_fast/prediction/aim_solution.hpp"

namespace auto_aim_fast::prediction
{
AimSolution AimSolver::solve(const auto_aim_fast::tracking::TrackOutput & trk) const
{
  AimSolution s;
  if (!trk.has_target) {
    s.ok = false;
    return s;
  }

  // Placeholder: direct mapping
  s.ok = true;
  s.yaw = trk.yaw;
  s.pitch = trk.pitch + ballistic_.pitchCompensation(trk.dist);
  s.confidence = trk.confidence;
  return s;
}
}  // namespace auto_aim_fast::prediction
