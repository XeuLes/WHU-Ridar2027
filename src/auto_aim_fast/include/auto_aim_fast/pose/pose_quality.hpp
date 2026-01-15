#pragma once
#include "auto_aim_fast/pose/pnp_solver.hpp"

namespace auto_aim_fast::pose
{
class PoseQuality
{
public:
  PoseQuality() = default;
  float confidence(const Pose3D & p) const;
};
}  // namespace auto_aim_fast::pose
