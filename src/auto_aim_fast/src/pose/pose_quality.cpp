#include "auto_aim_fast/pose/pose_quality.hpp"

namespace auto_aim_fast::pose
{
float PoseQuality::confidence(const Pose3D & p) const
{
  return p.ok ? 1.0f : 0.0f;
}
}  // namespace auto_aim_fast::pose
