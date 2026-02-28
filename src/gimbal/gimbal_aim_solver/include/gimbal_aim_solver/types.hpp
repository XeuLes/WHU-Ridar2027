#pragma once

#include <Eigen/Core>

namespace gimbal_aim_solver
{

struct TargetAngles
{
  bool valid{false};
  double yaw_deg{0.0};
  double pitch_deg{0.0};
};

struct GimbalState
{
  bool valid{false};
  double yaw_deg{0.0};
  double pitch_deg{0.0};
};

}  // namespace gimbal_aim_solver