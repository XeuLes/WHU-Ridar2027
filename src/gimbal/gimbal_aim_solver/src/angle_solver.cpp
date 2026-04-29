#include "gimbal_aim_solver/angle_solver.hpp"

#include <algorithm>
#include <cmath>

namespace gimbal_aim_solver
{

void AngleSolver::setOffsets(double yaw_offset_deg, double pitch_offset_deg)
{
  yaw_offset_deg_ = yaw_offset_deg;
  pitch_offset_deg_ = pitch_offset_deg;
}

void AngleSolver::setLimits(double yaw_min_deg, double yaw_max_deg,
                            double pitch_min_deg, double pitch_max_deg)
{
  yaw_min_deg_ = yaw_min_deg;
  yaw_max_deg_ = yaw_max_deg;
  pitch_min_deg_ = pitch_min_deg;
  pitch_max_deg_ = pitch_max_deg;
}

TargetAngles AngleSolver::solve(const Eigen::Vector3d& p_gimbal) const
{
  TargetAngles result;

  const double x = p_gimbal.x();  // right
  const double y = p_gimbal.y();  // down
  const double z = p_gimbal.z();  // forward

  // 目标在后方则无效
  if (z <= 1e-6) {
    result.valid = false;
    return result;
  }

  const double yaw_rad = std::atan2(-x, z);
  const double pitch_rad = -std::atan2(y, std::sqrt(x * x + z * z));

  double yaw_deg = yaw_rad * 180.0 / M_PI + yaw_offset_deg_;
  double pitch_deg = pitch_rad * 180.0 / M_PI + pitch_offset_deg_;

  yaw_deg = std::clamp(yaw_deg, yaw_min_deg_, yaw_max_deg_);
  pitch_deg = std::clamp(pitch_deg, pitch_min_deg_, pitch_max_deg_);

  result.valid = true;
  result.yaw_deg = yaw_deg;
  result.pitch_deg = pitch_deg;
  return result;
}

}  // namespace gimbal_aim_solver
