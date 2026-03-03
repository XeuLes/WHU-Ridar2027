#include "laser_aim_controller/angle_solver.hpp"

#include <algorithm>
#include <cmath>

namespace laser_aim_controller
{

void AngleSolver::setLimits(
  double yaw_min_deg, double yaw_max_deg,
  double pitch_min_deg, double pitch_max_deg)
{
  yaw_min_deg_ = yaw_min_deg;
  yaw_max_deg_ = yaw_max_deg;
  pitch_min_deg_ = pitch_min_deg;
  pitch_max_deg_ = pitch_max_deg;
}

CommandAngles AngleSolver::solveIncrement(const Eigen::Vector3d& v) const
{
  CommandAngles out;

  const double x = v.x();  // right
  const double y = v.y();  // down
  const double z = v.z();  // forward

  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 1e-6) {
    out.valid = false;
    return out;
  }

  // yaw: left positive, right negative
  const double yaw_rad = std::atan2(-x, z);

  // pitch: down positive, up negative
  const double pitch_rad = std::atan2(y, std::sqrt(x * x + z * z));

  double yaw_deg = yaw_rad * 180.0 / M_PI;
  double pitch_deg = pitch_rad * 180.0 / M_PI;

  yaw_deg = std::clamp(yaw_deg, yaw_min_deg_, yaw_max_deg_);
  pitch_deg = std::clamp(pitch_deg, pitch_min_deg_, pitch_max_deg_);

  out.valid = true;
  out.yaw_deg = yaw_deg;
  out.pitch_deg = pitch_deg;
  return out;
}

}  // namespace laser_aim_controller