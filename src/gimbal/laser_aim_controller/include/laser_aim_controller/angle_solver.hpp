#pragma once

#include <Eigen/Core>
#include "laser_aim_controller/types.hpp"

namespace laser_aim_controller
{

class AngleSolver
{
public:
  AngleSolver() = default;

  void setLimits(
    double yaw_min_deg, double yaw_max_deg,
    double pitch_min_deg, double pitch_max_deg);

  CommandAngles solveIncrement(const Eigen::Vector3d& target_from_laser) const;

private:
  double yaw_min_deg_{-180.0};
  double yaw_max_deg_{180.0};
  double pitch_min_deg_{-180.0};
  double pitch_max_deg_{180.0};
};

}  // namespace laser_aim_controller