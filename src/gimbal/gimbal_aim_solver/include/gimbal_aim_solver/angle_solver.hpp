#pragma once

#include <Eigen/Core>
#include "gimbal_aim_solver/types.hpp"

namespace gimbal_aim_solver
{

class AngleSolver
{
public:
  AngleSolver() = default;

  void setOffsets(double yaw_offset_deg, double pitch_offset_deg);
  void setLimits(double yaw_min_deg, double yaw_max_deg,
                 double pitch_min_deg, double pitch_max_deg);

  // 输入为云台坐标系下的目标点
  // 坐标系默认与 OpenCV camera 坐标系平行:
  // x 右, y 下, z 前
  TargetAngles solve(const Eigen::Vector3d& p_gimbal) const;

private:
  double yaw_offset_deg_{0.0};
  double pitch_offset_deg_{0.0};

  double yaw_min_deg_{-180.0};
  double yaw_max_deg_{180.0};
  double pitch_min_deg_{-45.0};
  double pitch_max_deg_{45.0};
};

}  // namespace gimbal_aim_solver