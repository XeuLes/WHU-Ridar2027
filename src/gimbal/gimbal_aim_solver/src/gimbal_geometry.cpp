#include "gimbal_aim_solver/gimbal_geometry.hpp"

#include <cmath>
#include <Eigen/Geometry>

namespace gimbal_aim_solver
{

void GimbalGeometry::setCameraToGimbalTranslation(const Eigen::Vector3d& t_camera_to_gimbal)
{
  t_camera_to_gimbal_ = t_camera_to_gimbal;
}

void GimbalGeometry::setCameraPitchRelativeToGimbal(double pitch_deg)
{
  constexpr double kPi = 3.14159265358979323846;
  const double pitch_rad = pitch_deg * kPi / 180.0;
  const Eigen::AngleAxisd camera_from_gimbal_pitch(pitch_rad, Eigen::Vector3d::UnitX());
  R_gimbal_from_camera_ = camera_from_gimbal_pitch.inverse().toRotationMatrix();
}

Eigen::Vector3d GimbalGeometry::cameraToGimbal(const Eigen::Vector3d& p_camera) const
{
  // 先平移到云台原点，再按安装俯仰把相机系点转到静止云台系。
  return R_gimbal_from_camera_ * (p_camera - t_camera_to_gimbal_);
}

}  // namespace gimbal_aim_solver
