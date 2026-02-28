#include "gimbal_aim_solver/gimbal_geometry.hpp"

namespace gimbal_aim_solver
{

void GimbalGeometry::setCameraToGimbalTranslation(const Eigen::Vector3d& t_camera_to_gimbal)
{
  t_camera_to_gimbal_ = t_camera_to_gimbal;
}

Eigen::Vector3d GimbalGeometry::cameraToGimbal(const Eigen::Vector3d& p_camera) const
{
  // 若云台坐标系与相机坐标系轴平行，仅原点不同
  // 则点在云台系下坐标 = 点在相机系下坐标 - 云台原点在相机系下坐标
  return p_camera - t_camera_to_gimbal_;
}

}  // namespace gimbal_aim_solver