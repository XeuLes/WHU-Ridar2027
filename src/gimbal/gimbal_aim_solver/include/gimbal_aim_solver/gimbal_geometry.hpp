#pragma once

#include <Eigen/Core>

namespace gimbal_aim_solver
{

class GimbalGeometry
{
public:
  GimbalGeometry() = default;

  // 云台原点在相机坐标系中的坐标
  void setCameraToGimbalTranslation(const Eigen::Vector3d& t_camera_to_gimbal);

  // camera -> gimbal
  Eigen::Vector3d cameraToGimbal(const Eigen::Vector3d& p_camera) const;

private:
  Eigen::Vector3d t_camera_to_gimbal_{Eigen::Vector3d::Zero()};
};

}  // namespace gimbal_aim_solver