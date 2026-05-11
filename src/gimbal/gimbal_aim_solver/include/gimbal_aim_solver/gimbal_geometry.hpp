#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace gimbal_aim_solver
{

class GimbalGeometry
{
public:
  GimbalGeometry() = default;

  // 云台原点在相机坐标系中的坐标
  void setCameraToGimbalTranslation(const Eigen::Vector3d& t_camera_to_gimbal);

  // 相机前向轴相对云台静止前向轴的安装俯仰角，向上为正，单位 deg
  void setCameraPitchRelativeToGimbal(double pitch_deg);

  // camera -> gimbal
  Eigen::Vector3d cameraToGimbal(const Eigen::Vector3d& p_camera) const;

private:
  Eigen::Vector3d t_camera_to_gimbal_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_gimbal_from_camera_{Eigen::Matrix3d::Identity()};
};

}  // namespace gimbal_aim_solver
