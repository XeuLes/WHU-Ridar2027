#pragma once

#include <Eigen/Core>

namespace laser_aim_controller
{

class LaserGeometry
{
public:
  LaserGeometry() = default;

  void setCameraOffset(const Eigen::Vector3d& camera_offset_xyz);
  void setLaserOffset(const Eigen::Vector3d& laser_offset_xyz);

  Eigen::Vector3d targetInRealtimeFrame(const Eigen::Vector3d& target_in_camera_frame) const;
  Eigen::Vector3d targetFromLaser(const Eigen::Vector3d& target_in_camera_frame) const;

private:
  Eigen::Vector3d camera_offset_xyz_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d laser_offset_xyz_{Eigen::Vector3d::Zero()};
};

}  // namespace laser_aim_controller