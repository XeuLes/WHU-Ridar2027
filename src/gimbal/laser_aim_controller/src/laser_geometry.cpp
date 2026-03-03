#include "laser_aim_controller/laser_geometry.hpp"

namespace laser_aim_controller
{

void LaserGeometry::setCameraOffset(const Eigen::Vector3d& camera_offset_xyz)
{
  camera_offset_xyz_ = camera_offset_xyz;
}

void LaserGeometry::setLaserOffset(const Eigen::Vector3d& laser_offset_xyz)
{
  laser_offset_xyz_ = laser_offset_xyz;
}

Eigen::Vector3d LaserGeometry::targetInRealtimeFrame(
  const Eigen::Vector3d& target_in_camera_frame) const
{
  return camera_offset_xyz_ + target_in_camera_frame;
}

Eigen::Vector3d LaserGeometry::targetFromLaser(
  const Eigen::Vector3d& target_in_camera_frame) const
{
  const Eigen::Vector3d target_rt = targetInRealtimeFrame(target_in_camera_frame);
  return target_rt - laser_offset_xyz_;
}

}  // namespace laser_aim_controller