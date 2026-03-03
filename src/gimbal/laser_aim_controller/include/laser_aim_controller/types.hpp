#pragma once

namespace laser_aim_controller
{

struct GimbalState
{
  bool valid{false};
  double yaw_deg{0.0};
  double pitch_deg{0.0};
};

struct CommandAngles
{
  bool valid{false};
  double yaw_deg{0.0};
  double pitch_deg{0.0};
};

}  // namespace laser_aim_controller