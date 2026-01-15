#pragma once
namespace auto_aim_fast::prediction
{
class Ballistic
{
public:
  Ballistic() = default;
  float pitchCompensation(float dist_m) const
  {
    (void)dist_m;
    return 0.0f;
  }
};
}  // namespace auto_aim_fast::prediction
