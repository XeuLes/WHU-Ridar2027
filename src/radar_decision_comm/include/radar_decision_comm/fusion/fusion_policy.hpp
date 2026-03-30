#pragma once

namespace radar_decision_comm
{

class FusionPolicy
{
public:
  FusionPolicy() = default;
  ~FusionPolicy() = default;

  void tick();
};

}  // namespace radar_decision_comm
