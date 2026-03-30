#pragma once

namespace radar_decision_comm
{

class DecisionEngine
{
public:
  DecisionEngine() = default;
  ~DecisionEngine() = default;

  void tick();
};

}  // namespace radar_decision_comm
