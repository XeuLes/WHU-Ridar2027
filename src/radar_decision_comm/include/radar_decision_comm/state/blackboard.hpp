#pragma once

namespace radar_decision_comm
{

class Blackboard
{
public:
  Blackboard() = default;
  ~Blackboard() = default;

  void reset();
};

}  // namespace radar_decision_comm
