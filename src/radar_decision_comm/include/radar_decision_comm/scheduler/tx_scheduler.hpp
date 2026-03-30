#pragma once

namespace radar_decision_comm
{

class TxScheduler
{
public:
  TxScheduler() = default;
  ~TxScheduler() = default;

  void tick();
};

}  // namespace radar_decision_comm
