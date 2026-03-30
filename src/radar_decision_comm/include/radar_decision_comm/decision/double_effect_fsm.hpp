#pragma once

#include <cstdint>
#include <string>

namespace radar_decision_comm
{

enum class DoubleEffectFSMState : uint8_t
{
  IDLE = 0,
  READY = 1,
  DECIDING = 2,
  ACTIVE = 3,
  BLOCKED = 4
};

struct DoubleEffectFSMInput
{
  bool referee_fresh{false};
  bool wireless_fresh{false};
  bool double_effect_available{false};
  bool double_effect_active{false};
  bool hero_marked{false};
  uint16_t hero_hp{0};
  uint16_t hero_ammo{0};
  bool has_buffs{false};
  bool has_macro{false};
  uint8_t available_times{0};
  int low_hp_threshold{120};
  int low_ammo_threshold{10};
};

struct DoubleEffectFSMOutput
{
  DoubleEffectFSMState state{DoubleEffectFSMState::IDLE};
  bool should_forward{false};
  bool should_send_radar_command{false};
  uint8_t forward_times{0};
  uint8_t radar_command_times{0};
  std::string reason;
};

class DoubleEffectFSM
{
public:
  DoubleEffectFSM() = default;
  ~DoubleEffectFSM() = default;

  void reset();
  DoubleEffectFSMOutput tick(const DoubleEffectFSMInput & in);
  DoubleEffectFSMState state() const;

private:
  DoubleEffectFSMState state_{DoubleEffectFSMState::IDLE};
};

}  // namespace radar_decision_comm
