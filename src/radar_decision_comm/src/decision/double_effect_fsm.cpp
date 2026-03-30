#include "radar_decision_comm/decision/double_effect_fsm.hpp"

namespace radar_decision_comm
{

void DoubleEffectFSM::reset()
{
  state_ = DoubleEffectFSMState::IDLE;
}

DoubleEffectFSMOutput DoubleEffectFSM::tick(const DoubleEffectFSMInput & in)
{
  DoubleEffectFSMOutput out;

  if (!in.referee_fresh) {
    state_ = DoubleEffectFSMState::BLOCKED;
    out.state = state_;
    out.reason = "referee_stale";
    return out;
  }

  if (in.double_effect_active) {
    state_ = DoubleEffectFSMState::ACTIVE;
    out.state = state_;
    out.reason = "already_active";
    return out;
  }

  if (!in.double_effect_available || in.available_times == 0) {
    state_ = DoubleEffectFSMState::IDLE;
    out.state = state_;
    out.reason = "no_available_times";
    return out;
  }

  const bool urgent =
    in.hero_marked ||
    (in.wireless_fresh &&
      ((in.hero_hp > 0 && in.hero_hp <= static_cast<uint16_t>(in.low_hp_threshold)) ||
       (in.hero_ammo > 0 && in.hero_ammo <= static_cast<uint16_t>(in.low_ammo_threshold)) ||
       in.has_buffs ||
       in.has_macro));

  if (urgent) {
    state_ = DoubleEffectFSMState::DECIDING;
    out.state = state_;
    out.should_forward = true;
    out.should_send_radar_command = true;
    out.forward_times = in.available_times;
    out.radar_command_times = in.available_times;
    out.reason = "urgent";
    return out;
  }

  state_ = DoubleEffectFSMState::READY;
  out.state = state_;
  out.should_forward = true;
  out.forward_times = in.available_times;
  out.reason = "ready_no_urgent";
  return out;
}

DoubleEffectFSMState DoubleEffectFSM::state() const
{
  return state_;
}

}  // namespace radar_decision_comm
