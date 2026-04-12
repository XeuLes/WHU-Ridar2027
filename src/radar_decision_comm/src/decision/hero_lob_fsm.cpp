#include "radar_decision_comm/decision/hero_lob_fsm.hpp"

#include <algorithm>

namespace radar_decision_comm
{

void HeroLobFSM::set_clear_repeat(int clear_repeat)
{
  clear_repeat_ = std::max(clear_repeat, 0);
}

void HeroLobFSM::reset()
{
  state_ = HeroLobState::WAIT_TRACK;
  clear_remaining_ = 0;
  clear_reason_.clear();
}

HeroLobFSMOutput HeroLobFSM::make_output(
  HeroLobState state,
  bool action_needed,
  bool publish_active,
  bool clear_mode,
  const std::string & reason) const
{
  HeroLobFSMOutput out;
  out.state = state;
  out.action_needed = action_needed;
  out.publish_active = publish_active;
  out.clear_mode = clear_mode;
  out.reason = reason;
  return out;
}

void HeroLobFSM::enter_clearing(const std::string & reason)
{
  state_ = HeroLobState::CLEARING;
  clear_remaining_ = clear_repeat_;
  clear_reason_ = reason;
}

void HeroLobFSM::note_action_sent()
{
  if (state_ == HeroLobState::CLEARING && clear_remaining_ > 0) {
    --clear_remaining_;
  }
}

HeroLobFSMOutput HeroLobFSM::tick(const HeroLobFSMInput & input)
{
  auto evaluate_base_state = [&]() -> HeroLobFSMOutput {
      if (!input.enabled) {
        state_ = HeroLobState::DISABLED;
        return make_output(state_, false, false, false, "disabled");
      }
      if (!input.hero_found || !input.our_track_fresh) {
        state_ = HeroLobState::WAIT_TRACK;
        return make_output(state_, false, false, false, "wait_track");
      }
      if (!input.window_ready) {
        state_ = HeroLobState::ACCUMULATING;
        return make_output(state_, false, false, false, "accumulating");
      }
      if (!input.pose_stable) {
        state_ = HeroLobState::WAIT_STABLE;
        return make_output(state_, false, false, false, "wait_stable");
      }
      if (input.enemy_nearby_3m) {
        state_ = HeroLobState::THREAT_BLOCKED;
        return make_output(state_, false, false, false, "enemy_nearby_3m");
      }
      state_ = HeroLobState::SOLVING;
      if (!input.solution_valid) {
        return make_output(state_, false, false, false, "solving_no_solution");
      }
      if (!input.quiet_enter_satisfied) {
        return make_output(state_, false, false, false, "enter_quiet");
      }
      state_ = HeroLobState::ACTIVE;
      return make_output(state_, true, true, false, "active");
    };

  if (state_ == HeroLobState::ACTIVE) {
    if (
      !input.enabled || !input.hero_found || !input.our_track_fresh || !input.window_ready ||
      !input.pose_stable || input.enemy_nearby_3m || !input.solution_valid)
    {
      enter_clearing(
        !input.enabled ? "disabled" :
        (!input.hero_found || !input.our_track_fresh) ? "track_lost" :
        !input.window_ready ? "window_not_ready" :
        !input.pose_stable ? "unstable" :
        input.enemy_nearby_3m ? "enemy_nearby_3m" :
        "solution_invalid");
    } else {
      return make_output(HeroLobState::ACTIVE, true, true, false, "active");
    }
  }

  if (state_ == HeroLobState::CLEARING) {
    if (clear_remaining_ > 0) {
      return make_output(HeroLobState::CLEARING, true, false, true, clear_reason_);
    }
    clear_reason_.clear();
    return evaluate_base_state();
  }

  return evaluate_base_state();
}

}  // namespace radar_decision_comm
