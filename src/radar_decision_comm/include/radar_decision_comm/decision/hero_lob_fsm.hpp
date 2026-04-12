#pragma once

#include <string>

#include "radar_decision_comm/decision/hero_lob_types.hpp"

namespace radar_decision_comm
{

class HeroLobFSM
{
public:
  HeroLobFSM() = default;

  void set_clear_repeat(int clear_repeat);
  void reset();
  HeroLobFSMOutput tick(const HeroLobFSMInput & input);
  void note_action_sent();

private:
  HeroLobFSMOutput make_output(
    HeroLobState state,
    bool action_needed,
    bool publish_active,
    bool clear_mode,
    const std::string & reason) const;
  void enter_clearing(const std::string & reason);

  HeroLobState state_{HeroLobState::WAIT_TRACK};
  int clear_repeat_{3};
  int clear_remaining_{0};
  std::string clear_reason_;
};

}  // namespace radar_decision_comm
