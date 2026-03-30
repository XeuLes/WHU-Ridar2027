#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace radar_decision_comm
{

struct EnemyFieldStateSlot
{
  uint8_t valid{0};
  float x{0.0F};
  float y{0.0F};
};

class InteractionPayloadBuilder
{
public:
  static std::vector<uint8_t> build_sentry_alert_angle_payload(
    uint16_t robot_id,
    float distance,
    uint16_t quadrant);

  static std::vector<uint8_t> build_sentry_field_state_payload(
    const std::array<EnemyFieldStateSlot, 6> & slots);

  static std::vector<uint8_t> build_hero_alert_flag_payload(bool is_alert);

  static std::vector<uint8_t> build_double_effect_forward_payload(uint8_t times);

private:
  static void append_u16_le(std::vector<uint8_t> & out, uint16_t value);
  static void append_float_le(std::vector<uint8_t> & out, float value);
};

}  // namespace radar_decision_comm
