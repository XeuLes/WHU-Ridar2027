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

  static std::vector<uint8_t> build_hero_lob_assist_payload(
    uint8_t flags,
    double yaw_field_cw_deg,
    double pitch_deg,
    uint16_t age_ms,
    double impact_error_m,
    double splash_radius_m,
    uint8_t state,
    uint8_t sequence_tag);

private:
  static void append_u16_le(std::vector<uint8_t> & out, uint16_t value);
  static void append_u32_le(std::vector<uint8_t> & out, uint32_t value);
  static void append_i16_le(std::vector<uint8_t> & out, int16_t value);
  static void append_float_le(std::vector<uint8_t> & out, float value);
};

}  // namespace radar_decision_comm
