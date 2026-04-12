#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <geometry_msgs/msg/point.hpp>

namespace radar_decision_comm
{

enum class HeroLobState : uint8_t
{
  DISABLED = 0,
  WAIT_TRACK = 1,
  ACCUMULATING = 2,
  WAIT_STABLE = 3,
  THREAT_BLOCKED = 4,
  SOLVING = 5,
  ACTIVE = 6,
  CLEARING = 7
};

inline const char * hero_lob_state_name(HeroLobState state)
{
  switch (state) {
    case HeroLobState::DISABLED: return "disabled";
    case HeroLobState::WAIT_TRACK: return "wait_track";
    case HeroLobState::ACCUMULATING: return "accumulating";
    case HeroLobState::WAIT_STABLE: return "wait_stable";
    case HeroLobState::THREAT_BLOCKED: return "threat_blocked";
    case HeroLobState::SOLVING: return "solving";
    case HeroLobState::ACTIVE: return "active";
    case HeroLobState::CLEARING: return "clearing";
    default: return "unknown";
  }
}

struct HeroLobWindowOutput
{
  bool ready{false};
  bool stable{false};
  std::size_t sample_count{0};
  double duration_s{0.0};
  double avg_displacement_m{std::numeric_limits<double>::infinity()};
  geometry_msgs::msg::Point mean_position;
};

struct HeroLobBallisticsConfig
{
  double muzzle_speed_mps{0.0};
  double drag_k{0.0};
  double yaw_sigma_deg{0.0};
  double pitch_sigma_deg{0.0};
  double yaw_bias_deg{0.0};
  double pitch_bias_deg{0.0};
  double integrate_dt_s{0.005};
  double max_flight_time_s{5.0};
  double yaw_refine_range_deg{1.0};
  double yaw_refine_step_deg{0.1};
  double pitch_search_min_deg{15.0};
  double pitch_search_max_deg{75.0};
  double pitch_search_step_deg{0.2};
};

struct HeroLobSolution
{
  bool valid{false};
  double yaw_field_cw_deg{0.0};
  double pitch_deg{0.0};
  double flight_time_s{0.0};
  double impact_error_m{std::numeric_limits<double>::infinity()};
  double splash_radius_m{0.0};
  geometry_msgs::msg::Point shooter_point;
  geometry_msgs::msg::Point target_point;
  geometry_msgs::msg::Point impact_center_point;
  std::string reason;
};

struct HeroLobFSMInput
{
  bool enabled{false};
  bool hero_found{false};
  bool our_track_fresh{false};
  bool window_ready{false};
  bool pose_stable{false};
  bool enemy_nearby_3m{false};
  bool quiet_enter_satisfied{false};
  bool solution_valid{false};
};

struct HeroLobFSMOutput
{
  HeroLobState state{HeroLobState::WAIT_TRACK};
  bool action_needed{false};
  bool publish_active{false};
  bool clear_mode{false};
  std::string reason;
};

}  // namespace radar_decision_comm
