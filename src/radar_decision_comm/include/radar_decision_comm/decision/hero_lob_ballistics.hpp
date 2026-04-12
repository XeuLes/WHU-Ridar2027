#pragma once

#include <geometry_msgs/msg/point.hpp>

#include "radar_decision_comm/decision/hero_lob_types.hpp"

namespace radar_decision_comm
{

class HeroLobBallistics
{
public:
  HeroLobBallistics() = default;

  void set_config(const HeroLobBallisticsConfig & config);
  const HeroLobBallisticsConfig & config() const;

  HeroLobSolution solve(
    const geometry_msgs::msg::Point & shooter_point,
    const geometry_msgs::msg::Point & target_point) const;

private:
  struct SimResult
  {
    bool valid{false};
    geometry_msgs::msg::Point closest_point;
    double min_distance{0.0};
    double time_at_closest{0.0};
  };

  SimResult simulate(
    double actual_yaw_field_cw_deg,
    double actual_pitch_deg,
    const geometry_msgs::msg::Point & shooter_point,
    const geometry_msgs::msg::Point & target_point) const;

  static double normalize_deg(double angle_deg);
  static double deg_to_rad(double angle_deg);
  static geometry_msgs::msg::Point make_point(double x, double y, double z);
  static double point_distance(
    const geometry_msgs::msg::Point & lhs,
    const geometry_msgs::msg::Point & rhs);

  HeroLobBallisticsConfig config_;
};

}  // namespace radar_decision_comm
