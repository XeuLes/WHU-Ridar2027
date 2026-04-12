#include "radar_decision_comm/decision/hero_lob_ballistics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace radar_decision_comm
{

namespace
{

constexpr double kGravityMps2 = 9.81;
constexpr double kPi = 3.14159265358979323846;

struct Vec3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

double norm(const Vec3 & v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

}  // namespace

void HeroLobBallistics::set_config(const HeroLobBallisticsConfig & config)
{
  config_ = config;
  config_.integrate_dt_s = std::max(config_.integrate_dt_s, 1e-4);
  config_.max_flight_time_s = std::max(config_.max_flight_time_s, config_.integrate_dt_s);
  config_.yaw_refine_range_deg = std::max(config_.yaw_refine_range_deg, 0.0);
  config_.yaw_refine_step_deg = std::max(config_.yaw_refine_step_deg, 1e-3);
  config_.pitch_search_step_deg = std::max(config_.pitch_search_step_deg, 1e-3);
  if (config_.pitch_search_max_deg < config_.pitch_search_min_deg) {
    std::swap(config_.pitch_search_min_deg, config_.pitch_search_max_deg);
  }
}

const HeroLobBallisticsConfig & HeroLobBallistics::config() const
{
  return config_;
}

double HeroLobBallistics::deg_to_rad(double angle_deg)
{
  return angle_deg * kPi / 180.0;
}

double HeroLobBallistics::normalize_deg(double angle_deg)
{
  while (angle_deg > 180.0) {
    angle_deg -= 360.0;
  }
  while (angle_deg <= -180.0) {
    angle_deg += 360.0;
  }
  return angle_deg;
}

geometry_msgs::msg::Point HeroLobBallistics::make_point(double x, double y, double z)
{
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

double HeroLobBallistics::point_distance(
  const geometry_msgs::msg::Point & lhs,
  const geometry_msgs::msg::Point & rhs)
{
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;
  const double dz = lhs.z - rhs.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

HeroLobBallistics::SimResult HeroLobBallistics::simulate(
  double actual_yaw_field_cw_deg,
  double actual_pitch_deg,
  const geometry_msgs::msg::Point & shooter_point,
  const geometry_msgs::msg::Point & target_point) const
{
  SimResult out;
  if (config_.muzzle_speed_mps <= 1e-3 || config_.integrate_dt_s <= 1e-6 || config_.max_flight_time_s <= 0.0) {
    return out;
  }

  const double yaw_rad = deg_to_rad(actual_yaw_field_cw_deg);
  const double pitch_rad = deg_to_rad(actual_pitch_deg);
  const double planar_speed = config_.muzzle_speed_mps * std::cos(pitch_rad);

  Vec3 pos{shooter_point.x, shooter_point.y, shooter_point.z};
  Vec3 vel{
    planar_speed * std::cos(yaw_rad),
    -planar_speed * std::sin(yaw_rad),
    config_.muzzle_speed_mps * std::sin(pitch_rad)};

  out.closest_point = shooter_point;
  out.min_distance = point_distance(shooter_point, target_point);
  out.time_at_closest = 0.0;

  for (double t = 0.0; t <= config_.max_flight_time_s; t += config_.integrate_dt_s) {
    const geometry_msgs::msg::Point current_point = make_point(pos.x, pos.y, pos.z);
    const double dist = point_distance(current_point, target_point);
    if (dist < out.min_distance) {
      out.min_distance = dist;
      out.closest_point = current_point;
      out.time_at_closest = t;
      out.valid = true;
    }

    const double speed = norm(vel);
    const Vec3 acc{
      -config_.drag_k * speed * vel.x,
      -config_.drag_k * speed * vel.y,
      -kGravityMps2 - config_.drag_k * speed * vel.z};

    vel.x += acc.x * config_.integrate_dt_s;
    vel.y += acc.y * config_.integrate_dt_s;
    vel.z += acc.z * config_.integrate_dt_s;

    pos.x += vel.x * config_.integrate_dt_s;
    pos.y += vel.y * config_.integrate_dt_s;
    pos.z += vel.z * config_.integrate_dt_s;

    if (pos.z < std::min(shooter_point.z, target_point.z) - 5.0) {
      break;
    }
  }

  return out;
}

HeroLobSolution HeroLobBallistics::solve(
  const geometry_msgs::msg::Point & shooter_point,
  const geometry_msgs::msg::Point & target_point) const
{
  HeroLobSolution out;
  out.shooter_point = shooter_point;
  out.target_point = target_point;

  if (config_.muzzle_speed_mps <= 1e-3) {
    out.reason = "invalid_muzzle_speed";
    return out;
  }

  const double dx = target_point.x - shooter_point.x;
  const double dy = target_point.y - shooter_point.y;
  const double planar_dist = std::hypot(dx, dy);
  if (planar_dist <= 1e-3) {
    out.reason = "target_too_close";
    return out;
  }

  const double nominal_yaw_field_cw_deg = normalize_deg(-std::atan2(dy, dx) * 180.0 / kPi);
  const double nominal_command_yaw_field_cw_deg =
    normalize_deg(nominal_yaw_field_cw_deg - config_.yaw_bias_deg);

  struct Candidate
  {
    bool valid{false};
    double command_yaw_deg{0.0};
    double command_pitch_deg{0.0};
    double objective{std::numeric_limits<double>::infinity()};
    double center_error_m{std::numeric_limits<double>::infinity()};
    double splash_radius_m{0.0};
    double flight_time_s{0.0};
    geometry_msgs::msg::Point impact_center_point;
  } best_candidate;

  const auto evaluate_candidate =
    [&](double command_yaw_deg, double command_pitch_deg) -> Candidate
    {
      Candidate candidate;
      const double actual_yaw_deg = normalize_deg(command_yaw_deg + config_.yaw_bias_deg);
      const double actual_pitch_deg = command_pitch_deg + config_.pitch_bias_deg;

      const auto nominal = simulate(actual_yaw_deg, actual_pitch_deg, shooter_point, target_point);
      if (!nominal.valid) {
        return candidate;
      }

      std::vector<geometry_msgs::msg::Point> points;
      points.reserve(5);
      points.push_back(nominal.closest_point);

      const std::array<std::pair<double, double>, 4> perturbations{{
        {config_.yaw_sigma_deg, 0.0},
        {-config_.yaw_sigma_deg, 0.0},
        {0.0, config_.pitch_sigma_deg},
        {0.0, -config_.pitch_sigma_deg},
      }};

      for (const auto & perturb : perturbations) {
        const auto perturbed = simulate(
          actual_yaw_deg + perturb.first,
          actual_pitch_deg + perturb.second,
          shooter_point,
          target_point);
        if (perturbed.valid) {
          points.push_back(perturbed.closest_point);
        }
      }

      if (points.empty()) {
        return candidate;
      }

      geometry_msgs::msg::Point impact_center;
      for (const auto & point : points) {
        impact_center.x += point.x;
        impact_center.y += point.y;
        impact_center.z += point.z;
      }
      const double inv_n = 1.0 / static_cast<double>(points.size());
      impact_center.x *= inv_n;
      impact_center.y *= inv_n;
      impact_center.z *= inv_n;

      double splash_radius_m = 0.0;
      for (const auto & point : points) {
        splash_radius_m += point_distance(point, impact_center);
      }
      splash_radius_m *= inv_n;

      const double center_error_m = point_distance(impact_center, target_point);
      candidate.valid = true;
      candidate.command_yaw_deg = normalize_deg(command_yaw_deg);
      candidate.command_pitch_deg = command_pitch_deg;
      candidate.center_error_m = center_error_m;
      candidate.splash_radius_m = splash_radius_m;
      candidate.flight_time_s = nominal.time_at_closest;
      candidate.impact_center_point = impact_center;
      candidate.objective = center_error_m + 0.25 * splash_radius_m;
      return candidate;
    };

  for (double yaw_offset_deg = -config_.yaw_refine_range_deg;
    yaw_offset_deg <= config_.yaw_refine_range_deg + 1e-6;
    yaw_offset_deg += config_.yaw_refine_step_deg)
  {
    const double command_yaw_deg =
      normalize_deg(nominal_command_yaw_field_cw_deg + yaw_offset_deg);
    for (double command_pitch_deg = config_.pitch_search_min_deg;
      command_pitch_deg <= config_.pitch_search_max_deg + 1e-6;
      command_pitch_deg += config_.pitch_search_step_deg)
    {
      const auto candidate = evaluate_candidate(command_yaw_deg, command_pitch_deg);
      if (!candidate.valid) {
        continue;
      }
      if (candidate.objective < best_candidate.objective) {
        best_candidate = candidate;
      }
    }
  }

  if (!best_candidate.valid || !std::isfinite(best_candidate.objective)) {
    out.reason = "no_valid_trajectory";
    return out;
  }

  out.valid = true;
  out.yaw_field_cw_deg = best_candidate.command_yaw_deg;
  out.pitch_deg = best_candidate.command_pitch_deg;
  out.flight_time_s = best_candidate.flight_time_s;
  out.impact_error_m = best_candidate.center_error_m;
  out.splash_radius_m = best_candidate.splash_radius_m;
  out.impact_center_point = best_candidate.impact_center_point;
  out.reason = "ok";
  return out;
}

}  // namespace radar_decision_comm
