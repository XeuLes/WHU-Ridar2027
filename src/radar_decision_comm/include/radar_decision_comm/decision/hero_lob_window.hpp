#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/time.hpp>

#include "radar_decision_comm/decision/hero_lob_types.hpp"

namespace radar_decision_comm
{

class HeroLobWindow
{
public:
  HeroLobWindow() = default;

  void set_window_seconds(double window_seconds);
  void set_min_samples(std::size_t min_samples);
  void set_max_avg_displacement_m(double max_avg_displacement_m);

  void reset();
  void ingest(const geometry_msgs::msg::Point & field_position, const rclcpp::Time & stamp);
  HeroLobWindowOutput evaluate() const;

private:
  struct Sample
  {
    rclcpp::Time stamp;
    geometry_msgs::msg::Point field_position;
  };

  void prune_old_samples(const rclcpp::Time & newest_stamp);

  double window_seconds_{2.0};
  std::size_t min_samples_{10};
  double max_avg_displacement_m_{0.6};
  int64_t last_sample_ns_{0};
  std::deque<Sample> samples_;
};

}  // namespace radar_decision_comm
