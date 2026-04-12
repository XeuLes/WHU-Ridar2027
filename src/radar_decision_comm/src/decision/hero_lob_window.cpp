#include "radar_decision_comm/decision/hero_lob_window.hpp"

#include <algorithm>
#include <cmath>

namespace radar_decision_comm
{

void HeroLobWindow::set_window_seconds(double window_seconds)
{
  window_seconds_ = std::max(window_seconds, 0.1);
}

void HeroLobWindow::set_min_samples(std::size_t min_samples)
{
  min_samples_ = std::max<std::size_t>(min_samples, 1);
}

void HeroLobWindow::set_max_avg_displacement_m(double max_avg_displacement_m)
{
  max_avg_displacement_m_ = std::max(max_avg_displacement_m, 0.0);
}

void HeroLobWindow::reset()
{
  samples_.clear();
  last_sample_ns_ = 0;
}

void HeroLobWindow::prune_old_samples(const rclcpp::Time & newest_stamp)
{
  const auto window_ns = static_cast<int64_t>(window_seconds_ * 1e9);
  while (!samples_.empty() &&
    (newest_stamp.nanoseconds() - samples_.front().stamp.nanoseconds()) > window_ns)
  {
    samples_.pop_front();
  }
}

void HeroLobWindow::ingest(const geometry_msgs::msg::Point & field_position, const rclcpp::Time & stamp)
{
  if (stamp.nanoseconds() <= 0 || stamp.nanoseconds() <= last_sample_ns_) {
    return;
  }

  samples_.push_back(Sample{stamp, field_position});
  last_sample_ns_ = stamp.nanoseconds();
  prune_old_samples(stamp);
}

HeroLobWindowOutput HeroLobWindow::evaluate() const
{
  HeroLobWindowOutput out;
  out.sample_count = samples_.size();
  if (samples_.empty()) {
    return out;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  for (const auto & sample : samples_) {
    sum_x += sample.field_position.x;
    sum_y += sample.field_position.y;
    sum_z += sample.field_position.z;
  }

  const double inv_n = 1.0 / static_cast<double>(samples_.size());
  out.mean_position.x = sum_x * inv_n;
  out.mean_position.y = sum_y * inv_n;
  out.mean_position.z = sum_z * inv_n;

  double disp_sum = 0.0;
  for (const auto & sample : samples_) {
    const double dx = sample.field_position.x - out.mean_position.x;
    const double dy = sample.field_position.y - out.mean_position.y;
    disp_sum += std::hypot(dx, dy);
  }
  out.avg_displacement_m = disp_sum * inv_n;

  out.duration_s = static_cast<double>(
    samples_.back().stamp.nanoseconds() - samples_.front().stamp.nanoseconds()) / 1e9;
  out.ready = (out.duration_s >= window_seconds_) && (out.sample_count >= min_samples_);
  out.stable = out.ready && (out.avg_displacement_m <= max_avg_displacement_m_);
  return out;
}

}  // namespace radar_decision_comm
