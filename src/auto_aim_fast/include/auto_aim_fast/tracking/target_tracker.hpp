#pragma once
#include "auto_aim_fast/pose/pnp_solver.hpp"

#include <cstdint>

namespace auto_aim_fast::tracking
{

enum class TrackState : int
{
  SEARCH = 0,
  TRACK  = 1,
  LOST   = 2
};

struct TrackerParams
{
  int confirm_track_frames{3};
  int lost_to_lost_frames{5};
  int lost_to_search_frames{20};

  float min_conf_to_track{0.5f};

  // ROI policy (normalized by a "base size" in pixels; later we will tie to target size)
  int roi_base_size{200};         // base square ROI size
  float roi_expand_lost{1.3f};    // LOST expansion factor per step
  int roi_max_size{800};          // clamp
};

struct TrackDebug
{
  int good_count{0};
  int miss_count{0};
};

struct RoiSuggest
{
  bool valid{false};
  int x{0}, y{0}, w{0}, h{0};
};

struct TrackOutput
{
  TrackState state{TrackState::SEARCH};
  bool has_target{false};

  float yaw{0};
  float pitch{0};
  float dist{0};
  float confidence{0};

  TrackDebug debug;
  RoiSuggest roi;
};

class TargetTracker
{
public:
  TargetTracker() = default;

  void setParams(const TrackerParams & p) { params_ = p; }
  const TrackerParams & params() const { return params_; }

  TrackOutput update(const auto_aim_fast::pose::Pose3D & meas, int64_t t_ns);
  void reset();

private:
  TrackerParams params_{};
  TrackState state_{TrackState::SEARCH};

  int good_count_{0};
  int miss_count_{0};

  float last_dist_{0.0f};
  float last_conf_{0.0f};

  // ROI anchor (image pixel coordinates) - will be filled by detector in next stage later
  // For now we only keep a placeholder center.
  int last_cx_{-1};
  int last_cy_{-1};
  int lost_expand_step_{0};
};

}  // namespace auto_aim_fast::tracking
