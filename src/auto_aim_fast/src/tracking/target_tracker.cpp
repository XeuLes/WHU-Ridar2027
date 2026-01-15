#include "auto_aim_fast/tracking/target_tracker.hpp"

namespace auto_aim_fast::tracking
{

static bool isGoodMeas(const auto_aim_fast::pose::Pose3D & m, float min_conf)
{
  return m.ok && (m.score >= min_conf);
}

static RoiSuggest makeRoi(int cx, int cy, int size)
{
  RoiSuggest r;
  if (cx < 0 || cy < 0 || size <= 0) return r;
  r.valid = true;
  r.w = size;
  r.h = size;
  r.x = cx - size / 2;
  r.y = cy - size / 2;
  return r;
}

TrackOutput TargetTracker::update(const auto_aim_fast::pose::Pose3D & meas, int64_t t_ns)
{
  (void)t_ns;

  const bool good = isGoodMeas(meas, params_.min_conf_to_track);

  // If we have 2D center, keep updating anchor (even during LOST)
  if (meas.has_img_center) {
    last_cx_ = static_cast<int>(meas.img_cx);
    last_cy_ = static_cast<int>(meas.img_cy);
  }

  if (good) {
    good_count_++;
    miss_count_ = 0;
    lost_expand_step_ = 0;
    last_dist_ = meas.z;
    last_conf_ = meas.score;
  } else {
    miss_count_++;
    good_count_ = 0;
    if (state_ == TrackState::LOST) {
      lost_expand_step_++;
    }
  }

  switch (state_) {
    case TrackState::SEARCH:
      if (good && good_count_ >= params_.confirm_track_frames) {
        state_ = TrackState::TRACK;
      }
      break;

    case TrackState::TRACK:
      if (!good && miss_count_ >= params_.lost_to_lost_frames) {
        state_ = TrackState::LOST;
        lost_expand_step_ = 0;
      }
      break;

    case TrackState::LOST:
      if (good && good_count_ >= params_.confirm_track_frames) {
        state_ = TrackState::TRACK;
        lost_expand_step_ = 0;
      } else if (!good && miss_count_ >= params_.lost_to_search_frames) {
        state_ = TrackState::SEARCH;
        lost_expand_step_ = 0;
      }
      break;

    default:
      state_ = TrackState::SEARCH;
      lost_expand_step_ = 0;
      break;
  }

  TrackOutput out;
  out.state = state_;
  out.has_target = (state_ == TrackState::TRACK) && good;
  out.dist = good ? meas.z : last_dist_;
  out.confidence = good ? meas.score : last_conf_;
  out.yaw = 0.0f;
  out.pitch = 0.0f;

  out.debug.good_count = good_count_;
  out.debug.miss_count = miss_count_;

  if (state_ == TrackState::SEARCH) {
    out.roi = RoiSuggest{};
  } else {
    int size = params_.roi_base_size;
    if (state_ == TrackState::LOST) {
      float s = 1.0f;
      for (int i = 0; i < lost_expand_step_; ++i) s *= params_.roi_expand_lost;
      size = static_cast<int>(size * s);
    }
    if (size > params_.roi_max_size) size = params_.roi_max_size;
    out.roi = makeRoi(last_cx_, last_cy_, size);
  }

  return out;
}

void TargetTracker::reset()
{
  state_ = TrackState::SEARCH;
  good_count_ = 0;
  miss_count_ = 0;
  lost_expand_step_ = 0;
  last_dist_ = 0.0f;
  last_conf_ = 0.0f;
  last_cx_ = -1;
  last_cy_ = -1;
}

}  // namespace auto_aim_fast::tracking
