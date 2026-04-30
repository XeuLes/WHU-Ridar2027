#include "auto_aim_fast/pipeline/pipeline.hpp"

#include "auto_aim_fast/util/time.hpp"
#include "auto_aim_fast/vision/preprocess.hpp"
#include "auto_aim_fast/vision/light_extractor.hpp"
#include "auto_aim_fast/vision/pairing.hpp"
#include "auto_aim_fast/vision/refine.hpp"
#include "auto_aim_fast/pose/pnp_solver.hpp"
#include "auto_aim_fast/pose/pose_quality.hpp"
#include "auto_aim_fast/tracking/target_tracker.hpp"
#include "auto_aim_fast/prediction/aim_solution.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace auto_aim_fast::pipeline
{

struct Pipeline::Impl
{
  mutable std::mutex m;
  std::condition_variable cv;
  std::optional<Frame> latest;
  bool has_new{false};

  std::thread worker;
  std::atomic<bool> running{false};

  mutable std::mutex out_m;
  PipelineOutput last_out;

  mutable std::mutex debug_m;
  cv::Mat last_filtered_debug_bgr;
  int32_t last_debug_stamp_sec{0};
  uint32_t last_debug_stamp_nanosec{0};
  std::string last_debug_frame_id;
  uint64_t last_debug_sequence{0};
  std::atomic<bool> debug_image_enabled{false};

  ROI last_roi;
  tracking::TrackState last_state{tracking::TrackState::SEARCH};

  std::atomic<uint64_t> pushed{0};
  std::atomic<uint64_t> processed{0};
  std::atomic<uint64_t> dropped{0};

  vision::Preprocess preprocess;
  vision::LightExtractor light_extractor;
  vision::Pairing pairing;
  vision::Refine refine;

  pose::PnPSolver pnp;
  pose::PoseQuality pose_quality;

  tracking::TargetTracker tracker;
  prediction::AimSolver aim_solver;

  LaserSpotFilterParams laser_spot_filter;
};

static std::string stateToString(tracking::TrackState s)
{
  switch (s) {
    case tracking::TrackState::TRACK: return "TRACK";
    case tracking::TrackState::LOST:  return "LOST";
    default: return "SEARCH";
  }
}

static bool armorBounds(const vision::Armor2D & armor, cv::Rect2f & bounds)
{
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();

  for (const auto & p : armor.corners) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
      continue;
    }
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }

  if (!std::isfinite(min_x) || !std::isfinite(min_y) ||
      !std::isfinite(max_x) || !std::isfinite(max_y) ||
      max_x <= min_x || max_y <= min_y)
  {
    return false;
  }

  bounds = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
  return true;
}

static cv::Rect localCropRect(
  const cv::Rect2f & bounds,
  const vision::PreprocessResult & pre,
  float padding_ratio)
{
  const float offset_x = pre.roi_used ? static_cast<float>(pre.roi_rect.x) : 0.0f;
  const float offset_y = pre.roi_used ? static_cast<float>(pre.roi_rect.y) : 0.0f;
  const float pad_x = std::max(0.0f, padding_ratio) * bounds.width;
  const float pad_y = std::max(0.0f, padding_ratio) * bounds.height;

  const int x0 = static_cast<int>(std::floor(bounds.x - offset_x - pad_x));
  const int y0 = static_cast<int>(std::floor(bounds.y - offset_y - pad_y));
  const int x1 = static_cast<int>(std::ceil(bounds.x + bounds.width - offset_x + pad_x));
  const int y1 = static_cast<int>(std::ceil(bounds.y + bounds.height - offset_y + pad_y));

  if (x1 <= x0 || y1 <= y0) {
    return cv::Rect();
  }

  const cv::Rect image_rect(0, 0, pre.bgr.cols, pre.bgr.rows);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0) & image_rect;
}

static bool hasLowVerticalBrightnessStructure(
  const cv::Mat & bgr,
  const LaserSpotFilterParams & params)
{
  if (bgr.empty() || bgr.rows < 4 || bgr.cols < 4) {
    return false;
  }

  cv::Mat gray;
  if (bgr.channels() == 1) {
    gray = bgr;
  } else {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  }

  const int band_x = gray.cols / 5;
  const int band_w = std::max(2, gray.cols - 2 * band_x);
  const cv::Mat center_band = gray(cv::Rect(band_x, 0, band_w, gray.rows));

  cv::Mat row_mean;
  cv::reduce(center_band, row_mean, 1, cv::REDUCE_AVG, CV_32F);

  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(row_mean, mean, stddev);

  const float mean_brightness = static_cast<float>(mean[0]);

  const float vertical_cv =
    static_cast<float>(stddev[0]) / std::max(1.0f, mean_brightness);

  float diff_sum = 0.0f;
  for (int r = 1; r < row_mean.rows; ++r) {
    diff_sum += std::abs(row_mean.at<float>(r, 0) - row_mean.at<float>(r - 1, 0));
  }
  const float vertical_gradient =
    diff_sum / (std::max(1, row_mean.rows - 1) * std::max(1.0f, mean_brightness));

  return vertical_cv < params.min_vertical_cv &&
         vertical_gradient < params.min_vertical_gradient;
}

static bool isBrightSmoothRoundSpot(
  const vision::Armor2D & armor,
  const vision::PreprocessResult & pre,
  const LaserSpotFilterParams & params)
{
  if (!params.enable || pre.bgr.empty()) {
    return false;
  }

  cv::Rect2f bounds;
  if (!armorBounds(armor, bounds)) {
    return false;
  }

  const float short_side = std::max(1.0f, std::min(bounds.width, bounds.height));
  const float long_side = std::max(bounds.width, bounds.height);
  if (short_side < std::max(1.0f, params.min_side_px)) {
    return false;
  }

  const float aspect = long_side / short_side;
  if (aspect > std::max(1.0f, params.max_near_circle_aspect)) {
    return false;
  }

  const cv::Rect crop = localCropRect(bounds, pre, params.roi_padding_ratio);
  if (crop.width < 4 || crop.height < 4) {
    return false;
  }

  return hasLowVerticalBrightnessStructure(pre.bgr(crop), params);
}

static void drawArmorGlobal(
  cv::Mat & image,
  const vision::Armor2D & armor,
  const cv::Scalar & color)
{
  for (int i = 0; i < 4; ++i) {
    cv::line(image, armor.corners[i], armor.corners[(i + 1) % 4], color, 2);
  }
}

Pipeline::Pipeline() : impl_(std::make_unique<Impl>()) {}
Pipeline::~Pipeline() { stop(); }

void Pipeline::start()
{
  bool expected = false;
  if (!impl_->running.compare_exchange_strong(expected, true)) return;

  impl_->worker = std::thread([this]() {
    while (impl_->running.load(std::memory_order_relaxed)) {
      std::optional<Frame> f;

      {
        std::unique_lock<std::mutex> lk(impl_->m);
        impl_->cv.wait(lk, [&]() {
          return !impl_->running.load(std::memory_order_relaxed) || impl_->has_new;
        });
        if (!impl_->running.load(std::memory_order_relaxed)) break;

        f = impl_->latest;
        impl_->latest.reset();
        impl_->has_new = false;
      }

      if (!f.has_value()) continue;

      PipelineOutput out = processFrame(*f);
      out.pushed_frames = impl_->pushed.load(std::memory_order_relaxed);
      out.processed_frames = impl_->processed.load(std::memory_order_relaxed);
      out.dropped_frames = impl_->dropped.load(std::memory_order_relaxed);

      {
        std::lock_guard<std::mutex> lk(impl_->out_m);
        impl_->last_out = out;
      }
    }
  });
}

void Pipeline::stop()
{
  if (!impl_->running.exchange(false)) return;
  {
    std::lock_guard<std::mutex> lk(impl_->m);
    impl_->has_new = true;
  }
  impl_->cv.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
}

void Pipeline::pushFrame(const Frame & in_frame)
{
  impl_->pushed.fetch_add(1, std::memory_order_relaxed);

  Frame frame = in_frame;

  {
    std::lock_guard<std::mutex> lk(impl_->out_m);
    if (impl_->last_state != tracking::TrackState::SEARCH && impl_->last_roi.valid) {
      frame.roi = impl_->last_roi;
    } else {
      frame.roi = ROI{};
    }
  }

  {
    std::lock_guard<std::mutex> lk(impl_->m);
    if (impl_->latest.has_value()) {
      impl_->dropped.fetch_add(1, std::memory_order_relaxed);
    }
    impl_->latest = frame;
    impl_->has_new = true;
  }
  impl_->cv.notify_one();
}

PipelineOutput Pipeline::getLatestOutput() const
{
  std::lock_guard<std::mutex> lk(impl_->out_m);
  return impl_->last_out;
}

void Pipeline::setTrackerParams(const tracking::TrackerParams & p)
{
  impl_->tracker.setParams(p);
}

void Pipeline::setDetectorParams(const SimpleLightDetector::Params & p)
{
  impl_->pairing.setLegacyParams(p);
}

void Pipeline::setDetectorMode(const vision::DetectorMode mode)
{
  impl_->pairing.setDetectorMode(mode);
}

void Pipeline::setYoloDetectorParams(const vision::YoloDetectorParams & p)
{
  impl_->pairing.setYoloParams(p);
}

void Pipeline::setLaserSpotFilterParams(const LaserSpotFilterParams & p)
{
  impl_->laser_spot_filter = p;
}

void Pipeline::setDebugImageEnabled(bool enable)
{
  impl_->debug_image_enabled.store(enable, std::memory_order_relaxed);
}

void Pipeline::setCamera(const cv::Mat & K, const cv::Mat & D)
{
  impl_->pnp.setCamera(K, D);
}

bool Pipeline::getLastDebugImages(
  cv::Mat & out_debug_bgr,
  cv::Mat & out_binary,
  int32_t & stamp_sec,
  uint32_t & stamp_nanosec,
  std::string & frame_id,
  uint64_t & sequence) const
{
  bool ok1 = false;
  stamp_sec = 0;
  stamp_nanosec = 0;
  frame_id.clear();
  sequence = 0;
  {
    std::lock_guard<std::mutex> lk(impl_->debug_m);
    if (!impl_->last_filtered_debug_bgr.empty()) {
      out_debug_bgr = impl_->last_filtered_debug_bgr.clone();
      stamp_sec = impl_->last_debug_stamp_sec;
      stamp_nanosec = impl_->last_debug_stamp_nanosec;
      frame_id = impl_->last_debug_frame_id;
      sequence = impl_->last_debug_sequence;
      ok1 = true;
    }
  }
  if (!ok1) {
    ok1 = impl_->pairing.getLastDebugBgr(out_debug_bgr);
  }
  const bool ok2 = impl_->pairing.getLastBinary(out_binary);
  return ok1 || ok2;
}

PipelineOutput Pipeline::processFrame(const Frame & f)
{
  impl_->processed.fetch_add(1, std::memory_order_relaxed);

  PipelineOutput out;

  // stamp from header ns
  out.stamp_sec = static_cast<int32_t>(f.stamp_ns / 1000000000LL);
  out.stamp_nanosec = static_cast<uint32_t>(f.stamp_ns % 1000000000LL);
  out.recv_ns = f.recv_ns;
  out.frame_id = f.frame_id;

  util::StageTimer timer;

  timer.tic();
  const auto pre = impl_->preprocess.run(f);
  out.stats.preprocess_ns = timer.tocNs();
  if (!pre.ok) {
    out.state = "SEARCH";
    return out;
  }

  out.roi_dbg.used = pre.roi_used;
  out.roi_dbg.x = pre.roi_rect.x;
  out.roi_dbg.y = pre.roi_rect.y;
  out.roi_dbg.w = pre.roi_rect.width;
  out.roi_dbg.h = pre.roi_rect.height;
  const cv::Mat & full_debug_source = pre.full_bgr.empty() ? pre.bgr : pre.full_bgr;
  out.roi_dbg.img_w = full_debug_source.cols;
  out.roi_dbg.img_h = full_debug_source.rows;

  timer.tic();
  auto lights = impl_->light_extractor.extract(pre);
  out.stats.lights_ns = timer.tocNs();

  out.budget.lights_in = static_cast<int>(lights.size());
  if (static_cast<int>(lights.size()) > f.budget.max_lights) {
    lights.resize(static_cast<size_t>(f.budget.max_lights));
  }
  out.budget.lights_used = static_cast<int>(lights.size());

  timer.tic();
  auto armors = impl_->pairing.pair(lights, pre);
  out.stats.pairing_ns = timer.tocNs();

  out.budget.pairs_in = static_cast<int>(armors.size());
  const auto armors_before_filter = armors;
  if (impl_->laser_spot_filter.enable && !armors.empty()) {
    armors.erase(
      std::remove_if(
        armors.begin(),
        armors.end(),
        [&](const vision::Armor2D & armor) {
          return isBrightSmoothRoundSpot(armor, pre, impl_->laser_spot_filter);
        }),
      armors.end());
  }
  out.budget.pairs_used = static_cast<int>(armors.size());

  if (impl_->debug_image_enabled.load(std::memory_order_relaxed)) {
    cv::Mat dbg = full_debug_source.clone();
    for (const auto & armor : armors_before_filter) {
      drawArmorGlobal(dbg, armor, cv::Scalar(0, 255, 255));
    }
    {
      std::lock_guard<std::mutex> lk(impl_->debug_m);
      impl_->last_filtered_debug_bgr = dbg;
      impl_->last_debug_stamp_sec = out.stamp_sec;
      impl_->last_debug_stamp_nanosec = out.stamp_nanosec;
      impl_->last_debug_frame_id = out.frame_id;
      impl_->last_debug_sequence++;
    }
  }

  if (armors.empty()) {
    const pose::Pose3D meas{};
    timer.tic();
    const auto trk = impl_->tracker.update(meas, f.stamp_ns);
    out.stats.tracker_ns = timer.tocNs();
    out.state = stateToString(trk.state);
    out.good_count = trk.debug.good_count;
    out.miss_count = trk.debug.miss_count;

    {
      std::lock_guard<std::mutex> lk(impl_->out_m);
      impl_->last_state = trk.state;
      impl_->last_roi = ROI{trk.roi.x, trk.roi.y, trk.roi.w, trk.roi.h, trk.roi.valid};
    }

    out.has_target_center = false;
    out.target_u = 0.0f;
    out.target_v = 0.0f;
    return out;
  }

  timer.tic();
  const auto best2d = impl_->refine.refine(armors.front());
  out.stats.refine_ns = timer.tocNs();

  timer.tic();
  auto pose3d = impl_->pnp.solve(best2d);
  out.stats.pnp_ns = timer.tocNs();

  out.has_target_center = true;
  out.target_u = best2d.center.x;
  out.target_v = best2d.center.y;

  pose3d.score = impl_->pose_quality.confidence(pose3d);

  if (pose3d.ok) {
    out.has_pose = true;
    out.tx = pose3d.x;
    out.ty = pose3d.y;
    out.tz = pose3d.z;

    const float x = out.tx, y = out.ty, z = out.tz;
    out.yaw = std::atan2(x, z);
    const float horiz = std::sqrt(x * x + z * z);
    out.pitch = std::atan2(-y, horiz);
    out.dist = std::sqrt(x * x + y * y + z * z);
    out.confidence = pose3d.score;
  }

  timer.tic();
  const auto trk = impl_->tracker.update(pose3d, f.stamp_ns);
  out.stats.tracker_ns = timer.tocNs();
  out.state = stateToString(trk.state);
  out.good_count = trk.debug.good_count;
  out.miss_count = trk.debug.miss_count;

  {
    std::lock_guard<std::mutex> lk(impl_->out_m);
    impl_->last_state = trk.state;
    impl_->last_roi = ROI{trk.roi.x, trk.roi.y, trk.roi.w, trk.roi.h, trk.roi.valid};
  }

  timer.tic();
  const auto aim = impl_->aim_solver.solve(trk);
  out.stats.aim_ns = timer.tocNs();

  out.has_target = aim.ok;
  if (!out.has_pose) {
    out.yaw = aim.yaw;
    out.pitch = aim.pitch;
    out.confidence = aim.confidence;
    out.dist = trk.dist;
  }

  return out;
}

}  // namespace auto_aim_fast::pipeline
