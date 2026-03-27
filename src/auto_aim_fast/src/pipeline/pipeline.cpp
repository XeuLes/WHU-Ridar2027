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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <cmath>
#include <opencv2/core.hpp>

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
};

static std::string stateToString(tracking::TrackState s)
{
  switch (s) {
    case tracking::TrackState::TRACK: return "TRACK";
    case tracking::TrackState::LOST:  return "LOST";
    default: return "SEARCH";
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

void Pipeline::setCamera(const cv::Mat & K, const cv::Mat & D)
{
  impl_->pnp.setCamera(K, D);
}

bool Pipeline::getLastDebugImages(cv::Mat & out_debug_bgr, cv::Mat & out_binary) const
{
  // snapshots from Pairing (thread-safe internally)
  const bool ok1 = impl_->pairing.getLastDebugBgr(out_debug_bgr);
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
  out.roi_dbg.img_w = pre.bgr.cols;
  out.roi_dbg.img_h = pre.bgr.rows;

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
  out.budget.pairs_used = out.budget.pairs_in;

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
