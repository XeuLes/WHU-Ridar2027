#pragma once

#include "auto_aim_fast/pipeline/frame.hpp"
#include "auto_aim_fast/tracking/target_tracker.hpp"
#include "auto_aim_fast/legacy/detector.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace cv { class Mat; }

namespace auto_aim_fast::pipeline
{

struct StageStats
{
  int64_t preprocess_ns{0};
  int64_t lights_ns{0};
  int64_t pairing_ns{0};
  int64_t refine_ns{0};
  int64_t pnp_ns{0};
  int64_t tracker_ns{0};
  int64_t aim_ns{0};
};

struct BudgetHit
{
  int lights_in{0};
  int lights_used{0};
  int pairs_in{0};
  int pairs_used{0};
};

struct RoiDebug
{
  bool used{false};
  int x{0}, y{0}, w{0}, h{0};
  int img_w{0}, img_h{0};
};

struct PipelineOutput
{
  int32_t stamp_sec{0};
  uint32_t stamp_nanosec{0};
  int64_t recv_ns{0};

  bool has_pose{false};
  float tx{0}, ty{0}, tz{0};

  bool has_target{false};
  float yaw{0};
  float pitch{0};
  float dist{0};
  float confidence{0};

  std::string state{"SEARCH"};

  StageStats stats;
  BudgetHit budget;
  RoiDebug roi_dbg;

  int good_count{0};
  int miss_count{0};

  uint64_t pushed_frames{0};
  uint64_t processed_frames{0};
  uint64_t dropped_frames{0};

  bool has_target_center{false};
  float target_u{0.0f};
  float target_v{0.0f};

};

class Pipeline
{
public:
  Pipeline();
  ~Pipeline();

  void start();
  void stop();

  void pushFrame(const Frame & frame);
  PipelineOutput getLatestOutput() const;

  void setTrackerParams(const tracking::TrackerParams & p);
  void setDetectorParams(const SimpleLightDetector::Params & p);
  void setCamera(const cv::Mat & K, const cv::Mat & D);

  // Debug images from legacy detector (thread-safe snapshots)
  bool getLastDebugImages(cv::Mat & out_debug_bgr, cv::Mat & out_binary) const;

private:
  PipelineOutput processFrame(const Frame & f);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace auto_aim_fast::pipeline
