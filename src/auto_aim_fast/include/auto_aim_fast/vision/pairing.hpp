#pragma once

#include "auto_aim_fast/vision/light_extractor.hpp"
#include "auto_aim_fast/vision/preprocess.hpp"
#include "auto_aim_fast/legacy/detector.hpp"

#include <string>

#include <array>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>


namespace auto_aim_fast { class YoloPoseTrtDetector; }

namespace auto_aim_fast::vision
{

enum class DetectorMode : int
{
  LEGACY = 0,
  YOLO = 1
};

struct YoloDetectorParams
{
  std::string engine_path;
  int input_w{640};
  int input_h{640};
  int max_det{300};
  int target_class_id{0};
  float conf_thresh{0.25f};
  float nms_thresh{0.45f};
  float kpt_conf_thresh{0.25f};
  float min_w_ratio{0.045f};
  float min_h_ratio{0.055f};
  float min_area_ratio{0.0025f};
  bool require_outer_corners{true};
  bool use_fp16_input{true};
  bool swap_rb_input{false};
  std::array<int, 4> outer_corner_indices{{0, 4, 7, 3}};  // lt lb rb rt
};

enum class ArmorType : int
{
  SMALL = 0,
  BIG   = 1
};

struct Armor2D
{
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f center{0.f, 0.f};

  ArmorType type{ArmorType::SMALL};

  float score{0.f};
  bool valid{false};

  SimpleLightDetector::PairResult legacy_pair{};
  bool has_legacy_pair{false};
};

class Pairing
{
public:
  Pairing();
  void setLegacyParams(const SimpleLightDetector::Params & p);
  void setDetectorMode(DetectorMode mode);
  void setYoloParams(const YoloDetectorParams & p);

  std::vector<Armor2D> pair(const std::vector<LightBar> & lights,
                            const PreprocessResult & pre);

  // Debug retrieval (thread-safe snapshots)
  bool getLastDebugBgr(cv::Mat & out_bgr) const;
  bool getLastBinary(cv::Mat & out_bin) const;

private:
  std::shared_ptr<SimpleLightDetector> detector_;
  std::shared_ptr<auto_aim_fast::YoloPoseTrtDetector> yolo_detector_;
  DetectorMode mode_{DetectorMode::LEGACY};
  YoloDetectorParams yolo_params_{};
  bool yolo_init_ok_{false};
  bool yolo_init_attempted_{false};

  // stored debug images
  mutable std::mutex dbg_m_;
  mutable cv::Mat last_debug_bgr_;
  mutable cv::Mat last_binary_;
};

}  // namespace auto_aim_fast::vision
