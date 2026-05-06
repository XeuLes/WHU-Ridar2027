
#include "auto_aim_fast/vision/pairing.hpp"
#include "auto_aim_fast/legacy/yolo_trt_detector.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace auto_aim_fast::vision
{

namespace
{

inline void pairResultToFallbackCorners(const SimpleLightDetector::PairResult & pr,
                                        std::array<cv::Point2f, 4> & corners)
{
  const float half_w = pr.width * 0.5f;
  const float half_h = pr.height * 0.5f;
  const float c = std::cos(pr.angle);
  const float s = std::sin(pr.angle);

  auto rot = [&](float x, float y) -> cv::Point2f {
    return cv::Point2f(pr.center.x + c * x - s * y,
                       pr.center.y + s * x + c * y);
  };

  corners[0] = rot(-half_w, -half_h);  // LT
  corners[1] = rot( half_w, -half_h);  // RT
  corners[2] = rot( half_w,  half_h);  // RB
  corners[3] = rot(-half_w,  half_h);  // LB
}

inline void applyOffset(SimpleLightDetector::PairResult & pr, float ox, float oy)
{
  const cv::Point2f d(ox, oy);
  pr.center += d;
  pr.light1_center += d;
  pr.light2_center += d;

  if (pr.has_precise_corners) {
    pr.precise_left_top += d;
    pr.precise_left_bottom += d;
    pr.precise_right_top += d;
    pr.precise_right_bottom += d;
  }

  if (pr.has_keypoints) {
    for (auto & kp : pr.keypoints) {
      kp += d;
    }
  }
}

inline Armor2D makeArmorFromPair(const SimpleLightDetector::PairResult & pr,
                                 bool use_legacy_pair)
{
  Armor2D a;
  a.center = pr.center;
  a.score = pr.confidence;
  a.valid = true;
  a.type = ArmorType::SMALL;

  if (pr.has_precise_corners) {
    a.corners[0] = pr.precise_left_top;
    a.corners[1] = pr.precise_right_top;
    a.corners[2] = pr.precise_right_bottom;
    a.corners[3] = pr.precise_left_bottom;
  } else {
    pairResultToFallbackCorners(pr, a.corners);
  }

  a.has_legacy_pair = use_legacy_pair;
  if (use_legacy_pair) {
    a.legacy_pair = pr;
  }

  return a;
}

}  // namespace

Pairing::Pairing()
: detector_(std::make_shared<SimpleLightDetector>())
{
}

void Pairing::setLegacyParams(const SimpleLightDetector::Params & p)
{
  std::lock_guard<std::mutex> lk(params_m_);
  detector_->setParams(p);
}

void Pairing::setDetectorMode(DetectorMode mode)
{
  std::lock_guard<std::mutex> lk(params_m_);
  mode_ = mode;
}

void Pairing::setYoloParams(const YoloDetectorParams & p)
{
  std::lock_guard<std::mutex> lk(params_m_);
  yolo_params_ = p;
  yolo_init_ok_ = false;
  yolo_init_attempted_ = false;
  yolo_detector_.reset();
}

void Pairing::setDebugEnabled(bool enable)
{
  debug_enabled_.store(enable, std::memory_order_relaxed);
}

std::vector<Armor2D> Pairing::pair(const std::vector<LightBar> & lights,
                                   const PreprocessResult & pre)
{
  (void)lights;

  if (!pre.ok || pre.bgr.empty()) {
    return {};
  }

  std::lock_guard<std::mutex> lk(params_m_);

  std::vector<SimpleLightDetector::PairResult> pairs;
  const bool build_debug = debug_enabled_.load(std::memory_order_relaxed);

  if (mode_ == DetectorMode::YOLO) {
    if (!yolo_detector_) {
      yolo_detector_ = std::make_shared<auto_aim_fast::YoloPoseTrtDetector>();
    }

    if (!yolo_init_attempted_) {
      yolo_init_attempted_ = true;
      auto_aim_fast::YoloPoseTrtDetector::Params yp;
      yp.engine_path = yolo_params_.engine_path;
      yp.input_w = yolo_params_.input_w;
      yp.input_h = yolo_params_.input_h;
      yp.max_det = yolo_params_.max_det;
      yp.target_class_id = yolo_params_.target_class_id;
      yp.conf_thresh = yolo_params_.conf_thresh;
      yp.nms_thresh = yolo_params_.nms_thresh;
      yp.kpt_conf_thresh = yolo_params_.kpt_conf_thresh;
      yp.min_w_ratio = yolo_params_.min_w_ratio;
      yp.min_h_ratio = yolo_params_.min_h_ratio;
      yp.min_area_ratio = yolo_params_.min_area_ratio;
      yp.require_outer_corners = yolo_params_.require_outer_corners;
      yp.use_fp16_input = yolo_params_.use_fp16_input;
      yp.swap_rb_input = yolo_params_.swap_rb_input;
      yp.outer_corner_indices = yolo_params_.outer_corner_indices;
      yolo_init_ok_ = yolo_detector_->init(yp);
    }

    if (yolo_init_ok_ && yolo_detector_->isReady()) {
      pairs = yolo_detector_->detectDetailed(pre.bgr);

      if (build_debug) {
        cv::Mat dbg = pre.bgr.clone();
        for (const auto & pr : pairs) {
          cv::Scalar box_color(0, 255, 255);
          if (pr.has_precise_corners) {
            cv::line(dbg, pr.precise_left_top, pr.precise_right_top, box_color, 2);
            cv::line(dbg, pr.precise_right_top, pr.precise_right_bottom, box_color, 2);
            cv::line(dbg, pr.precise_right_bottom, pr.precise_left_bottom, box_color, 2);
            cv::line(dbg, pr.precise_left_bottom, pr.precise_left_top, box_color, 2);
          } else {
            std::array<cv::Point2f, 4> c{};
            pairResultToFallbackCorners(pr, c);
            for (int i = 0; i < 4; ++i) {
              cv::line(dbg, c[i], c[(i + 1) % 4], box_color, 2);
            }
          }

          cv::circle(dbg, pr.center, 4, cv::Scalar(0, 0, 255), -1);

          if (pr.has_keypoints) {
            for (size_t i = 0; i < pr.keypoints.size(); ++i) {
              const auto & kp = pr.keypoints[i];
              const float ks = pr.keypoint_scores[i];
              cv::circle(dbg, kp, 3, ks >= yolo_params_.kpt_conf_thresh ?
                cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), -1);
            }
          }
        }

        cv::Mat bin(pre.bgr.rows, pre.bgr.cols, CV_8UC1, cv::Scalar(0));
        {
          std::lock_guard<std::mutex> lk(dbg_m_);
          last_debug_bgr_ = dbg;
          last_binary_ = bin;
        }
      }
    }
  }

  if (pairs.empty() && mode_ == DetectorMode::LEGACY) {
    pairs = detector_->detectDetailed(pre.bgr);

    static uint64_t debug_build_counter = 0;
    constexpr uint64_t kDebugDecimation = 3;
    debug_build_counter++;

    if (build_debug && (debug_build_counter % kDebugDecimation) == 0) {
      cv::Mat dbg = pre.bgr.clone();
      detector_->drawResults(dbg, true, true);

      cv::Mat bin = detector_->getBinaryImage();

      std::lock_guard<std::mutex> lk(dbg_m_);
      last_debug_bgr_ = dbg;
      last_binary_ = bin;
    }
  } else if (pairs.empty()) {
    // YOLO mode failed or no result: keep debug image but fall back to empty result.
  }

  if (pairs.empty()) {
    return {};
  }

  const float offset_x = pre.roi_used ? static_cast<float>(pre.roi_rect.x) : 0.0f;
  const float offset_y = pre.roi_used ? static_cast<float>(pre.roi_rect.y) : 0.0f;

  std::vector<Armor2D> out;
  out.reserve(pairs.size());

  for (auto pr : pairs) {
    applyOffset(pr, offset_x, offset_y);
    const bool use_legacy_pair = (mode_ == DetectorMode::LEGACY);
    out.push_back(makeArmorFromPair(pr, use_legacy_pair));
  }

  std::sort(out.begin(), out.end(),
            [](const Armor2D & a, const Armor2D & b) { return a.score > b.score; });

  return out;
}

bool Pairing::getLastDebugBgr(cv::Mat & out_bgr) const
{
  std::lock_guard<std::mutex> lk(dbg_m_);
  if (last_debug_bgr_.empty()) return false;
  out_bgr = last_debug_bgr_;
  return true;
}

bool Pairing::getLastBinary(cv::Mat & out_bin) const
{
  std::lock_guard<std::mutex> lk(dbg_m_);
  if (last_binary_.empty()) return false;
  out_bin = last_binary_;
  return true;
}

}  // namespace auto_aim_fast::vision
