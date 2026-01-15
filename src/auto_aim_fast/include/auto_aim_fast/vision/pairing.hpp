#pragma once

#include "auto_aim_fast/vision/light_extractor.hpp"
#include "auto_aim_fast/vision/preprocess.hpp"
#include "auto_aim_fast/legacy/detector.hpp"

#include <array>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

namespace auto_aim_fast::vision
{

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

  std::vector<Armor2D> pair(const std::vector<LightBar> & lights,
                            const PreprocessResult & pre) const;

  // Debug retrieval (thread-safe snapshots)
  bool getLastDebugBgr(cv::Mat & out_bgr) const;
  bool getLastBinary(cv::Mat & out_bin) const;

private:
  std::shared_ptr<SimpleLightDetector> detector_;

  // stored debug images
  mutable std::mutex dbg_m_;
  mutable cv::Mat last_debug_bgr_;
  mutable cv::Mat last_binary_;
};

}  // namespace auto_aim_fast::vision
