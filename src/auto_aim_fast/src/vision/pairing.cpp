#include "auto_aim_fast/vision/pairing.hpp"

#include <algorithm>

namespace auto_aim_fast::vision
{

Pairing::Pairing()
: detector_(std::make_shared<SimpleLightDetector>())
{
}

void Pairing::setLegacyParams(const SimpleLightDetector::Params & p)
{
  detector_->setParams(p);
}

std::vector<Armor2D> Pairing::pair(const std::vector<LightBar> & lights,
                                   const PreprocessResult & pre) const
{
  (void)lights;

  if (!pre.ok || pre.bgr.empty()) {
    return {};
  }

  // 1) run legacy detector
  const auto pairs = detector_->detectDetailed(pre.bgr);

  // 2) build debug images (降频，避免每帧 clone/draw)
{
  // 最小改法：先写死每10帧生成一次
  // 如果效果好，后面再把这个值做成参数传下来
  static uint64_t debug_build_counter = 0;
  constexpr uint64_t kDebugDecimation = 3;

  debug_build_counter++;

  if ((debug_build_counter % kDebugDecimation) == 0) {
    cv::Mat dbg = pre.bgr.clone();
    detector_->drawResults(dbg, true, true);

    cv::Mat bin = detector_->getBinaryImage(); // usually MONO8

    std::lock_guard<std::mutex> lk(dbg_m_);
    last_debug_bgr_ = dbg;
    last_binary_ = bin.clone();
  }
}

  if (pairs.empty()) {
    return {};
  }

  // 3) convert to Armor2D list
  std::vector<Armor2D> out;
  out.reserve(pairs.size());

  // 计算 ROI 偏移量（如果使用了 ROI）
  float offset_x = 0.0f;
  float offset_y = 0.0f;
  if (pre.roi_used) {
    offset_x = static_cast<float>(pre.roi_rect.x);
    offset_y = static_cast<float>(pre.roi_rect.y);
  }

  for (const auto & pr : pairs) {
    Armor2D a;
    
    // 将 ROI 局部坐标转换为全局坐标
    a.center.x = pr.center.x + offset_x;
    a.center.y = pr.center.y + offset_y;
    
    a.score = pr.confidence;
    a.valid = true;

    a.legacy_pair = pr;
    a.has_legacy_pair = true;

    a.type = ArmorType::SMALL;
    
    out.push_back(a);
  }

  // highest confidence first
  std::sort(out.begin(), out.end(),
            [](const Armor2D & a, const Armor2D & b) { return a.score > b.score; });

  return out;
}

bool Pairing::getLastDebugBgr(cv::Mat & out_bgr) const
{
  std::lock_guard<std::mutex> lk(dbg_m_);
  if (last_debug_bgr_.empty()) return false;
  out_bgr = last_debug_bgr_.clone();
  return true;
}

bool Pairing::getLastBinary(cv::Mat & out_bin) const
{
  std::lock_guard<std::mutex> lk(dbg_m_);
  if (last_binary_.empty()) return false;
  out_bin = last_binary_.clone();
  return true;
}

}  // namespace auto_aim_fast::vision
