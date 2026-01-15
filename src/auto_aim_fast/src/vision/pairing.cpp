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

  // 2) build debug images (复现优先：每次都产出最新 debug/binary)
  {
    cv::Mat dbg = pre.bgr.clone();
    detector_->drawResults(dbg, true, true);

    cv::Mat bin = detector_->getBinaryImage(); // usually MONO8

    std::lock_guard<std::mutex> lk(dbg_m_);
    last_debug_bgr_ = dbg;
    last_binary_ = bin.clone();
  }

  if (pairs.empty()) {
    return {};
  }

  // 3) convert to Armor2D list
  std::vector<Armor2D> out;
  out.reserve(pairs.size());

  for (const auto & pr : pairs) {
    Armor2D a;
    a.center = pr.center;
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
