#pragma once

#include "auto_aim_fast/pipeline/frame.hpp"
#include <opencv2/core.hpp>

namespace auto_aim_fast::vision
{

struct PreprocessResult
{
  bool ok{false};

  // BGR view (possibly ROI-cropped)
  cv::Mat bgr;

  // record ROI actually used
  bool roi_used{false};
  cv::Rect roi_rect;
};

class Preprocess
{
public:
  Preprocess() = default;
  PreprocessResult run(const auto_aim_fast::pipeline::Frame & frame) const;
};

}  // namespace auto_aim_fast::vision
