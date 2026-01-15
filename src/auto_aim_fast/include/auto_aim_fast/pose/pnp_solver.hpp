#pragma once

#include "auto_aim_fast/vision/pairing.hpp"

#include <memory>

namespace cv { class Mat; }

namespace auto_aim_fast::pose
{

struct Pose3D
{
  // translation in camera frame
  float x{0}, y{0}, z{0};

  // 2D measurement center (for ROI)
  bool has_img_center{false};
  float img_cx{0}, img_cy{0};

  float score{0};
  bool ok{false};
};

class PnPSolver
{
public:
  PnPSolver();
  ~PnPSolver();

  void setCamera(const cv::Mat & K, const cv::Mat & D);

  Pose3D solve(const auto_aim_fast::vision::Armor2D & armor) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace auto_aim_fast::pose
