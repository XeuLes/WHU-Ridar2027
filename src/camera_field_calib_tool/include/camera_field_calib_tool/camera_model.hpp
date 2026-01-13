#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace camera_field_calib_tool
{

class CameraModel
{
public:
  CameraModel() = default;

  // Initialize from intrinsics
  bool SetIntrinsics(
    const cv::Matx33d & K,
    const std::vector<double> & D,
    int width = 0,
    int height = 0);

  bool IsValid() const;

  // Accessors
  const cv::Matx33d & K() const { return K_; }
  const std::vector<double> & D() const { return D_; }
  int width() const { return width_; }
  int height() const { return height_; }

  // -----------------------------
  // Pixel processing
  // -----------------------------

  // Undistort a single pixel (u,v) -> normalized camera coordinates (x,y,1)
  // Returned vector is NOT normalized to unit length.
  cv::Vec3d PixelToNormalizedRay(
    const cv::Point2d & pixel) const;

  // Batch version
  std::vector<cv::Vec3d> PixelToNormalizedRays(
    const std::vector<cv::Point2d> & pixels) const;

  // Project a 3D point in camera coordinates to pixel (no distortion)
  cv::Point2d CameraToPixelNoDistortion(
    const cv::Point3d & pc) const;

  // Utility: check if pixel lies inside image bounds (if width/height known)
  bool IsPixelInside(const cv::Point2d & pixel) const;

private:
  bool valid_{false};

  cv::Matx33d K_{cv::Matx33d::eye()};
  std::vector<double> D_{};

  int width_{0};
  int height_{0};
};

}  // namespace camera_field_calib_tool
