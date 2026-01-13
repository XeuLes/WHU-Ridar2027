#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "camera_field_calib_tool/camera_model.hpp"
#include "camera_field_calib_tool/reprojection_metrics.hpp"

namespace camera_field_calib_tool
{

struct PnPResult
{
  bool success{false};
  cv::Vec3d rvec{0.0, 0.0, 0.0};
  cv::Vec3d tvec{0.0, 0.0, 0.0};
  ReprojectionStats reproj{};
};

class PnPCalibrator
{
public:
  PnPCalibrator() = default;

  void SetCameraModel(const CameraModel & cam);

  // Point management
  void Reset();
  size_t Size() const;

  const std::vector<cv::Point3d> & ObjectPoints() const;
  const std::vector<cv::Point2d> & ImagePoints() const;

  // Add a correspondence pair
  void AddPoint(const cv::Point3d & object_pt, const cv::Point2d & image_pt);

  // Undo last point
  bool Undo();

  // Solve PnP
  // method: OpenCV solvePnP flag (e.g. cv::SOLVEPNP_EPNP, cv::SOLVEPNP_ITERATIVE)
  // use_refine: if true, try solvePnPRefineLM (requires ITERATIVE/initial)
  PnPResult Solve(int method = 0, bool use_refine = true) const;

  // Minimal required points for the chosen method (guidance)
  static int RecommendedMinPoints(int method);

private:
  CameraModel cam_;
  bool has_cam_{false};

  std::vector<cv::Point3d> object_pts_;
  std::vector<cv::Point2d> image_pts_;

  bool ComputeReprojection(
    const cv::Vec3d & rvec,
    const cv::Vec3d & tvec,
    ReprojectionStats & stats) const;
};

}  // namespace camera_field_calib_tool
