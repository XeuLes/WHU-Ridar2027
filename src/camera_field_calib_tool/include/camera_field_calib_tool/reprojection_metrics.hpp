#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace camera_field_calib_tool
{

struct ReprojectionStats
{
  double rmse{0.0};     // Root-mean-square error (pixels)
  double max_error{0.0};
  std::vector<double> per_point_errors;  // optional, same order as input
};

class ReprojectionMetrics
{
public:
  // Compute reprojection errors using OpenCV projectPoints convention
  //
  // Inputs:
  //  - object_points: world 3D points
  //  - image_points: observed image 2D points
  //  - rvec, tvec: extrinsics (OpenCV convention)
  //  - K: camera matrix
  //  - D: distortion coefficients
  //
  // Returns false if input sizes mismatch or projection fails.
  static bool Compute(
    const std::vector<cv::Point3d> & object_points,
    const std::vector<cv::Point2d> & image_points,
    const cv::Vec3d & rvec,
    const cv::Vec3d & tvec,
    const cv::Matx33d & K,
    const std::vector<double> & D,
    ReprojectionStats & stats);

  // Utility: pretty-print stats to string (for UI overlay / logging)
  static std::string ToString(const ReprojectionStats & stats);
};

}  // namespace camera_field_calib_tool
