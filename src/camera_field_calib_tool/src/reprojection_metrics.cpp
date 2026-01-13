#include "camera_field_calib_tool/reprojection_metrics.hpp"

#include <cmath>
#include <sstream>

#include <opencv2/calib3d.hpp>

namespace camera_field_calib_tool
{

bool ReprojectionMetrics::Compute(
  const std::vector<cv::Point3d> & object_points,
  const std::vector<cv::Point2d> & image_points,
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec,
  const cv::Matx33d & K,
  const std::vector<double> & D,
  ReprojectionStats & stats)
{
  const size_t n = object_points.size();
  if (n == 0 || image_points.size() != n) {
    return false;
  }

  std::vector<cv::Point2d> projected;
  projected.reserve(n);

  cv::Mat K_mat(K);
  cv::Mat D_mat(D);

  try {
    cv::projectPoints(
      object_points,
      rvec,
      tvec,
      K_mat,
      D_mat,
      projected);
  } catch (...) {
    return false;
  }

  double sum_sq = 0.0;
  double max_err = 0.0;

  stats.per_point_errors.clear();
  stats.per_point_errors.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const double dx = projected[i].x - image_points[i].x;
    const double dy = projected[i].y - image_points[i].y;
    const double err = std::sqrt(dx * dx + dy * dy);

    stats.per_point_errors.push_back(err);
    sum_sq += err * err;
    if (err > max_err) {
      max_err = err;
    }
  }

  stats.rmse = std::sqrt(sum_sq / static_cast<double>(n));
  stats.max_error = max_err;

  return true;
}

std::string ReprojectionMetrics::ToString(const ReprojectionStats & stats)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  oss << "RMSE: " << stats.rmse << " px, "
      << "Max: " << stats.max_error << " px";
  return oss.str();
}

}  // namespace camera_field_calib_tool
