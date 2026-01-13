#include "camera_field_calib_tool/camera_model.hpp"

#include <opencv2/calib3d.hpp>

namespace camera_field_calib_tool
{

bool CameraModel::SetIntrinsics(
  const cv::Matx33d & K,
  const std::vector<double> & D,
  int width,
  int height)
{
  K_ = K;
  D_ = D;
  width_ = width;
  height_ = height;

  // Basic sanity check
  if (K_(0, 0) <= 0.0 || K_(1, 1) <= 0.0) {
    valid_ = false;
    return false;
  }

  valid_ = true;
  return true;
}

bool CameraModel::IsValid() const
{
  return valid_;
}

cv::Vec3d CameraModel::PixelToNormalizedRay(
  const cv::Point2d & pixel) const
{
  // Assumes camera model is valid
  // Use OpenCV undistortPoints to get normalized image coordinates
  std::vector<cv::Point2d> src{pixel};
  std::vector<cv::Point2d> dst;

  cv::Mat K_mat(K_);
  cv::Mat D_mat(D_);

  cv::undistortPoints(
    src,
    dst,
    K_mat,
    D_mat,
    cv::noArray(),
    cv::noArray());

  // undistortPoints returns normalized coordinates (x,y)
  // corresponding to ray direction (x, y, 1)
  return cv::Vec3d(dst[0].x, dst[0].y, 1.0);
}

std::vector<cv::Vec3d> CameraModel::PixelToNormalizedRays(
  const std::vector<cv::Point2d> & pixels) const
{
  std::vector<cv::Vec3d> rays;
  rays.reserve(pixels.size());

  if (pixels.empty()) {
    return rays;
  }

  std::vector<cv::Point2d> dst;

  cv::Mat K_mat(K_);
  cv::Mat D_mat(D_);

  cv::undistortPoints(
    pixels,
    dst,
    K_mat,
    D_mat,
    cv::noArray(),
    cv::noArray());

  for (const auto & p : dst) {
    rays.emplace_back(p.x, p.y, 1.0);
  }

  return rays;
}

cv::Point2d CameraModel::CameraToPixelNoDistortion(
  const cv::Point3d & pc) const
{
  // Simple pinhole projection without distortion
  // u = fx * x / z + cx
  // v = fy * y / z + cy
  const double inv_z = 1.0 / pc.z;

  const double u = K_(0, 0) * pc.x * inv_z + K_(0, 2);
  const double v = K_(1, 1) * pc.y * inv_z + K_(1, 2);

  return cv::Point2d(u, v);
}

bool CameraModel::IsPixelInside(const cv::Point2d & pixel) const
{
  if (width_ <= 0 || height_ <= 0) {
    return true;  // unknown bounds, assume valid
  }
  return pixel.x >= 0.0 && pixel.y >= 0.0 &&
         pixel.x < static_cast<double>(width_) &&
         pixel.y < static_cast<double>(height_);
}

}  // namespace camera_field_calib_tool
