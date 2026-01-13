#include "camera_field_calib_tool/pnp_calibrator.hpp"

#include <opencv2/calib3d.hpp>

namespace camera_field_calib_tool
{

void PnPCalibrator::SetCameraModel(const CameraModel & cam)
{
  cam_ = cam;
  has_cam_ = cam_.IsValid();
}

void PnPCalibrator::Reset()
{
  object_pts_.clear();
  image_pts_.clear();
}

size_t PnPCalibrator::Size() const
{
  return object_pts_.size();
}

const std::vector<cv::Point3d> & PnPCalibrator::ObjectPoints() const
{
  return object_pts_;
}

const std::vector<cv::Point2d> & PnPCalibrator::ImagePoints() const
{
  return image_pts_;
}

void PnPCalibrator::AddPoint(const cv::Point3d & object_pt, const cv::Point2d & image_pt)
{
  object_pts_.push_back(object_pt);
  image_pts_.push_back(image_pt);
}

bool PnPCalibrator::Undo()
{
  if (object_pts_.empty()) return false;
  object_pts_.pop_back();
  image_pts_.pop_back();
  return true;
}

int PnPCalibrator::RecommendedMinPoints(int method)
{
  // Conservative guidance. EPnP works with >=4, but for stability usually >=6.
  // ITERATIVE requires >=4 (non-coplanar preferred), but again use >=6.
  (void)method;
  return 6;
}

bool PnPCalibrator::ComputeReprojection(
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec,
  ReprojectionStats & stats) const
{
  if (!has_cam_) return false;

  return ReprojectionMetrics::Compute(
    object_pts_,
    image_pts_,
    rvec,
    tvec,
    cam_.K(),
    cam_.D(),
    stats);
}

PnPResult PnPCalibrator::Solve(int method, bool use_refine) const
{
  PnPResult result;
  result.success = false;

  if (!has_cam_) {
    return result;
  }
  if (object_pts_.size() < 4 || image_pts_.size() != object_pts_.size()) {
    return result;
  }

  // OpenCV solvePnP expects std::vector<Point3f/Point2f> or doubles; we use double.
  cv::Mat K_mat(cam_.K());
  cv::Mat D_mat(cam_.D());

  cv::Vec3d rvec(0.0, 0.0, 0.0);
  cv::Vec3d tvec(0.0, 0.0, 0.0);

  // Default method if not specified
  if (method == 0) {
    method = cv::SOLVEPNP_EPNP;
  }

  bool ok = false;
  try {
    ok = cv::solvePnP(
      object_pts_,
      image_pts_,
      K_mat,
      D_mat,
      rvec,
      tvec,
      false,
      method);
  } catch (...) {
    ok = false;
  }

  if (!ok) {
    return result;
  }

  // Optional refine (LM)
  if (use_refine) {
    try {
#if CV_VERSION_MAJOR >= 4
      // solvePnPRefineLM signature exists in OpenCV 4+
      cv::solvePnPRefineLM(
        object_pts_,
        image_pts_,
        K_mat,
        D_mat,
        rvec,
        tvec);
#endif
    } catch (...) {
      // ignore refine failure
    }
  }

  // Reprojection stats
  ReprojectionStats stats;
  (void)ComputeReprojection(rvec, tvec, stats);

  result.success = true;
  result.rvec = rvec;
  result.tvec = tvec;
  result.reproj = stats;

  return result;
}

}  // namespace camera_field_calib_tool
