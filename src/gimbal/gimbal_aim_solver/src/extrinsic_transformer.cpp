#include "gimbal_aim_solver/extrinsic_transformer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace gimbal_aim_solver
{

bool ExtrinsicTransformer::init(const std::vector<double>& rvec_data,
                                const std::vector<double>& tvec_data)
{
  if (rvec_data.size() != 3 || tvec_data.size() != 3) {
    initialized_ = false;
    return false;
  }

  cv::Mat rvec = (cv::Mat_<double>(3, 1)
    << rvec_data[0], rvec_data[1], rvec_data[2]);

  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  R_cw_ <<
    R_cv.at<double>(0, 0), R_cv.at<double>(0, 1), R_cv.at<double>(0, 2),
    R_cv.at<double>(1, 0), R_cv.at<double>(1, 1), R_cv.at<double>(1, 2),
    R_cv.at<double>(2, 0), R_cv.at<double>(2, 1), R_cv.at<double>(2, 2);

  t_cw_ << tvec_data[0], tvec_data[1], tvec_data[2];

  initialized_ = true;
  return true;
}

Eigen::Vector3d ExtrinsicTransformer::worldToCamera(const Eigen::Vector3d& p_world) const
{
  return R_cw_ * p_world + t_cw_;
}

}  // namespace gimbal_aim_solver