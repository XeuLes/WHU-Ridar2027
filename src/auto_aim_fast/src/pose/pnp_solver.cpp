#include "auto_aim_fast/pose/pnp_solver.hpp"
#include "auto_aim_fast/legacy/legacy_api.hpp"

#include <opencv2/core.hpp>
#include <memory>

namespace auto_aim_fast::pose
{

static float readVec3(const cv::Mat & v, int idx)
{
  if (v.empty()) return 0.0f;

  if (v.rows == 3 && v.cols == 1) {
    if (v.type() == CV_64F) return static_cast<float>(v.at<double>(idx, 0));
    if (v.type() == CV_32F) return v.at<float>(idx, 0);
  }
  if (v.rows == 1 && v.cols == 3) {
    if (v.type() == CV_64F) return static_cast<float>(v.at<double>(0, idx));
    if (v.type() == CV_32F) return v.at<float>(0, idx);
  }

  cv::Mat vv = v.reshape(1, 1);
  if (vv.cols >= 3) {
    if (vv.type() == CV_64F) return static_cast<float>(vv.at<double>(0, idx));
    if (vv.type() == CV_32F) return vv.at<float>(0, idx);
  }
  return 0.0f;
}

struct PnPSolver::Impl
{
  cv::Mat K;
  cv::Mat D;
  std::unique_ptr<LaserPnP> pnp;

  Impl()
  {
    K = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0);
    D = cv::Mat::zeros(1, 5, CV_64F);
    pnp = std::make_unique<LaserPnP>(K, D);
  }
};

PnPSolver::PnPSolver()
: impl_(std::make_unique<Impl>())
{
}

PnPSolver::~PnPSolver() = default;

void PnPSolver::setCamera(const cv::Mat & K, const cv::Mat & D)
{
  impl_->K = K.clone();
  impl_->D = D.clone();
  impl_->pnp = std::make_unique<LaserPnP>(impl_->K, impl_->D);
}

Pose3D PnPSolver::solve(const auto_aim_fast::vision::Armor2D & armor) const
{
  Pose3D out;

  // Always pass 2D center to tracker (ROI anchor), even if PnP fails
  out.has_img_center = true;
  out.img_cx = armor.center.x;
  out.img_cy = armor.center.y;

  if (!armor.valid) {
    out.ok = false;
    out.score = 0.0f;
    return out;
  }

  cv::Mat rvec, tvec;
  bool ok = false;

  if (armor.has_legacy_pair) {
    ok = impl_->pnp->solve(armor.legacy_pair, rvec, tvec);
  } else {
    const cv::Point2f & LT = armor.corners[0];
    const cv::Point2f & RT = armor.corners[1];
    const cv::Point2f & RB = armor.corners[2];
    const cv::Point2f & LB = armor.corners[3];
    ok = impl_->pnp->solve(LT, LB, RT, RB, rvec, tvec);
  }

  if (!ok) {
    out.ok = false;
    out.score = armor.score;
    return out;
  }

  out.ok = true;
  out.x = readVec3(tvec, 0);
  out.y = readVec3(tvec, 1);
  out.z = readVec3(tvec, 2);
  out.score = armor.score;

  return out;
}

}  // namespace auto_aim_fast::pose
