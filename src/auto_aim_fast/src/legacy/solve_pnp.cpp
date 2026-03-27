#include "auto_aim_fast/legacy/solve_pnp.hpp"
#include "auto_aim_fast/legacy/detector.hpp"

#include <cmath>
#include <opencv2/calib3d.hpp>

namespace {

inline void pairResultToCorners(const SimpleLightDetector::PairResult& pair,
                                cv::Point2f& left_top,
                                cv::Point2f& left_bottom,
                                cv::Point2f& right_top,
                                cv::Point2f& right_bottom)
{
    if (pair.has_precise_corners) {
        left_top = pair.precise_left_top;
        left_bottom = pair.precise_left_bottom;
        right_top = pair.precise_right_top;
        right_bottom = pair.precise_right_bottom;
        return;
    }

    const float half_w = pair.width * 0.5f;
    const float half_h = pair.height * 0.5f;

    const float c = std::cos(pair.angle);
    const float s = std::sin(pair.angle);

    auto rot = [&](float x, float y) -> cv::Point2f {
        return cv::Point2f(pair.center.x + c * x - s * y,
                           pair.center.y + s * x + c * y);
    };

    left_top     = rot(-half_w, -half_h);
    right_top    = rot( half_w, -half_h);
    right_bottom = rot( half_w,  half_h);
    left_bottom  = rot(-half_w,  half_h);
}

}  // namespace

LaserPnP::LaserPnP(const cv::Mat& K, const cv::Mat& D)
{
    K_ = K.clone();
    D_ = D.clone();

    const float target_width_m  = 0.0515f;
    const float target_height_m = 0.05f;

    const float half_w = target_width_m * 0.5f;
    const float half_h = target_height_m * 0.5f;

    object_pts_ = {
        cv::Point3f(-half_w, -half_h, 0.0f),
        cv::Point3f(-half_w,  half_h, 0.0f),
        cv::Point3f( half_w,  half_h, 0.0f),
        cv::Point3f( half_w, -half_h, 0.0f)
    };
}

bool LaserPnP::solve(const PairResult& pair, cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Point2f lt, lb, rt, rb;
    pairResultToCorners(pair, lt, lb, rt, rb);
    return solve(lt, lb, rt, rb, rvec, tvec);
}

bool LaserPnP::solve(const cv::Point2f& left_top,
                     const cv::Point2f& left_bottom,
                     const cv::Point2f& right_top,
                     const cv::Point2f& right_bottom,
                     cv::Mat& rvec,
                     cv::Mat& tvec)
{
    std::vector<cv::Point2f> image_pts = {
        left_top,
        left_bottom,
        right_bottom,
        right_top
    };

    return cv::solvePnP(object_pts_, image_pts, K_, D_, rvec, tvec);
}
