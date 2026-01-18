#include "auto_aim_fast/legacy/solve_pnp.hpp"  // 再包含自己的头文件
#include "auto_aim_fast/legacy/detector.hpp"  // 先包含定义 PairResult 的头文件

#include <cmath>
#include <opencv2/calib3d.hpp>

namespace {

inline void pairResultToCorners(const SimpleLightDetector::PairResult& pair,
                                cv::Point2f& left_top,
                                cv::Point2f& left_bottom,
                                cv::Point2f& right_top,
                                cv::Point2f& right_bottom)
{
    float h = pair.width * 0.5f;
    float w = pair.height * 0.5f;

    float theta = pair.angle;  // 如果是角度，这里统一转弧度
    float c = std::cos(theta);
    float s = std::sin(theta);

    // 修复 lambda 函数：添加明确的返回类型
    auto rot = [&](float x, float y) -> cv::Point2f {
        return cv::Point2f(pair.center.x + c * x - s * y,
                           pair.center.y + s * x + c * y);
    };

    left_bottom = rot(-w, -h);
    left_top = rot(-w, h);
    right_top = rot(w, h);
    right_bottom = rot(w, -h);
}

}  // namespace

LaserPnP::LaserPnP(const cv::Mat& K, const cv::Mat& D)
{
    K_ = K.clone();
    D_ = D.clone();

    const float module_height_m = 0.0499f;
    const float half_L = 0.09602f / 2.0f;  // 灯条半长（水平）
    const float half_H = module_height_m / 2.0f;  // 上下灯条间距

    // 原点 = 激光模块中心
    object_pts_ = {
        {0, -half_L, -half_H},  // 左下
        {0, -half_L, half_H},   // 左上
        {0, half_L, half_H},    // 右上
        {0, half_L, -half_H}    // 右下
    };
}

bool LaserPnP::solve(const PairResult& pair, cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Point2f lt, lb, rt, rb;
    pairResultToCorners(pair, lt, lb, rt, rb);

    return solve(lt, lb, rt, rb, rvec, tvec);
}

// 添加另一个 solve 方法的实现
bool LaserPnP::solve(const cv::Point2f& left_top,
                     const cv::Point2f& left_bottom,
                     const cv::Point2f& right_top,
                     const cv::Point2f& right_bottom, cv::Mat& rvec,
                     cv::Mat& tvec)
{
    std::vector<cv::Point2f> image_pts = {
        left_bottom,  // 对应 object_pts_[0]
        left_top,     // 对应 object_pts_[1]
        right_top,    // 对应 object_pts_[2]
        right_bottom  // 对应 object_pts_[3]
    };

    // 使用 solvePnP 进行位姿求解
    return cv::solvePnP(object_pts_, image_pts, K_, D_, rvec, tvec);
}