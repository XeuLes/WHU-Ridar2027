#pragma once
#include "auto_aim_fast/legacy/detector.hpp"
#include <opencv2/opencv.hpp>



class LaserPnP
{
public:
using PairResult = SimpleLightDetector::PairResult;
    LaserPnP(const cv::Mat &K,
             const cv::Mat &D);

    bool solve(const PairResult &pair,
               cv::Mat &rvec,
               cv::Mat &tvec);

    bool solve(const cv::Point2f &left_top,
               const cv::Point2f &left_bottom,
               const cv::Point2f &right_top,
               const cv::Point2f &right_bottom,
               cv::Mat &rvec,
               cv::Mat &tvec);

private:
    cv::Mat K_;
    cv::Mat D_;
    std::vector<cv::Point3f> object_pts_;
};
