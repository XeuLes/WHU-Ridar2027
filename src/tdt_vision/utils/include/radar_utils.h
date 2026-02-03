#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace tdt_radar {

// -----------------------------------------------------------------------------
// Parser_Points: one region's polygon points (3D) + projected 2D contour.
// Now supports parameterized config paths.
// -----------------------------------------------------------------------------
class Parser_Points {
public:
    // 新构造：传入三个配置文件路径（ROS2 参数化入口）
    Parser_Points(const std::string& points_name,
                  const std::string& out_matrix_path,
                  const std::string& camera_params_path,
                  const std::string& region_points_path);

    // 保留接口
    float return_height(cv::Point2f& input_point);
    void  Update();
    void  World2Camera();

    std::vector<cv::Point3f> ReadPoints(const std::string& points_name);
    std::vector<cv::Point>   Float2Int(std::vector<cv::Point2f>& FloatPoint);

    // 与你现有 cpp 保持兼容：点集与高度仍公开
    std::vector<cv::Point3f> Points_3D;
    std::vector<cv::Point>   Points_2D;
    float                    Height = 0.0f;

    // 禁拷贝（points_map 里是裸指针，拷贝没有意义且危险）
    Parser_Points(const Parser_Points&)            = delete;
    Parser_Points& operator=(const Parser_Points&) = delete;

private:
    // 三个路径（由 ROS2 参数传入）
    std::string out_matrix_path_;
    std::string camera_params_path_;
    std::string region_points_path_;

    // 内外参
    cv::Mat world_rvec;
    cv::Mat world_tvec;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

// -----------------------------------------------------------------------------
// parser: manages all regions; parses an image point to world 2D using inferred height.
// Now supports parameterized config paths.
// -----------------------------------------------------------------------------
class parser {
public:
    // 新构造：传入三个配置文件路径（ROS2 参数化入口）
    parser(const std::string& out_matrix_path,
           const std::string& camera_params_path,
           const std::string& region_points_path);

    // 释放 points_map 里 new 出来的 Parser_Points，避免内存泄漏
    ~parser();

    // 禁拷贝（持有裸指针）
    parser(const parser&)            = delete;
    parser& operator=(const parser&) = delete;

    // 保留接口
    void        Change_Matrix();
    cv::Point2f parse(cv::Point2f& input_point);
    cv::Point2f parse_with_height(cv::Point2f& input_point, float height);

    void        draw_ui(cv::Mat& img);
    float       get_height(cv::Point2f& input_point);

    cv::Point2f get_2d(cv::Point2f& input_point, float height);

    // 你原本声明过的重载：支持指定标定点集名称
    cv::Point2f get_2d(cv::Point2f& input_point,
                       float height,
                       const std::string& region_name);

    // 与你现有 cpp 保持兼容：这些仍公开
    cv::Mat world_rvec;
    cv::Mat world_tvec;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    std::map<std::string, Parser_Points*> points_map;

private:
    // 三个路径（由 ROS2 参数传入）
    std::string out_matrix_path_;
    std::string camera_params_path_;
    std::string region_points_path_;
};

}  // namespace tdt_radar

#endif  // RADAR_UTILS_H
