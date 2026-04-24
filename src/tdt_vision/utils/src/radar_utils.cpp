#include "radar_utils.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>

// 定义装甲板高度常量（单位：米）
#define ARMOR_HEIGHT 0.15

namespace tdt_radar {

namespace {

bool readMatrixFromNode(const cv::FileNode& node, cv::Mat& out)
{
    if (node.empty()) {
        return false;
    }

    node >> out;
    if (!out.empty()) {
        return true;
    }

    const cv::FileNode data = node["data"];
    if (data.empty() || !data.isSeq()) {
        return false;
    }

    std::vector<double> values;
    values.reserve(data.size());
    for (auto it = data.begin(); it != data.end(); ++it) {
        double value = 0.0;
        (*it) >> value;
        values.push_back(value);
    }
    if (values.empty()) {
        return false;
    }

    int rows = static_cast<int>(node["rows"]);
    int cols = static_cast<int>(node["cols"]);
    if (rows <= 0 || cols <= 0) {
        rows = 1;
        cols = static_cast<int>(values.size());
    }
    if (static_cast<size_t>(rows * cols) != values.size()) {
        return false;
    }

    out = cv::Mat(rows, cols, CV_64F);
    std::copy(values.begin(), values.end(),
              out.ptr<double>(0));
    return true;
}

bool readFirstAvailableMatrix(cv::FileStorage& fs,
                              cv::Mat& out,
                              std::initializer_list<const char*> keys)
{
    for (const char* key : keys) {
        if (readMatrixFromNode(fs[key], out)) {
            return true;
        }
    }
    return false;
}

cv::Mat makeCameraMatrix3x3(const cv::Mat& camera_matrix)
{
    cv::Mat K;
    camera_matrix.convertTo(K, CV_64F);
    if (K.rows == 3 && K.cols == 3) {
        return K;
    }
    throw std::runtime_error("camera_matrix must be 3x3");
}

void projectPointsNoDistortion(const std::vector<cv::Point3f>& world_points,
                               const cv::Mat& world_rvec,
                               const cv::Mat& world_tvec,
                               const cv::Mat& camera_matrix,
                               std::vector<cv::Point2f>& image_points)
{
    image_points.clear();
    image_points.reserve(world_points.size());

    cv::Mat R;
    cv::Rodrigues(world_rvec, R);
    R.convertTo(R, CV_64F);

    cv::Mat tvec;
    world_tvec.convertTo(tvec, CV_64F);
    if (tvec.rows == 1 && tvec.cols == 3) {
        tvec = tvec.t();
    }

    const cv::Mat K = makeCameraMatrix3x3(camera_matrix);
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    for (const auto& point : world_points) {
        cv::Mat pw = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
        cv::Mat pc = R * pw + tvec;
        const double z = pc.at<double>(2, 0);
        if (std::abs(z) < 1e-9) {
            image_points.emplace_back(
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN());
            continue;
        }

        const double x = pc.at<double>(0, 0) / z;
        const double y = pc.at<double>(1, 0) / z;
        image_points.emplace_back(
            static_cast<float>(fx * x + cx),
            static_cast<float>(fy * y + cy));
    }
}

}  // namespace

/**
 * @brief 判断2D点是否在屏幕范围内
 */
bool isPointInsideScreen(cv::Point2f point, int screenWidth, int screenHeight)
{
    return point.x >= 0 && point.x <= screenWidth && point.y >= 0 &&
           point.y <= screenHeight;
}

/**
 * @brief parser 构造：从参数化路径加载外参/内参/区域点集
 */
parser::parser(const std::string& out_matrix_path,
               const std::string& camera_params_path,
               const std::string& region_points_path)
: out_matrix_path_(out_matrix_path),
  camera_params_path_(camera_params_path),
  region_points_path_(region_points_path)
{
    // 读取世界坐标系到相机坐标系外参（旋转向量和平移向量）
    {
        cv::FileStorage fs;
        fs.open(out_matrix_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open out_matrix yaml: " +
                                     out_matrix_path_);
        }
        fs["world_tvec"] >> this->world_tvec;
        fs["world_rvec"] >> this->world_rvec;
        fs.release();
    }

    // 读取相机内参和畸变系数
    {
        cv::FileStorage fs1;
        fs1.open(camera_params_path_, cv::FileStorage::READ);
        if (!fs1.isOpened()) {
            throw std::runtime_error("Failed to open camera_params yaml: " +
                                     camera_params_path_);
        }
        if (!readFirstAvailableMatrix(fs1, this->camera_matrix,
                                      {"camera_matrix", "K"})) {
            throw std::runtime_error(
                "Missing camera_matrix/K in camera params: " +
                camera_params_path_);
        }
        if (!readFirstAvailableMatrix(fs1, this->dist_coeffs,
                                      {"dist_coeffs", "dist_coeff",
                                       "distortion_coefficients", "D"})) {
            this->dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
        }
        fs1.release();
    }

    // 从 YAML 读取区域列表（一级 key 作为区域名）
    YAML::Node root;
    try {
        root = YAML::LoadFile(region_points_path_);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to load region points yaml: ") +
                                 e.what() + " | path=" + region_points_path_);
    }

    if (!root || !root.IsMap()) {
        throw std::runtime_error("Region points yaml is empty or not a map: " +
                                 region_points_path_);
    }

    // 动态发现所有区域名，并构造 Parser_Points
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string region_name = it->first.as<std::string>();

        points_map[region_name] = new Parser_Points(
            region_name,
            out_matrix_path_,
            camera_params_path_,
            region_points_path_);

        // 高度自动从点集第一个点的 z 推断
        float inferred_height = 0.0f;
        if (!points_map[region_name]->Points_3D.empty()) {
            inferred_height = points_map[region_name]->Points_3D.front().z;
        }
        points_map[region_name]->Height = inferred_height;
    }
}

/**
 * @brief 释放 points_map 里的 new，避免泄漏
 */
parser::~parser()
{
    for (auto& kv : points_map) {
        delete kv.second;
        kv.second = nullptr;
    }
    points_map.clear();
}

cv::Point2f parser::parse_with_height(cv::Point2f& input_point, float height)
{
    // 绕过 get_height()，直接用外部给定高度
    return get_2d(input_point, height);
}

/**
 * @brief 更新外参，并刷新所有区域投影
 */
void parser::Change_Matrix()
{
    // 重新读取世界变换矩阵
    cv::FileStorage fs;
    fs.open(out_matrix_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open out_matrix yaml: " +
                                 out_matrix_path_);
    }
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    // 更新所有区域的点集投影
    for (auto& points : points_map) {
        points.second->Update();
    }
}

/**
 * @brief 绘制所有区域轮廓
 */
void parser::draw_ui(cv::Mat& img)
{
    for (auto& points : points_map) {
        cv::polylines(img, points.second->Points_2D, true,
                      cv::Scalar(255, 255, 255));
    }
}

/**
 * @brief 将图像2D点解析为世界坐标系2D点
 */
cv::Point2f parser::parse(cv::Point2f& input_point)
{
    float temp_height = get_height(input_point);

    // 你的原特殊逻辑保留
    if (temp_height > 0.79f) {
        return cv::Point2f(19.322f, -1.915f);
    }

    return get_2d(input_point, temp_height);
}

/**
 * @brief 获取点所在区域高度
 */
float parser::get_height(cv::Point2f& input_point)
{
    for (auto& points : points_map) {
        if (points.second->return_height(input_point)) {
            return points.second->Height;
        }
    }
    return 0.0f;
}

/**
 * @brief 图像2D点 -> 世界2D点（基于指定高度）
 */
cv::Point2f parser::get_2d(cv::Point2f& input_point, float height)
{
    // 4 个世界参考点（矩形），z = ARMOR_HEIGHT + height
    std::vector<cv::Point3f> world_points;
    world_points.emplace_back(12, -6, ARMOR_HEIGHT + height);
    world_points.emplace_back(16, -6, ARMOR_HEIGHT + height);
    world_points.emplace_back(16, -8, ARMOR_HEIGHT + height);
    world_points.emplace_back(12, -8, ARMOR_HEIGHT + height);

    // 投影到图像得到 2D 对应点
    std::vector<cv::Point2f> image_points;
    projectPointsNoDistortion(world_points, world_rvec, world_tvec,
                              camera_matrix, image_points);

    // 参考点在世界2D（忽略 z）
    std::vector<cv::Point2f> world_points2D;
    world_points2D.emplace_back(12, -6);
    world_points2D.emplace_back(16, -6);
    world_points2D.emplace_back(16, -8);
    world_points2D.emplace_back(12, -8);

    // 透视变换：image -> world2D
    cv::Mat perspective_matrix =
        cv::getPerspectiveTransform(image_points, world_points2D);

    cv::Mat srcPointMat(1, 1, CV_32FC2);
    srcPointMat.at<cv::Point2f>(0, 0) = input_point;
    cv::perspectiveTransform(srcPointMat, srcPointMat, perspective_matrix);

    return srcPointMat.at<cv::Point2f>(0, 0);
}

/**
 * @brief 你头文件里声明了这个重载，这里给一个“向后兼容”的实现：
 *        - 目前 get_2d 的几何映射只依赖 height（不依赖 region 形状）
 *        - region_name 主要用于你将来如果要 region-specific 标定/矩形参数
 */
cv::Point2f parser::get_2d(cv::Point2f& input_point,
                           float height,
                           const std::string& /*region_name*/)
{
    return get_2d(input_point, height);
}

/* ========================== Parser_Points ========================== */

std::vector<cv::Point3f>
Parser_Points::ReadPoints(const std::string& points_name)
{
    cv::FileStorage fs(region_points_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open region points yaml: " +
                                 region_points_path_);
    }

    std::vector<cv::Point3f> points;

    cv::FileNode pointsNode = fs[points_name];
    if (pointsNode.type() != cv::FileNode::SEQ) {
        throw std::runtime_error("Points node is not a sequence: " +
                                 points_name + " | file=" + region_points_path_);
    }

    for (auto&& it : pointsNode) {
        cv::Point3f p;
        it["x"] >> p.x;
        it["y"] >> p.y;
        it["z"] >> p.z;
        points.push_back(p);
    }
    return points;
}

std::vector<cv::Point>
Parser_Points::Float2Int(std::vector<cv::Point2f>& FloatPoint)
{
    std::vector<cv::Point> dstPoint;
    dstPoint.reserve(FloatPoint.size());
    for (auto& i : FloatPoint) {
        dstPoint.emplace_back(int(i.x), int(i.y));
    }
    return dstPoint;
}

void Parser_Points::World2Camera()
{
    std::vector<cv::Point2f> temp_2D;
    projectPointsNoDistortion(Points_3D, world_rvec, world_tvec,
                              camera_matrix, temp_2D);
    Points_2D = Float2Int(temp_2D);
}

Parser_Points::Parser_Points(const std::string& points_name,
                             const std::string& out_matrix_path,
                             const std::string& camera_params_path,
                             const std::string& region_points_path)
: out_matrix_path_(out_matrix_path),
  camera_params_path_(camera_params_path),
  region_points_path_(region_points_path)
{
    // 读取相机内参和畸变系数
    {
        cv::FileStorage fs;
        fs.open(camera_params_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open camera_params yaml: " +
                                     camera_params_path_);
        }
        if (!readFirstAvailableMatrix(fs, this->camera_matrix,
                                      {"camera_matrix", "K"})) {
            throw std::runtime_error(
                "Missing camera_matrix/K in camera params: " +
                camera_params_path_);
        }
        if (!readFirstAvailableMatrix(fs, this->dist_coeffs,
                                      {"dist_coeffs", "dist_coeff",
                                       "distortion_coefficients", "D"})) {
            this->dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
        }
        fs.release();
    }

    // 读取外参
    {
        cv::FileStorage fs;
        fs.open(out_matrix_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open out_matrix yaml: " +
                                     out_matrix_path_);
        }
        fs["world_tvec"] >> this->world_tvec;
        fs["world_rvec"] >> this->world_rvec;
        fs.release();
    }

    // 读取点集并投影
    this->Points_3D = ReadPoints(points_name);
    World2Camera();
}

float Parser_Points::return_height(cv::Point2f& input_point)
{
    if (cv::pointPolygonTest(Points_2D,
                             cv::Point((int)input_point.x, (int)input_point.y),
                             false) > 0) {
        return this->Height;
    }
    return 0.0f;
}

void Parser_Points::Update()
{
    cv::FileStorage fs;
    fs.open(out_matrix_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open out_matrix yaml: " +
                                 out_matrix_path_);
    }
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    World2Camera();
}

}  // namespace tdt_radar
