#include "radar_utils.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

// -----------------------------------------------------------------------------
// Fixed config paths (same idea as out_matrix): keep them local and
// explicit.
// -----------------------------------------------------------------------------
namespace {
static const std::string kOutMatrixPath = "./config/out_matrix.yaml";
static const std::string kCameraParamsPath = "./config/camera_params.yaml";
static const std::string kRegionPointsPath = "./config/RM2025_Points.yaml";
}  // namespace

// 定义装甲板高度常量（单位：米）
#define ARMOR_HEIGHT 0.15

namespace tdt_radar {

/**
 * @brief 判断2D点是否在屏幕范围内
 * @param point 待判断的2D点坐标
 * @param screenWidth 屏幕宽度
 * @param screenHeight 屏幕高度
 * @return 若点在屏幕内（含边界）返回true，否则返回false
 */
bool isPointInsideScreen(cv::Point2f point, int screenWidth,
                         int screenHeight)
{
    return point.x >= 0 && point.x <= screenWidth && point.y >= 0 &&
           point.y <= screenHeight;
}

/**
 * @brief
 * parser类构造函数：初始化坐标解析器，加载相机参数、世界变换矩阵和区域点集
 */
parser::parser()
{
    // 读取世界坐标系到相机坐标系的变换矩阵（旋转向量和平移向量）
    cv::FileStorage fs;
    fs.open(kOutMatrixPath, cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;  // 平移向量
    fs["world_rvec"] >> this->world_rvec;  // 旋转向量
    std::cout << world_tvec << std::endl;  // 打印平移向量（调试用）
    std::cout << world_rvec << std::endl;  // 打印旋转向量（调试用）
    fs.release();

    // 读取相机内参和畸变系数
    cv::FileStorage fs1;
    fs1.open(kCameraParamsPath, cv::FileStorage::READ);
    fs1["camera_matrix"] >> this->camera_matrix;  // 相机内参矩阵
    fs1["dist_coeffs"] >> this->dist_coeffs;      // 畸变系数
    fs1.release();
    std::cout << "Resolve camera" << camera_matrix
              << std::endl;  // 打印相机内参（调试用）
    std::cout << "Resolve dist" << dist_coeffs
              << std::endl;  // 打印畸变系数（调试用）

    // ---------------------------------------------------------------------
    // 从 YAML 读取区域列表（不再在 CPP 里写死区域名/高度）。
    // 规则：
    //   - 区域名 = RM2025_Points.yaml 的一级 key
    //   - 区域高度 = 该区域点集里第一个点的 z（若没有点则为 0）
    //     这样你在编辑/保存点集时，只需要把正确的 z 写进
    //     YAML，即可自动生效。
    // ---------------------------------------------------------------------
    YAML::Node root;
    try {
        root = YAML::LoadFile(kRegionPointsPath);
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Failed to load region points yaml: ") + e.what());
    }

    if (!root || !root.IsMap()) {
        throw std::runtime_error(
            "Region points yaml is empty or not a map: " +
            kRegionPointsPath);
    }

    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string region_name = it->first.as<std::string>();

        // 构造时 Parser_Points 内部仍然读取点集并做投影；
        // 这里负责动态发现 key（区域名）并根据点集的 z 自动设置高度。
        points_map[region_name] = new Parser_Points(region_name);

        float inferred_height = 0.0f;
        if (!points_map[region_name]->Points_3D.empty()) {
            inferred_height = points_map[region_name]->Points_3D.front().z;
        }
        points_map[region_name]->Height = inferred_height;
    }
}

cv::Point2f parser::parse_with_height(cv::Point2f& input_point,
                                      float        height)
{
    // 核心：绕过 get_height()，直接用外部给定高度
    return get_2d(input_point, height);
}

/**
 * @brief 更新世界变换矩阵，并刷新所有区域的点集投影
 * 作用：当外参（旋转/平移向量）变化时，重新加载并更新所有区域的2D投影点
 */
void parser::Change_Matrix()
{
    // 重新读取世界变换矩阵
    cv::FileStorage fs;
    fs.open(kOutMatrixPath, cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    // 更新所有区域的点集（重新计算3D到2D的投影）
    for (auto& points : points_map) {
        points.second->Update();
    }
}

/**
 * @brief 在图像上绘制所有区域的2D轮廓
 * @param img 待绘制的图像
 */
void parser::draw_ui(cv::Mat& img)
{
    for (auto& points : points_map) {
        // 绘制闭合多边形（白色）
        cv::polylines(img, points.second->Points_2D, true,
                      cv::Scalar(255, 255, 255));
    }
}

/**
 * @brief 将图像2D点解析为世界坐标系2D点
 * @param input_point 图像上的2D点
 * @return 世界坐标系中的2D点（x,y）
 */
cv::Point2f parser::parse(cv::Point2f& input_point)
{
    // 获取输入点所在区域的高度
    float temp_height = get_height(input_point);

    // 特殊高度判断（可能对应无效区域或特殊处理）
    if (temp_height > 0.79) {
        return cv::Point2f(19.322, -1.915);  // 返回固定坐标
    }

    // 根据高度将图像点转换为世界2D点
    return get_2d(input_point, temp_height);
}

/**
 * @brief 获取输入点所在区域的高度
 * @param input_point 图像上的2D点
 * @return 区域高度（若点不在任何区域，返回0）
 */
float parser::get_height(cv::Point2f& input_point)
{
    // 遍历所有区域，判断点是否在区域内
    for (auto& points : points_map) {
        if (points.second->return_height(input_point)) {
            return points.second->Height;  // 返回该区域的高度
        }
    }
    return 0;  // 不在任何区域，返回0
}

/**
 * @brief 将图像2D点转换为世界坐标系2D点（基于指定高度）
 * @param input_point 图像上的2D点
 * @param height 点所在区域的高度（用于3D坐标构建）
 * @return 世界坐标系中的2D点（x,y）
 */
cv::Point2f parser::get_2d(cv::Point2f& input_point, float height)
{
    // 定义世界坐标系中的4个参考点（用于计算透视变换矩阵）
    // 这些点构成一个矩形，z坐标为：装甲板高度 + 区域高度
    std::vector<cv::Point3f> world_points;
    world_points.push_back(cv::Point3f(12, -6, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(16, -6, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(16, -8, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(12, -8, ARMOR_HEIGHT + height));

    // 将世界参考点投影到图像上，得到对应的2D图像点
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(world_points, world_rvec, world_tvec, camera_matrix,
                      dist_coeffs, image_points);

    // 定义世界参考点的2D坐标（忽略z轴）
    std::vector<cv::Point2f> world_points2D;
    world_points2D.push_back(cv::Point2f(12, -6));
    world_points2D.push_back(cv::Point2f(16, -6));
    world_points2D.push_back(cv::Point2f(16, -8));
    world_points2D.push_back(cv::Point2f(12, -8));

    // 计算从图像点到世界2D点的透视变换矩阵
    cv::Mat Perspective_matrix =
        cv::getPerspectiveTransform(image_points, world_points2D);

    // 应用透视变换，将输入点转换为世界2D点
    cv::Mat srcPointMat(1, 1, CV_32FC2);  // 包装输入点为矩阵形式
    srcPointMat.at<cv::Point2f>(0, 0) = input_point;
    cv::perspectiveTransform(srcPointMat, srcPointMat,
                             Perspective_matrix);  // 执行变换

    std::cout << "[get_2d] output_point: (" << srcPointMat.at<cv::Point2f>(0, 0).x << ", " << srcPointMat.at<cv::Point2f>(0, 0).y << ")" << std::endl;

    return srcPointMat.at<cv::Point2f>(0, 0);  // 返回转换后的世界2D点
}

/**
 * @brief 从YAML文件读取指定名称的3D点集
 * @param points_name 点集名称（对应YAML中的节点）
 * @return 读取到的3D点集
 */
std::vector<cv::Point3f>
Parser_Points::ReadPoints(const std::string& points_name)
{
    // 打开存储点集的YAML文件
    cv::FileStorage fs(kRegionPointsPath, cv::FileStorage::READ);

    if (!fs.isOpened()) {  // 检查文件是否打开成功
        std::cout << "无法打开文件" << std::endl;
        exit(-1);  // 打开失败则退出程序
    }

    std::vector<cv::Point3f> points;  // 存储读取的3D点

    cv::FileNode pointsNode = fs[points_name];  // 获取对应名称的节点
    if (pointsNode.type() != cv::FileNode::SEQ) {  // 检查节点是否为序列类型
        std::cout << "points节点不是序列" << std::endl;
        exit(-1);
    }

    // 遍历节点，读取每个点的x、y、z坐标
    for (auto&& it : pointsNode) {
        cv::Point3f point;
        it["x"] >> point.x;
        it["y"] >> point.y;
        it["z"] >> point.z;

        points.push_back(point);
    }
    return points;
}

/**
 * @brief 将浮点型2D点转换为整型2D点（用于绘制）
 * @param FloatPoint 浮点型2D点集
 * @return 整型2D点集
 */
std::vector<cv::Point>
Parser_Points::Float2Int(std::vector<cv::Point2f>& FloatPoint)
{
    std::vector<cv::Point> dstPoint;
    for (auto& i : FloatPoint) {
        dstPoint.emplace_back(int(i.x), int(i.y));  // 类型转换
    }
    return dstPoint;
}

/**
 * @brief 将3D世界点投影到2D图像上（考虑相机内外参）
 */
void Parser_Points::World2Camera()
{
    std::vector<cv::Point2f> temp_2D;
    // 投影3D点到图像：world_rvec/world_tvec为外参，camera_matrix/dist_coeffs为内参
    cv::projectPoints(Points_3D, world_rvec, world_tvec, camera_matrix,
                      dist_coeffs, temp_2D);
    Points_2D = Float2Int(temp_2D);  // 转换为整型点，用于后续绘制
}

/**
 * @brief Parser_Points类构造函数：初始化区域点集，加载参数并投影到2D图像
 * @param points_name 区域名称（对应YAML中的点集）
 */
Parser_Points::Parser_Points(const std::string& points_name)
{
    // 读取相机内参和畸变系数
    cv::FileStorage fs;
    fs.open(kCameraParamsPath, cv::FileStorage::READ);
    fs["camera_matrix"] >> this->camera_matrix;
    fs["dist_coeffs"] >> this->dist_coeffs;
    fs.release();

    // 读取世界到相机的变换矩阵
    fs.open(kOutMatrixPath, cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    // 读取该区域的3D点集并投影到2D图像
    std::vector<cv::Point3f> temp_3d = ReadPoints(points_name);
    this->Points_3D = temp_3d;
    World2Camera();  // 计算3D点对应的2D图像点
}

/**
 * @brief 判断输入点是否在当前区域内，返回区域高度
 * @param input_point 图像上的2D点
 * @return 若点在区域内返回区域高度，否则返回0
 */
float Parser_Points::return_height(cv::Point2f& input_point)
{
    // 使用pointPolygonTest判断点是否在多边形内（Points_2D为区域轮廓）
    if (cv::pointPolygonTest(
            Points_2D, cv::Point((int)input_point.x, (int)input_point.y),
            false) > 0) {
        return this->Height;  // 在区域内，返回高度
    } else {
        return 0;  // 不在区域内，返回0
    }
}

/**
 * @brief 更新变换矩阵并重新计算3D点到2D图像的投影
 */
void Parser_Points::Update()
{
    // 重新读取世界变换矩阵
    cv::FileStorage fs;
    fs.open(kOutMatrixPath, cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    // 重新投影3D点到2D图像
    World2Camera();
}

}  // namespace tdt_radar
