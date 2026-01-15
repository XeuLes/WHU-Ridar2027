#include "auto_aim_fast/legacy/detector.hpp" // 确保包含正确的头文件
#include <algorithm>
#include <cmath>
#include <numeric> // For std::accumulate if needed


// 静态成员初始化
const cv::Scalar SimpleLightDetector::RED_COLOR = cv::Scalar(0, 0, 255);
const cv::Scalar SimpleLightDetector::BLUE_COLOR = cv::Scalar(255, 0, 0);
const cv::Scalar SimpleLightDetector::GREEN_COLOR = cv::Scalar(0, 255, 0);
const cv::Scalar SimpleLightDetector::YELLOW_COLOR = cv::Scalar(0, 255, 255);

// ==================== Light 结构体方法实现 ====================

SimpleLightDetector::Light::Light(const cv::RotatedRect& r)
    : rect(r),
      center(r.center),
      width(std::min(r.size.width, r.size.height)),
      length(std::max(r.size.width, r.size.height)),
      area(r.size.width * r.size.height) {

    // 计算角度（确保角度在[0, 90)范围内）
    angle = r.size.width < r.size.height ? r.angle : r.angle - 90;
    if (angle < -90) angle += 180;
    if (angle < 0) angle = -angle;

    // 计算上下端点
    cv::Point2f vertices[4];
    r.points(vertices);

    // 找到y值最小和最大的点
    int min_idx = 0, max_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (vertices[i].y < vertices[min_idx].y) min_idx = i;
        if (vertices[i].y > vertices[max_idx].y) max_idx = i;
    }

    top = vertices[min_idx];
    bottom = vertices[max_idx];

    // 初始颜色设为未知
    color = -1;
}

cv::Rect SimpleLightDetector::Light::boundingRect() const {
    return rect.boundingRect();
}

// ==================== SimpleLightDetector 方法实现 ====================

// 构造函数
SimpleLightDetector::SimpleLightDetector() {
    // 使用默认参数
}

SimpleLightDetector::SimpleLightDetector(const Params& params)
    : params_(params) {
}

// 设置参数
void SimpleLightDetector::setParams(const Params& params) {
    params_ = params;
}

// 获取参数
SimpleLightDetector::Params SimpleLightDetector::getParams() const {
    return params_;
}

// 主检测函数
std::vector<cv::Point2f> SimpleLightDetector::detect(const cv::Mat& image) {
    // 清空上一帧的数据
    last_lights_.clear();
    last_pairs_.clear();

    // 预处理
    last_binary_ = preprocess(image);

    // 检测灯条
    last_lights_ = detectLights(image, last_binary_);

    // 灯条配对
    last_pairs_ = pairLights(last_lights_);

    // 提取中心点
    std::vector<cv::Point2f> centers;
    centers.reserve(last_pairs_.size());
    for (const auto& pair : last_pairs_) {
        centers.push_back(pair.center);
    }

    return centers;
}

// 扩展检测函数
std::vector<SimpleLightDetector::PairResult> SimpleLightDetector::detectDetailed(const cv::Mat& image) {
    // 先执行普通检测
    detect(image);
    return last_pairs_;
}

// 单次检测
bool SimpleLightDetector::detectSingle(const cv::Mat& image, cv::Point2f& center) {
    // 执行检测
    auto pairs = detectDetailed(image);

    if (pairs.empty()) {
        return false;
    }

    // 找到置信度最高的配对
    auto best_pair = std::max_element(pairs.begin(), pairs.end(),
        [](const PairResult& a, const PairResult& b) {
            return a.confidence < b.confidence;
        });

    center = best_pair->center;
    return true;
}

// 绘制结果
// 注意：移除了 const 限定符
void SimpleLightDetector::drawResults(cv::Mat& image, bool draw_lights, bool draw_centers) {
    // 创建调试图像
    createDebugImage(image, last_binary_); // 现在调用的是 const 版本

    if (params_.show_contours) {
        // 绘制二值图像轮廓
        cv::Mat contours_img = image.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(last_binary_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(contours_img, contours, -1, cv::Scalar(255, 255, 255), 1);
        cv::imshow("Contours", contours_img);
    }

    // 绘制灯条
    if (draw_lights) {
        for (const auto& light : last_lights_) {
            cv::Scalar color = (light.color == 0) ? RED_COLOR : BLUE_COLOR;

            // 绘制旋转矩形
            cv::Point2f vertices[4];
            light.rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], color, 1);
            }

            // 绘制上下端点
            cv::circle(image, light.top, 3, GREEN_COLOR, -1);
            cv::circle(image, light.bottom, 3, GREEN_COLOR, -1);

            // 绘制中心点
            cv::circle(image, light.center, 2, YELLOW_COLOR, -1);
        }
    }

    // 绘制配对结果
    if (draw_centers && params_.show_pairs) {
        for (const auto& pair : last_pairs_) {
            // 绘制配对中心点
            cv::circle(image, pair.center, 6, YELLOW_COLOR, 2);

            // 绘制连线
            cv::line(image, pair.light1_center, pair.light2_center, GREEN_COLOR, 2);

            // 绘制配对矩形 (使用 OpenCV 内置函数更简洁)
            cv::Rect bounding_box(cv::Point(pair.center.x - pair.height/2, pair.center.y - pair.width/2),
                                  cv::Size(pair.height, pair.width));
            cv::rectangle(image, bounding_box, YELLOW_COLOR, 1);

            // 显示置信度
            std::string conf_text = "C:" + std::to_string(int(pair.confidence * 100)) + "%";
            cv::putText(image, conf_text, pair.center + cv::Point2f(0,pair.width/4),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1);
        }
    }

    // 显示灯条数量
    std::string light_text = "Lights: " + std::to_string(last_lights_.size());
    cv::putText(image, light_text, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, GREEN_COLOR, 2);

    // 显示配对数量
    std::string pair_text = "Pairs: " + std::to_string(last_pairs_.size());
    cv::putText(image, pair_text, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, YELLOW_COLOR, 2);
}

// 获取最后检测到的灯条
const std::vector<SimpleLightDetector::Light>& SimpleLightDetector::getLastLights() const {
    return last_lights_;
}

// 获取最后的配对结果
const std::vector<SimpleLightDetector::PairResult>& SimpleLightDetector::getLastPairs() const {
    return last_pairs_;
}

// 获取调试图像
cv::Mat SimpleLightDetector::getBinaryImage() const {
    cv::Mat color_binary;
    cv::cvtColor(last_binary_, color_binary, cv::COLOR_GRAY2BGR);
    return color_binary;
}

cv::Mat SimpleLightDetector::getDebugImage() const {
    return last_debug_.clone();
}

// ==================== 私有方法实现 ====================

// 预处理
cv::Mat SimpleLightDetector::preprocess(const cv::Mat& image) const {
    cv::Mat gray, binary;

    // 转为灰度图
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::threshold(gray, binary, params_.binary_thresh, 255, cv::THRESH_BINARY);

    // 可选：形态学操作去除噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    return binary;
}

// 检测灯条
std::vector<SimpleLightDetector::Light> SimpleLightDetector::detectLights(
    const cv::Mat& color_img, const cv::Mat& binary_img) {

    std::vector<Light> lights;
    std::vector<std::vector<cv::Point>> contours;

    // 查找轮廓
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;

        // 计算轮廓面积
        double area = cv::contourArea(contour);
        if (area < params_.min_area || area > params_.max_area) continue;

        // 最小外接矩形
        cv::RotatedRect rect = cv::minAreaRect(contour);
        Light light(rect);

        // 检查是否为有效灯条
        if (isValidLight(light)) {
            // 检测颜色
            light.color = detectColor(light, color_img);

            // 只保留目标颜色的灯条
            if (light.color == params_.target_color) {
                lights.push_back(light);
            }
        }
    }

    return lights;
}

// 判断是否为有效灯条
bool SimpleLightDetector::isValidLight(const Light& light) const {
    // 长宽比过滤
    float ratio = light.width / light.length;
    if (ratio < params_.min_ratio || ratio > params_.max_ratio) {
        return false;
    }

    // 面积过滤 (虽然在detectLights里也做了，这里再确认一遍)
    if (light.area < params_.min_area || light.area > params_.max_area) {
        return false;
    }

    // 角度过滤
    if (light.angle > params_.max_angle) {
        return false;
    }

    return true;
}

// 检测灯条颜色
int SimpleLightDetector::detectColor(const Light& light, const cv::Mat& color_img) const {
    cv::Rect rect = light.boundingRect();

    // 确保ROI在图像范围内
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, color_img.cols - rect.x);
    rect.height = std::min(rect.height, color_img.rows - rect.y);

    if (rect.width <= 0 || rect.height <= 0) {
        return -1; // ROI无效
    }

    // 采样检查轮廓内的像素颜色
    cv::Mat roi = color_img(rect);

    // 修复：删除了错误的 std::vector<cv::Point> contour 初始化行
    // std::vector<cv::Point> contour = std::vector<cv::Point>(light.rect.boundingRect());

    int red_sum = 0, blue_sum = 0;
    int pixel_count = 0;

    // 获取旋转矩形的四个顶点用于精确判断点是否在内部
    std::vector<cv::Point2f> rect_points(4);
    light.rect.points(rect_points.data());
    // Convert Point2f to Point for pointPolygonTest
    std::vector<cv::Point> polygon_points(rect_points.begin(), rect_points.end());

    // 在ROI内采样检查
    for (int i = 0; i < roi.rows; i += 2) {
        const cv::Vec3b* row = roi.ptr<cv::Vec3b>(i);
        for (int j = 0; j < roi.cols; j += 2) {
            // 计算点在原图中的绝对坐标
            cv::Point abs_pt(j + rect.x, i + rect.y);

            // 检查点是否在旋转矩形内 (更精确)
            double test_result = cv::pointPolygonTest(polygon_points, abs_pt, false);
            if (test_result >= 0) { // 点在多边形内部或边上
                red_sum += row[j][2];  // BGR格式，索引2是红色通道
                blue_sum += row[j][0]; // 索引0是蓝色通道
                pixel_count++;
            }
            // 原始近似方法（如果 pointPolygonTest 性能不好或有问题可考虑，但 boundingRect 太粗糙）
            /*
            else if (rect.contains(abs_pt)) { // rect 是 axis-aligned bounding box
                red_sum += row[j][2];
                blue_sum += row[j][0];
                pixel_count++;
            }
            */
        }
    }

    if (pixel_count == 0) {
        return -1; // 没有采样到有效像素
    }

    // 判断颜色
    if (red_sum > blue_sum + params_.color_threshold) {
        return 0;  // 红色
    } else if (blue_sum > red_sum + params_.color_threshold) {
        return 1;  // 蓝色
    } else {
        return -1; // 不确定
    }
}

// 灯条配对
std::vector<SimpleLightDetector::PairResult> SimpleLightDetector::pairLights(
    const std::vector<Light>& lights) {

    std::vector<PairResult> pairs;

    if (lights.size() < 2) {
        return pairs;
    }

    // 遍历所有灯条对
    for (size_t i = 0; i < lights.size(); i++) {
        for (size_t j = i + 1; j < lights.size(); j++) {
            const Light& l1 = lights[i];
            const Light& l2 = lights[j];

            float center_dist, angle_diff;
            if (canPair(l1, l2, center_dist, angle_diff)) {
                PairResult pair_result;

                // 计算配对中心点
                pair_result.center = (l1.center + l2.center) * 0.5f;
                pair_result.light1_center = l1.center;
                pair_result.light2_center = l2.center;

                // 计算配对尺寸
                pair_result.width = distance(l1.center, l2.center) + (l1.width + l2.width) * 0.5f;
                pair_result.height = std::max(l1.length, l2.length) * 1.2f;  // 留出余量

                // 计算配对角度
                pair_result.angle = std::atan2(l2.center.y - l1.center.y,
                                              l2.center.x - l1.center.x) * 180.0f / CV_PI;

                // 计算置信度
                pair_result.confidence = calculatePairConfidence(l1, l2, center_dist, angle_diff);

                pairs.push_back(pair_result);
            }
        }
    }

    return pairs;
}

// 判断两个灯条是否可以配对
bool SimpleLightDetector::canPair(const Light& l1, const Light& l2,
                                 float& center_dist, float& angle_diff) const {

    // 1. 长度比例检查
    float length_ratio = std::min(l1.length, l2.length) / std::max(l1.length, l2.length);
    if (length_ratio < params_.min_light_ratio) {
        return false;
    }

    // 2. 中心距离检查
    float avg_length = (l1.length + l2.length) * 0.5f;
    center_dist = distance(l1.center, l2.center) / avg_length;

    if (center_dist < params_.min_center_dist || center_dist > params_.max_center_dist) {
        return false;
    }

    // 3. 角度差检查
    angle_diff = angleDifference(l1.angle, l2.angle);
    if (angle_diff > params_.max_pair_angle) {
        return false;
    }

    // 4. 方向一致性检查 (可选，但原逻辑较复杂且可能不必要，简化判断)
    // 这里简化为只要角度差和中心距满足条件即可配对。
    // 如果需要更严格的几何约束，可以在此处添加。

    return true;
}

// 计算配对置信度
float SimpleLightDetector::calculatePairConfidence(const Light& l1, const Light& l2,
                                                   float center_dist, float angle_diff) const {
    float confidence = 1.0f;

    // 1. 长度比例相似度
    float length_ratio = std::min(l1.length, l2.length) / std::max(l1.length, l2.length);
    confidence *= length_ratio;  // 比例越接近1，置信度越高

    // 2. 角度差相似度
    float angle_similarity = 1.0f - (angle_diff / params_.max_pair_angle);
    // --- 修改开始 ---
    // 使用 std::max 确保非负，替代 C++17 的 std::clamp
    confidence *= std::max(0.0f, angle_similarity);
    // --- 修改结束 ---

    // 3. 中心距适中度（在中间位置最好）
    float ideal_dist = (params_.min_center_dist + params_.max_center_dist) * 0.5f;
    float dist_deviation = std::abs(center_dist - ideal_dist);
    float max_deviation = (params_.max_center_dist - params_.min_center_dist) * 0.5f;
    float dist_ratio = 1.0f - (dist_deviation / max_deviation);
    // --- 修改开始 ---
    // 使用 std::max 确保非负
    confidence *= std::max(0.0f, dist_ratio);
    // --- 修改结束 ---

    // 4. 面积相似度
    float area_ratio = std::min(l1.area, l2.area) / std::max(l1.area, l2.area);
    confidence *= area_ratio;

    // --- 修改开始 ---
    // 手动实现 clamp 效果，确保最终置信度在 [0, 1] 范围内
    // 先确保不超过 1.0
    confidence = std::min(confidence, 1.0f);
    // 再确保不低于 0.0
    confidence = std::max(confidence, 0.0f);
    // --- 修改结束 ---

    return confidence; // 返回最终置信度
}


// 创建调试图像
// 注意：添加了 const 限定符
void SimpleLightDetector::createDebugImage(const cv::Mat& color_img, const cv::Mat& binary_img) const {
    // 创建三通道的调试图像
    cv::Mat binary_color;
    cv::cvtColor(binary_img, binary_color, cv::COLOR_GRAY2BGR);

    // 水平拼接原图和二值图
    // 由于 last_debug_ 是 mutable，const 函数可以修改它
    cv::hconcat(color_img, binary_color, last_debug_);
    // 如果不想用 mutable，可以用 const_cast，但这不推荐：
    // cv::hconcat(color_img, binary_color, const_cast<cv::Mat&>(last_debug_));
}

// 辅助函数：计算两点距离
float SimpleLightDetector::distance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 辅助函数：计算角度差
float SimpleLightDetector::angleDifference(float a1, float a2) const {
    float diff = std::abs(a1 - a2);
    if (diff > 90.0f) {
        diff = 180.0f - diff;
    }
    return diff;
}