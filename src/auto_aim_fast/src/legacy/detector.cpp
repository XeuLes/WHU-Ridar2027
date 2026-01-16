#include "auto_aim_fast/legacy/detector.hpp" // 确保包含正确的头文件
#include <algorithm>
#include <cmath>
#include <numeric> // For std::accumulate if needed
#include <iostream> // For std::cout debug logging


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

            // 绘制配对矩形
            // 注意：cv::Rect(左上角, Size(宽度, 高度))
            // pair.width 是配对的实际宽度，pair.height 是配对的实际高度
            cv::Rect bounding_box(cv::Point(pair.center.x - pair.width/2, pair.center.y - pair.height/2),
                                  cv::Size(pair.width, pair.height));
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

    // ✅ 输出检测结果统计
    printf("[Detection] Total contours: %zu, Valid lights: %zu, Target color: %d\n", 
           contours.size(), lights.size(), params_.target_color);
    fflush(stdout);

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

    // ✅ 输出灯条数量
    printf("[Pairing] Total lights detected: %zu\n", lights.size());
    fflush(stdout);

    if (lights.size() < 2) {
        printf("[Pairing] ❌ Not enough lights for pairing (need >= 2)\n");
        fflush(stdout);
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
                
                // 根据配对模式设置灯条顺序
                if (params_.pair_mode == 1) {
                    // 上下配对：按y坐标排序，y小的在上
                    if (l1.center.y < l2.center.y) {
                        pair_result.light1_center = l1.center;  // 上侧
                        pair_result.light2_center = l2.center;  // 下侧
                    } else {
                        pair_result.light1_center = l2.center;  // 上侧
                        pair_result.light2_center = l1.center;  // 下侧
                    }
                } else {
                    // 左右配对：按x坐标排序，x小的在左
                    if (l1.center.x < l2.center.x) {
                        pair_result.light1_center = l1.center;  // 左侧
                        pair_result.light2_center = l2.center;  // 右侧
                    } else {
                        pair_result.light1_center = l2.center;  // 左侧
                        pair_result.light2_center = l1.center;  // 右侧
                    }
                }

                // 计算配对尺寸
                if (params_.pair_mode == 1) {
                    // 上下配对：宽度=灯条长度，高度=垂直距离+灯条宽度
                    pair_result.width = std::max(l1.length, l2.length);
                    pair_result.height = std::abs(l2.center.y - l1.center.y) + (l1.width + l2.width) * 0.5f;
                } else {
                    // 左右配对：宽度=水平距离+灯条宽度，高度=灯条长度
                    pair_result.width = std::abs(l2.center.x - l1.center.x) + (l1.width + l2.width) * 0.5f;
                    pair_result.height = std::max(l1.length, l2.length) * 1.2f;
                }

                // 计算配对角度
                pair_result.angle = std::atan2(l2.center.y - l1.center.y,
                                              l2.center.x - l1.center.x) * 180.0f / CV_PI;

                // 设置配对类型
                pair_result.pair_type = params_.pair_mode;

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

    printf("[canPair] Testing L1(%.1f,%.1f) vs L2(%.1f,%.1f)\n", 
           l1.center.x, l1.center.y, l2.center.x, l2.center.y);

    // 1. 长度比例检查 - 两个灯条长度应该相似
    float length_ratio = std::min(l1.length, l2.length) / std::max(l1.length, l2.length);
    printf("  [Step1] length_ratio=%.3f (min=%.3f)\n", length_ratio, params_.min_light_ratio);
    if (length_ratio < params_.min_light_ratio) {
        printf("  ❌ Step1 FAILED: length_ratio too small\n");
        return false;
    }

    // 2. 角度差检查 - 两个灯条应该平行
    angle_diff = angleDifference(l1.angle, l2.angle);
    printf("  [Step2] angle_diff=%.2f° (max=%.2f°)\n", angle_diff, params_.max_pair_angle);
    if (angle_diff > params_.max_pair_angle) {
        printf("  ❌ Step2 FAILED: angle_diff too large\n");
        return false;
    }

    float avg_length = (l1.length + l2.length) * 0.5f;
    float dx = std::abs(l2.center.x - l1.center.x);
    float dy = std::abs(l2.center.y - l1.center.y);
    
    // ========== 上下配对模式(立方体上边/下边在图像中上下分布) ==========
    
    // 3. X坐标差异检查 - 两个灯条应该在同一垂直线上（图像中）
    float x_diff_ratio = dx / avg_length;
    printf("  [Step3] x_diff_ratio=%.3f (max=%.3f), dx=%.2f, avg_length=%.2f\n", 
           x_diff_ratio, params_.max_x_diff_ratio, dx, avg_length);
    if (x_diff_ratio > params_.max_x_diff_ratio) {
        printf("  ❌ Step3 FAILED: x_diff_ratio too large\n");
        return false;
    }
    
    // 4. 垂直距离检查 - 上下距离应该在合理范围内
    float vertical_dist = dy / avg_length;
    printf("  [Step4] vertical_dist=%.3f (range=[%.3f, %.3f]), dy=%.2f\n", 
           vertical_dist, params_.min_vertical_dist, params_.max_vertical_dist, dy);
    if (vertical_dist < params_.min_vertical_dist || vertical_dist > params_.max_vertical_dist) {
        printf("  ❌ Step4 FAILED: vertical_dist out of range\n");
        return false;
    }
    
    // 5. 中心连线垂直度检查 - 两灯条中心连线应该接近垂直
    // Step 5: 连线角度范围检查 - 过滤水平配对，保留接近垂直的配对
    float connect_angle = std::atan2(dy, dx) * 180.0f / CV_PI;
    
    printf("  Step5: connect_angle=%.2f° (range: [%.2f°, %.2f°]), dx=%.2f, dy=%.2f\n",
           connect_angle, params_.min_connect_angle, params_.max_connect_angle, dx, dy);
    fflush(stdout);
    
    if (connect_angle < params_.min_connect_angle || connect_angle > params_.max_connect_angle) {
        printf("  ❌ Step5 FAILED: angle %.2f° not in valid range [%.2f°, %.2f°] (L1=(%.1f,%.1f), L2=(%.1f,%.1f))\n",
               connect_angle, params_.min_connect_angle, params_.max_connect_angle,
               l1.center.x, l1.center.y, l2.center.x, l2.center.y);
        fflush(stdout);
        return false;
    }
    
    printf("  ✅ Step5 PASSED: angle %.2f° in valid range\n", connect_angle);
    fflush(stdout);
    
    center_dist = vertical_dist;  // 返回垂直距离作为中心距

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