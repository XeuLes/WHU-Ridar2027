#include "uav_light_detector.h"

namespace tdt_radar {

LightBarDetector::LightBarDetector() {
    // 单灯条筛选参数（与SimpleLightDetector一致）
    min_area_ = 10.0f;
    max_area_ = 10000.0f;
    min_aspect_ratio_ = 0.1f;    ///< 宽/长下限
    max_aspect_ratio_ = 0.4f;    ///< 宽/长上限
    max_angle_ = 90.0f;
    
    // 上下配对参数（pair_mode=1模式）
    max_light_angle_diff_ = 10.0f;   ///< 两灯条角度差阈值
    max_light_length_ratio_ = 0.6f;  ///< 长度比例min_ratio（短/长≥0.6）
    min_vertical_ratio_ = 0.8f;      ///< 垂直间距/宽度下限
    max_vertical_ratio_ = 4.0f;      ///< 垂直间距/宽度上限
    max_horizontal_offset_ = 0.5f;   ///< 水平偏移/宽度上限
    min_connect_angle_ = 60.0f;      ///< 中心连线角度下限（接近垂直）
    max_connect_angle_ = 120.0f;     ///< 中心连线角度上限
}

std::vector<LightBarDetector::LightBar> LightBarDetector::detectLights(
    const cv::Mat& roi, int color_filter) {
    std::vector<LightBar> lights;
    
    if (roi.empty() || roi.cols < 20 || roi.rows < 20) {
        return lights;
    }

    // 步骤1：颜色分离
    cv::Mat binary;
    if (color_filter == 0) {
        // 蓝色：提取B通道
        std::vector<cv::Mat> channels;
        cv::split(roi, channels);
        cv::Mat blue_minus_red;
        cv::subtract(channels[0], channels[2], blue_minus_red);  // B - R
        cv::threshold(blue_minus_red, binary, 80, 255, cv::THRESH_BINARY);
    } else if (color_filter == 2) {
        // 红色：提取R通道
        std::vector<cv::Mat> channels;
        cv::split(roi, channels);
        cv::Mat red_minus_blue;
        cv::subtract(channels[2], channels[0], red_minus_blue);  // R - B
        cv::threshold(red_minus_blue, binary, 80, 255, cv::THRESH_BINARY);
    } else {
        // 无颜色过滤：灰度二值化
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY);
    }

    // 步骤2：形态学闭运算（去除小孔洞）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // 步骤3：轮廓提取
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 步骤4：轮廓筛选为灯条
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        
        cv::RotatedRect rrect = cv::minAreaRect(contour);
        float area = rrect.size.area();
        
        // 面积阈值
        if (area < min_area_ || area > max_area_) continue;
        
        // 长宽比计算（width/length）
        float length = std::max(rrect.size.width, rrect.size.height);
        float width = std::min(rrect.size.width, rrect.size.height);
        float aspect_ratio = width / (length + 1e-6f);
        
        if (aspect_ratio < min_aspect_ratio_ || aspect_ratio > max_aspect_ratio_) continue;
        
        // 角度归一化到[0, 180]
        float angle = rrect.angle;
        if (rrect.size.width < rrect.size.height) {
            angle += 90.0f;
        }
        if (std::abs(angle) > max_angle_ && std::abs(angle - 180.0f) > max_angle_) continue;
        
        LightBar light;
        light.rect = rrect;
        light.center = rrect.center;
        light.angle = angle;
        light.length = length;
        light.width = width;
        light.aspect_ratio = aspect_ratio;
        
        lights.push_back(light);
    }
    
    return lights;
}

std::vector<LightBarDetector::UAVTarget> LightBarDetector::matchUAVTargets(
    const std::vector<LightBar>& lights) {
    std::vector<UAVTarget> targets;
    
    if (lights.size() < 2) return targets;
    
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            const auto& light1 = lights[i];
            const auto& light2 = lights[j];
            
            // 角度差过滤
            float angle_diff = std::abs(light1.angle - light2.angle);
            if (angle_diff > max_light_angle_diff_) continue;
            
            // 长度比过滤 (与auto_aim_fast一致)
            float length_ratio = std::min(light1.length, light2.length) / 
                                std::max(light1.length, light2.length);
            if (length_ratio < max_light_length_ratio_) continue;
            
            // 计算垂直和水平距离
            float dx = std::abs(light1.center.x - light2.center.x);
            float dy = std::abs(light1.center.y - light2.center.y);
            
            // 计算目标参数
            cv::Point2f center = (light1.center + light2.center) * 0.5f;
            float avg_length = (light1.length + light2.length) * 0.5f;
            
            // X坐标差异检查
            float x_diff_ratio = dx / avg_length;
            if (x_diff_ratio > max_horizontal_offset_) continue;
            
            // 垂直距离检查
            float vertical_dist = dy / avg_length;
            if (vertical_dist < min_vertical_ratio_ || 
                vertical_dist > max_vertical_ratio_) continue;
            
            // 中心连线垂直度检查
            float connect_angle = std::atan2(dy, dx) * 180.0f / CV_PI;
            if (connect_angle < min_connect_angle_ || 
                connect_angle > max_connect_angle_) continue;
            
            UAVTarget target;
            // 确定上下关系
            target.top = (light1.center.y < light2.center.y) ? light1 : light2;
            target.bottom = (light1.center.y < light2.center.y) ? light2 : light1;
            target.center = center;
            target.width = avg_length;
            target.height = dy;
            target.type = 0;
            
            targets.push_back(target);
        }
    }
    
    return targets;
}

cv::Rect LightBarDetector::calculateFineROI(const cv::Rect& uav_rect) {
    int roi_width = uav_rect.width;
    int roi_height = uav_rect.height;
    
    // 确定长边和短边
    bool is_vertical = (roi_height > roi_width);
    
    cv::Rect fine_roi_local;
    if (is_vertical) {
        // 竖直方向：长边是height，短边是width
        int fine_height = static_cast<int>(roi_height * 0.4f);
        int fine_width = static_cast<int>(roi_width * 0.5f);
        int fine_y = static_cast<int>(roi_height * 0.3f);
        int fine_x = (roi_width - fine_width) / 2;
        
        fine_roi_local = cv::Rect(fine_x, fine_y, fine_width, fine_height);
    } else {
        // 水平方向：长边是width，短边是height
        int fine_width = static_cast<int>(roi_width * 0.4f);
        int fine_height = static_cast<int>(roi_height * 0.4f);
        int fine_x = static_cast<int>(roi_width * 0.3f);
        int fine_y = 0;
        
        fine_roi_local = cv::Rect(fine_x, fine_y, fine_width, fine_height);
    }
    
    // 确保精细ROI在有效范围内
    fine_roi_local.x = std::max(0, fine_roi_local.x);
    fine_roi_local.y = std::max(0, fine_roi_local.y);
    fine_roi_local.width = std::min(fine_roi_local.width, roi_width - fine_roi_local.x);
    fine_roi_local.height = std::min(fine_roi_local.height, roi_height - fine_roi_local.y);
    
    return fine_roi_local;
}

cv::Point2f LightBarDetector::performFineDetection(
    const cv::Mat& uav_roi,
    const cv::Rect& uav_rect,
    int uav_color,
    cv::Rect& fine_roi_local,
    cv::Point2f& fine_roi_center_global,
    std::vector<LightBar>& detected_lights,
    std::vector<UAVTarget>& detected_targets) {
    
    // 1. 计算精细ROI
    fine_roi_local = calculateFineROI(uav_rect);
    
    // 确保 fine_roi_local 在 uav_roi 的有效范围内
    fine_roi_local.x = std::max(0, std::min(fine_roi_local.x, uav_roi.cols - 1));
    fine_roi_local.y = std::max(0, std::min(fine_roi_local.y, uav_roi.rows - 1));
    fine_roi_local.width = std::min(fine_roi_local.width, uav_roi.cols - fine_roi_local.x);
    fine_roi_local.height = std::min(fine_roi_local.height, uav_roi.rows - fine_roi_local.y);
    
    // 检查ROI是否有效
    if (fine_roi_local.width <= 0 || fine_roi_local.height <= 0) {
        // ROI无效，返回UAV矩形中心
        return cv::Point2f(
            uav_rect.x + uav_rect.width / 2.0f,
            uav_rect.y + uav_rect.height / 2.0f
        );
    }
    
    // 提取精细ROI图像
    cv::Mat fine_roi = uav_roi(fine_roi_local);
    
    // 计算精细ROI的中心点（全局坐标）
    fine_roi_center_global = cv::Point2f(
        uav_rect.x + fine_roi_local.x + fine_roi_local.width / 2.0f,
        uav_rect.y + fine_roi_local.y + fine_roi_local.height / 2.0f
    );
    
    // 默认使用精细ROI中心
    cv::Point2f refined_center = fine_roi_center_global;
    
    // 2. 在精细ROI中检测灯条
    if (!fine_roi.empty()) {
        detected_lights = detectLights(fine_roi, uav_color);
        
        // 3. 匹配UAV目标（上下配对）
        if (detected_lights.size() >= 2) {
            detected_targets = matchUAVTargets(detected_lights);
            
            // 4. 如果检测到灯条对，使用最佳目标的中心
            if (!detected_targets.empty()) {
                // 选择最靠近精细ROI中心的目标
                cv::Point2f fine_roi_center_local(fine_roi.cols / 2.0f, fine_roi.rows / 2.0f);
                float min_dist = std::numeric_limits<float>::max();
                int best_target_idx = 0;
                
                for (size_t k = 0; k < detected_targets.size(); ++k) {
                    float dx = detected_targets[k].center.x - fine_roi_center_local.x;
                    float dy = detected_targets[k].center.y - fine_roi_center_local.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_target_idx = k;
                    }
                }
                
                // 更新精细定位中心点（fine_roi局部坐标 → 全局坐标）
                refined_center.x = uav_rect.x + fine_roi_local.x + detected_targets[best_target_idx].center.x;
                refined_center.y = uav_rect.y + fine_roi_local.y + detected_targets[best_target_idx].center.y;
            }
        }
    }
    
    return refined_center;
}

void LightBarDetector::drawLightDetection(
    cv::Mat& img,
    const cv::Rect& uav_rect,
    const cv::Rect& fine_roi_local,
    const std::vector<LightBar>& detected_lights,
    const std::vector<UAVTarget>& detected_targets,
    const cv::Point2f& refined_center,
    const cv::Point2f& fine_roi_center_global) {
    
    // 绘制精细ROI框
    cv::Rect fine_roi_global(
        uav_rect.x + fine_roi_local.x,
        uav_rect.y + fine_roi_local.y,
        fine_roi_local.width,
        fine_roi_local.height
    );
    cv::rectangle(img, fine_roi_global, cv::Scalar(255, 255, 0), 2);
    
    // 绘制精细ROI中心点
    cv::circle(img, fine_roi_center_global, 3, cv::Scalar(255, 255, 0), -1);
    cv::line(img, fine_roi_center_global - cv::Point2f(8, 0), 
            fine_roi_center_global + cv::Point2f(8, 0), cv::Scalar(255, 255, 0), 1);
    cv::line(img, fine_roi_center_global - cv::Point2f(0, 8), 
            fine_roi_center_global + cv::Point2f(0, 8), cv::Scalar(255, 255, 0), 1);
    
    // 绘制检测到的灯条
    for (const auto& light : detected_lights) {
        cv::Point2f pts[4];
        light.rect.points(pts);
        for (int k = 0; k < 4; k++) {
            pts[k].x += uav_rect.x + fine_roi_local.x;
            pts[k].y += uav_rect.y + fine_roi_local.y;
        }
        for (int k = 0; k < 4; k++) {
            cv::line(img, pts[k], pts[(k+1)%4], cv::Scalar(0, 255, 255), 2);
        }
    }
    
    // 绘制UAV目标（上下灯条对）
    for (const auto& target : detected_targets) {
        cv::Point2f target_center_global(
            uav_rect.x + fine_roi_local.x + target.center.x,
            uav_rect.y + fine_roi_local.y + target.center.y
        );
        cv::circle(img, target_center_global, 5, cv::Scalar(0, 255, 0), -1);
        
        cv::Point2f top_center_global(
            uav_rect.x + fine_roi_local.x + target.top.center.x,
            uav_rect.y + fine_roi_local.y + target.top.center.y
        );
        cv::Point2f bottom_center_global(
            uav_rect.x + fine_roi_local.x + target.bottom.center.x,
            uav_rect.y + fine_roi_local.y + target.bottom.center.y
        );
        cv::line(img, top_center_global, bottom_center_global, cv::Scalar(0, 255, 0), 2);
        
        cv::Point2f top_left(target_center_global.x - target.width/2, top_center_global.y);
        cv::Point2f bottom_right(target_center_global.x + target.width/2, bottom_center_global.y);
        cv::rectangle(img, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
    }
    
    // 绘制精细定位中心点
    if (!detected_targets.empty()) {
        cv::circle(img, refined_center, 8, cv::Scalar(0, 255, 0), 2);
        cv::line(img, refined_center - cv::Point2f(10, 0), 
                refined_center + cv::Point2f(10, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(img, refined_center - cv::Point2f(0, 10), 
                refined_center + cv::Point2f(0, 10), cv::Scalar(0, 255, 0), 2);
    }
}

}  // namespace tdt_radar
