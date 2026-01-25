#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <rclcpp/rclcpp.hpp>

namespace tdt_radar {

/**
 * @brief UAV灯条检测器类
 * @details 检测正方体无人机的上下配对灯条（垂直排列）
 *          参数与auto_aim_fast的SimpleLightDetector保持一致
 *          检测流程：颜色分离 → 二值化 → 轮廓筛选 → 上下配对
 */
class LightBarDetector {
public:
    struct LightBar {
        cv::RotatedRect rect;      ///< 旋转矩形包围盒
        cv::Point2f center;        ///< 中心点坐标
        float angle;               ///< 旋转角度 [0, 90]°
        float length;              ///< 长边长度
        float width;               ///< 短边宽度
        float aspect_ratio;        ///< 宽长比 width/length
    };

    struct UAVTarget {
        LightBar top;              ///< 上灯条
        LightBar bottom;           ///< 下灯条
        cv::Point2f center;        ///< 配对目标中心点
        float width;               ///< 目标宽度（两灯条平均宽度）
        float height;              ///< 目标高度（上下灯条间距）
        int type;                  ///< 目标类型（0-正方体UAV）
    };

    /**
     * @brief 构造函数，初始化检测参数（与auto_aim_fast统一）
     */
    LightBarDetector();

    /**
     * @brief 在ROI中检测单个灯条
     * @param roi 输入图像ROI
     * @param color_filter 颜色过滤：0-蓝色，2-红色，-1-无过滤
     * @return 检测到的所有灯条列表
     */
    std::vector<LightBar> detectLights(const cv::Mat& roi, int color_filter = -1);

    /**
     * @brief 匹配UAV目标
     * @param lights 单灯条列表
     * @return 匹配成功的UAV目标列表
     * @note 遍历所有灯条对，验证：角度差、长度比、垂直/水平间距、连线角度
     */
    std::vector<UAVTarget> matchUAVTargets(const std::vector<LightBar>& lights);

    /**
     * @brief 计算精细ROI区域
     * @param uav_rect UAV检测框
     * @return 精细ROI的局部坐标（相对于uav_rect）
     */
    cv::Rect calculateFineROI(const cv::Rect& uav_rect);

    /**
     * @brief 执行完整的UAV精细检测流程（高层封装）
     * @param uav_roi UAV区域图像
     * @param uav_rect UAV全局坐标框
     * @param uav_color UAV颜色（0-蓝，2-红）
     * @param[out] fine_roi_local 输出：精细ROI局部坐标
     * @param[out] fine_roi_center_global 输出：精细ROI中心（全局坐标）
     * @param[out] detected_lights 输出：检测到的灯条
     * @param[out] detected_targets 输出：检测到的目标对
     * @return 精细定位后的中心点（全局坐标）
     */
    cv::Point2f performFineDetection(
        const cv::Mat& uav_roi,
        const cv::Rect& uav_rect,
        int uav_color,
        cv::Rect& fine_roi_local,
        cv::Point2f& fine_roi_center_global,
        std::vector<LightBar>& detected_lights,
        std::vector<UAVTarget>& detected_targets
    );

    /**
     * @brief 绘制灯条检测结果
     * @param img 输出图像
     * @param uav_rect UAV全局坐标框
     * @param fine_roi_local 精细ROI局部坐标
     * @param detected_lights 检测到的灯条列表
     * @param detected_targets 检测到的UAV目标列表
     * @param refined_center 精细定位中心点
     * @param fine_roi_center_global 精细ROI中心点（全局坐标）
     */
    void drawLightDetection(
        cv::Mat& img,
        const cv::Rect& uav_rect,
        const cv::Rect& fine_roi_local,
        const std::vector<LightBar>& detected_lights,
        const std::vector<UAVTarget>& detected_targets,
        const cv::Point2f& refined_center,
        const cv::Point2f& fine_roi_center_global
    );

private:
    // 灯条检测参数 (与auto_aim_fast一致)
    float min_area_;
    float max_area_;
    float min_aspect_ratio_;        // 宽高比下限 (width/length)
    float max_aspect_ratio_;        // 宽高比上限 (width/length)
    float max_angle_;
    
    // 上下匹配参数 (与auto_aim_fast的pair_mode=1一致)
    float max_light_angle_diff_;    // 配对灯条角度差上限
    float max_light_length_ratio_;  // 配对灯条长度比例下限
    float min_vertical_ratio_;      // 最小垂直距离(归一化)
    float max_vertical_ratio_;      // 最大垂直距离(归一化)
    float max_horizontal_offset_;   // 最大水平偏移(归一化)
    float min_connect_angle_;       // 中心连线角度下限
    float max_connect_angle_;       // 中心连线角度上限
};

}  // namespace tdt_radar
