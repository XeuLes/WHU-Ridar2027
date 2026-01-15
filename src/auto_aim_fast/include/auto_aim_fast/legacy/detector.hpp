#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class SimpleLightDetector {
public:
    struct Params {
        // 预处理参数
        int binary_thresh = 150;
        
        // 灯条筛选参数
        double min_area = 10;
        double max_area = 10000;
        float min_ratio = 0.1f;  // 宽高比下限 (width/length)
        float max_ratio = 0.4f;  // 宽高比上限
        float max_angle = 90.0f; // 灯条最大倾斜角度
        
        // 颜色检测参数
        int target_color = 0; // 0: Red, 1: Blue
        int color_threshold = 30; // 颜色通道差异阈值
        
        // 灯条配对参数
        float min_light_ratio = 0.6f;    // 配对灯条长度比例下限
        float min_center_dist = 1.0f;    // 归一化中心距离下限
        float max_center_dist = 5.0f;    // 归一化中心距离上限
        float max_pair_angle = 10.0f;    // 配对灯条角度差上限
        
        // 显示参数
        bool show_contours = false;
        bool show_pairs = true;
    };

    struct Light {
        cv::RotatedRect rect; // 最小外接矩形
        cv::Point2f center;   // 中心点
        float width;          // 短边 (灯条宽度)
        float length;         // 长边 (灯条长度)
        float angle;          // 角度 (相对于水平线，[0, 90))
        float area;           // 面积
        cv::Point2f top;      // 上端点
        cv::Point2f bottom;   // 下端点
        int color;            // 颜色 (0: Red, 1: Blue, -1: Unknown)

        Light(const cv::RotatedRect& r);
        cv::Rect boundingRect() const; // 获取轴对齐包围盒
    };

    struct PairResult {
        cv::Point2f center;          // 配对中心点
        cv::Point2f light1_center;   // 左侧灯条中心
        cv::Point2f light2_center;   // 右侧灯条中心
        float width;                 // 配对宽度
        float height;                // 配对高度
        float angle;                 // 配对角度
        float confidence;            // 配对置信度
    };

    // 静态颜色常量
    static const cv::Scalar RED_COLOR;
    static const cv::Scalar BLUE_COLOR;
    static const cv::Scalar GREEN_COLOR;
    static const cv::Scalar YELLOW_COLOR;


    // 构造函数
    SimpleLightDetector();
    explicit SimpleLightDetector(const Params& params);

    // 参数设置与获取
    void setParams(const Params& params);
    Params getParams() const;

    // 主检测接口
    std::vector<cv::Point2f> detect(const cv::Mat& image);
    std::vector<PairResult> detectDetailed(const cv::Mat& image);
    bool detectSingle(const cv::Mat& image, cv::Point2f& center);

    // 结果可视化
    // 注意：此函数不再是 const，因为它会修改 last_debug_
    void drawResults(cv::Mat& image, bool draw_lights = true, bool draw_centers = true);

    // 获取历史数据
    const std::vector<Light>& getLastLights() const;
    const std::vector<PairResult>& getLastPairs() const;

    // 获取调试图像
    cv::Mat getBinaryImage() const;
    cv::Mat getDebugImage() const;

private:
    Params params_;

    // 存储上一次检测的结果
    std::vector<Light> last_lights_;
    std::vector<PairResult> last_pairs_;
    cv::Mat last_binary_;
    mutable cv::Mat last_debug_; // mutable 允许 const 函数修改它

    // 核心处理步骤
    cv::Mat preprocess(const cv::Mat& image) const;
    std::vector<Light> detectLights(const cv::Mat& color_img, const cv::Mat& binary_img);
    bool isValidLight(const Light& light) const;
    int detectColor(const Light& light, const cv::Mat& color_img) const;
    std::vector<PairResult> pairLights(const std::vector<Light>& lights);
    bool canPair(const Light& l1, const Light& l2, float& center_dist, float& angle_diff) const;
    float calculatePairConfidence(const Light& l1, const Light& l2, float center_dist, float angle_diff) const;

    // 辅助函数
    // 注意：此函数现在是 const
    void createDebugImage(const cv::Mat& color_img, const cv::Mat& binary_img) const;
    float distance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    float angleDifference(float a1, float a2) const;
};
