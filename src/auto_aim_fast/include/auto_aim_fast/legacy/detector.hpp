#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class SimpleLightDetector {
public:
    struct Params {
        // 预处理参数
        int binary_thresh = 150;
        bool use_sigma_thresh = true;
        float sigma_k = 2.5f;
        int min_dynamic_thresh = 80;
        int max_dynamic_thresh = 220;
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
        
        // 配对模式参数 (0: 左右配对装甲板, 1: 上下配对立方体)
        int pair_mode = 1;               // 配对模式：0=左右，1=上下
        float max_x_diff_ratio = 0.5f;   // x坐标差与平均长度比值上限（上下配对用）
        float max_y_diff_ratio = 0.5f;   // y坐标差与平均长度比值上限（左右配对用）
        float min_vertical_dist = 0.8f;  // 上下最小垂直距离（归一化）
        float max_vertical_dist = 4.0f;  // 上下最大垂直距离（归一化）
        float min_connect_angle = 60.0f; // 中心连线角度下限（避免接近水平的配对）
        float max_connect_angle = 120.0f; // 中心连线角度上限（允许一定倾斜）
        
        // 显示参数
        bool show_contours = false;
        bool show_pairs = true;

        bool use_spot_pattern = true;
        
        //点阵参数
        double spot_min_area = 3.0;
        double spot_max_area = 300.0;
        double spot_min_ratio = 0.3;
        double spot_max_ratio = 1.0;

        int min_row_spots = 2;
        double row_center_x_tol_ratio = 0.35;
        double row_height_tol_ratio = 0.5;
        double row_width_similarity_ratio = 0.5;

        double pattern_min_aspect = 0.6;
        double pattern_max_aspect = 4.0;


    };

    // struct Light {
    //     cv::RotatedRect rect; // 最小外接矩形
    //     cv::Point2f center;   // 中心点
    //     float width;          // 短边 (灯条宽度)
    //     float length;         // 长边 (灯条长度)
    //     float angle;          // 角度 (相对于水平线，[0, 90))
    //     float area;           // 面积
    //     cv::Point2f top;      // 上端点
    //     cv::Point2f bottom;   // 下端点
    //     int color;            // 颜色 (0: Red, 1: Blue, -1: Unknown)

    //     Light(const cv::RotatedRect& r);
    //     cv::Rect boundingRect() const; // 获取轴对齐包围盒
    // };

    struct Spot {
        cv::RotatedRect rect;
        cv::Point2f center;
        float width;
        float height;
        float area;
        int color;

        Spot() = default;
        explicit Spot(const cv::RotatedRect& r)
            : rect(r),
            center(r.center),
            width(std::min(r.size.width, r.size.height)),
            height(std::max(r.size.width, r.size.height)),
            area(r.size.width * r.size.height),
            color(-1) {}

        cv::Rect boundingRect() const { return rect.boundingRect(); }
    };

    struct PairResult {
        cv::Point2f center;         // 目标中心（最终目标中心）
        cv::Point2f light1_center;  // 上行中心
        cv::Point2f light2_center;  // 下行中心
        float width;                // 目标整体宽度（像素）
        float height;               // 目标整体高度（像素）
        float angle;                // 顶边方向角（弧度）
        int pair_type;              // 可自定义一个 pattern type
        float confidence;
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
    const std::vector<Spot>& getLastSpots() const;
    const std::vector<PairResult>& getLastPairs() const;

    // 获取调试图像
    cv::Mat getBinaryImage() const;
    cv::Mat getDebugImage() const;

private:
    Params params_;

    // 存储上一次检测的结果
    std::vector<Spot> last_spots_;
    std::vector<PairResult> last_pairs_;
    cv::Mat last_binary_;
    mutable cv::Mat last_debug_; // mutable 允许 const 函数修改它

    // 核心处理步骤
    cv::Mat preprocess(const cv::Mat& image) const;

    // 亮斑检测
    std::vector<Spot> detectSpots(const cv::Mat& color_img, const cv::Mat& binary_img);
    bool isValidSpot(const Spot& spot) const;
    int detectColorFromRect(const cv::RotatedRect& rect, const cv::Mat& color_img) const;

    // 亮斑分组为上下双行矩形目标
    std::vector<PairResult> groupSpotsToPatterns(const std::vector<Spot>& spots);
    float calculatePatternConfidence(const std::vector<Spot>& top,
                                 const std::vector<Spot>& bottom,
                                 float width_px,
                                 float height_px) const;

    // 调试图像
    void createDebugImage(const cv::Mat& color_img, const cv::Mat& binary_img) const;

    // 辅助函数
    float distance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    float angleDifference(float a1, float a2) const;
};
