#include "detect.h"
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#include <future> 

// ==================== 全局定义 ====================
#define TDT_INFO(msg) std::cout << msg << std::endl
#define UAV_ID 7          // UAV目标的ID编号
#define MAX_CARS 12       // 最大车辆数量
#define MAX_ARMORS 20     // 最大装甲板数量

// ==================== 性能优化配置 ====================
// 光流检测策略配置：
//   0 = 完全禁用光流（总是用零光流占位符）
//   1 = 总是计算光流（原始同步模式，Detect Time ~220-250ms）
//   2 = 低频异步光流（推荐模式，Detect Time ~40-70ms）
#define OPTICAL_FLOW_MODE 2

// 低频异步光流专用参数（仅当 OPTICAL_FLOW_MODE=2 时有效）
// 控制真实光流计算的频率：每N帧后台计算一次
// 推荐值：5-20（值越大速度越快，但UAV检测召回率可能略降）
#define FLOW_COMPUTE_INTERVAL 10
// ====================================================

namespace tdt_radar {

unsigned int count_img = 0;  // 图像帧计数器

/**
 * @brief 光流生成器类
 * @details 负责计算帧间光流并与RGB图像合并，用于UAV检测
 *          采用多级优化策略：跳帧、低分辨率、精简参数
 */
class OpticalFlowGenerator {
public:
    OpticalFlowGenerator(int flow_size = 320, int merge_size = 1280)
        : flow_size_(flow_size), merge_size_(merge_size), flow_dtype_(CV_32F), 
          skip_frame_count_(0), skip_frame_interval_(2) {}

    /**
     * @brief 计算两帧之间的光流
     * @param prev_img 前一帧图像
     * @param curr_img 当前帧图像
     * @return 归一化的光流矩阵 (CV_32FC2)
     * @note 采用跳帧策略，每2帧计算一次，其余帧复用缓存
     */
    cv::Mat computeOpticalFlow(const cv::Mat& prev_img, const cv::Mat& curr_img) {
        if (prev_img.empty() || curr_img.empty()) {
            return cv::Mat::zeros(flow_size_, flow_size_, CV_32FC2);
        }

        skip_frame_count_++;
        
        // 优化：每2帧计算一次光流（减少50%计算量）
        if (skip_frame_count_ % skip_frame_interval_ != 0) {
            return last_flow_.clone();  // 复用上一帧光流
        }

        // 优化：低分辨率计算（减少75%像素处理）
        cv::Mat prev_resized, curr_resized;
        cv::resize(prev_img, prev_resized, cv::Size(flow_size_, flow_size_));
        cv::resize(curr_img, curr_resized, cv::Size(flow_size_, flow_size_));

        cv::Mat prev_gray, curr_gray;
        cv::cvtColor(prev_resized, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(curr_resized, curr_gray, cv::COLOR_BGR2GRAY);

        // 优化：精简Farneback参数（金字塔层级3，窗口5）
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            prev_gray, curr_gray, flow,
            0.5, 2, 3, 2, 5, 1.2,
            cv::OPTFLOW_FARNEBACK_GAUSSIAN
        );

        // 归一化到[0, 1]范围
        double flow_min, flow_max;
        cv::Mat flow_reshaped = flow.reshape(1); 
        cv::minMaxLoc(flow_reshaped, &flow_min, &flow_max);
        double div = flow_max - flow_min + 1e-8;
        flow = (flow - flow_min) / div;

        last_flow_ = flow.clone();  // 缓存光流供跳帧使用
        return flow; 
    }

    /**
     * @brief 合并RGB图像与光流为3通道输入
     * @param raw_bgr_img 原始BGR图像
     * @param flow 光流矩阵 (CV_32FC2)
     * @return 合并后的[B, FlowX, FlowY]格式图像，用于YOLO推理
     * @note 仅B通道保留RGB信息，G/R通道替换为光流X/Y分量
     */
    cv::Mat mergeFlowToRGB(const cv::Mat& raw_bgr_img, const cv::Mat& flow) {
        cv::Mat b_resized;
        cv::resize(raw_bgr_img, b_resized, cv::Size(merge_size_, merge_size_));

        cv::Mat flow_resized;
        if (flow.empty()) {
            flow_resized = cv::Mat::zeros(merge_size_, merge_size_, CV_32FC2);
        } else {
            // 上采样光流到推理尺寸
            cv::resize(flow, flow_resized, cv::Size(merge_size_, merge_size_), 0, 0, cv::INTER_LINEAR);
        }

        std::vector<cv::Mat> bgr_channels;
        cv::split(b_resized, bgr_channels);  // [0]=B, [1]=G, [2]=R

        // 提取光流X/Y分量（CV_32FC2）
        cv::Mat flow_x_f32, flow_y_f32;
        if (flow.channels() == 2) {
            std::vector<cv::Mat> flow_channels;
            cv::split(flow_resized, flow_channels);
            flow_x_f32 = flow_channels[0];
            flow_y_f32 = flow_channels[1];
        } else {
            flow_x_f32 = cv::Mat::zeros(merge_size_, merge_size_, CV_32F);
            flow_y_f32 = cv::Mat::zeros(merge_size_, merge_size_, CV_32F);
        }

        // 转换为uint8 [0, 255]范围
        cv::Mat flow_x, flow_y;
        flow_x_f32.convertTo(flow_x, CV_8U, 255.0);
        flow_y_f32.convertTo(flow_y, CV_8U, 255.0);

        // 合并为[B, FlowX, FlowY]（与Python训练格式一致）
        cv::Mat merged;
        std::vector<cv::Mat> merged_channels = {bgr_channels[0], flow_x, flow_y};
        cv::merge(merged_channels, merged);
        return merged; 
    }

private:
    int flow_size_;      ///< 光流计算分辨率
    int merge_size_;     ///< 推理输入分辨率（1280x1280）
    int flow_dtype_;
    cv::Mat last_flow_;  ///< 跳帧缓存的光流数据
    int skip_frame_count_;
    int skip_frame_interval_;
};

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
    LightBarDetector() {
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

    /**
     * @brief 在ROI中检测单个灯条
     * @param roi 输入图像ROI
     * @param color_filter 颜色过滤：0-蓝色，1-红色，-1-无过滤
     * @return 检测到的所有灯条列表
     */
    std::vector<LightBar> detectLights(const cv::Mat& roi, int color_filter = -1) {
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

    /**
     * @brief 匹配UAV目标
     * @param lights 单灯条列表
     * @return 匹配成功的UAV目标列表
     * @note 遍历所有灯条对，验证：角度差、长度比、垂直/水平间距、连线角度
     */
    std::vector<UAVTarget> matchUAVTargets(const std::vector<LightBar>& lights) {
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
                float length_ratio = std::min(light1.length, light2.length) / std::max(light1.length, light2.length);
                if (length_ratio < max_light_length_ratio_) continue;  // 使用min_light_ratio作为下限
                
                // 计算垂直和水平距离
                float dx = std::abs(light1.center.x - light2.center.x);
                float dy = std::abs(light1.center.y - light2.center.y);
                
                // 计算目标参数
                cv::Point2f center = (light1.center + light2.center) * 0.5f;
                float avg_length = (light1.length + light2.length) * 0.5f;  // 平均长度
                
                // X坐标差异检查 (与auto_aim_fast一致)
                float x_diff_ratio = dx / avg_length;
                if (x_diff_ratio > max_horizontal_offset_) continue;
                
                // 垂直距离检查 (与auto_aim_fast一致)
                float vertical_dist = dy / avg_length;
                if (vertical_dist < min_vertical_ratio_ || vertical_dist > max_vertical_ratio_) continue;
                
                // 中心连线垂直度检查 (与auto_aim_fast一致，新增)
                float connect_angle = std::atan2(dy, dx) * 180.0f / CV_PI;
                if (connect_angle < min_connect_angle_ || connect_angle > max_connect_angle_) continue;
                
                UAVTarget target;
                // 确定上下关系
                target.top = (light1.center.y < light2.center.y) ? light1 : light2;
                target.bottom = (light1.center.y < light2.center.y) ? light2 : light1;
                target.center = center;
                target.width = avg_length;   // 使用平均长度作为宽度
                target.height = dy;          // 垂直距离作为高度
                target.type = 0;             // 正方体目标
                
                targets.push_back(target);
            }
        }
        
        return targets;
    }

private:
    // 灯条检测参数 (与auto_aim_fast一致)
    float min_area_;
    float max_area_;
    float min_aspect_ratio_;  // 宽高比下限 (width/length)
    float max_aspect_ratio_;  // 宽高比上限 (width/length)
    float max_angle_;
    
    // 上下匹配参数 (与auto_aim_fast的pair_mode=1一致)
    float max_light_angle_diff_;     // 配对灯条角度差上限
    float max_light_length_ratio_;   // 配对灯条长度比例下限
    float min_vertical_ratio_;       // 最小垂直距离(归一化)
    float max_vertical_ratio_;       // 最大垂直距离(归一化)
    float max_horizontal_offset_;    // 最大水平偏移(归一化)
    float min_connect_angle_;        // 中心连线角度下限
    float max_connect_angle_;        // 中心连线角度上限
};

int getColor(cv::Mat& img)
{
    if (img.empty()) return 1;
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    cv::Mat blueMinusRed = channels[0] - channels[2];
    cv::Mat redMinusBlue = channels[2] - channels[0];

    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);

    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0; // Blue
    } else {
        return 2; // Red
    }
}

cv::Rect getSafeRect(cv::Mat& image, cv::Rect& rect)
{
    cv::Rect save_rect;
    save_rect.x = std::max(0, rect.x);
    save_rect.y = std::max(0, rect.y);
    save_rect.width = std::min(image.cols - save_rect.x, rect.width);
    save_rect.height = std::min(image.rows - save_rect.y, rect.height);
    return save_rect;
}

Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    debug = 1;  // 默认开启调试绘制，避免窗口看不到框
    cv::namedWindow("detect", cv::WINDOW_NORMAL);

    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }

    cv::FileStorage fs;
    fs.open("./config/detect_params.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "[INIT] ❌ Failed to open config file: ./config/detect_params.yaml");
    }
    fs["yolo_path"] >> yolo_path;
    fs["armor_path"] >> armor_path;
    fs["classify_path"] >> classify_path;
    fs["uav_path"] >> uav_path;
    fs.release();
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Config loaded: yolo=%s, armor=%s, classify=%s, uav=%s", 
                yolo_path.c_str(), armor_path.c_str(), classify_path.c_str(), uav_path.c_str());

    std::ifstream file1(yolo_path.c_str());
    if (!file1.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/RM2025.onnx --saveEngine=model/TensorRT/yolo.engine --minBatch 1 --optBatch 1 --maxBatch 2 --Shape=1280x1280 --input_name=images");
    
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/armor_yolo.onnx --saveEngine=model/TensorRT/armor_yolo.engine --minBatch 1 --optBatch 5 --maxBatch 12 --Shape=192x192 --input_name=images");

    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/classify.onnx --saveEngine=model/TensorRT/classify.engine --minBatch 1 --optBatch 10 --maxBatch 20 --Shape=224x224 --input_name=input");

    this->classifier = classify::load(classify_path, classify::Type::densenet121);
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.55f);  // NMS 阈值提高：0.45f -> 0.55f（减少过滤）
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);

    std::ifstream file_uav(uav_path.c_str());
    if (!file_uav.good()) {
        RCLCPP_WARN(this->get_logger(), "[INIT] ⚠ UAV TRT engine file NOT found at: %s", uav_path.c_str());
        RCLCPP_INFO(this->get_logger(), "[INIT] Attempting ONNX to TensorRT conversion...");
        system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/best.onnx --saveEngine=model/TensorRT/uav.engine --minBatch 1 --optBatch 5 --maxBatch 12 --Shape=1280x1280 --input_name=images");
    } else {
        RCLCPP_INFO(this->get_logger(), "[INIT] ✓ UAV TRT engine file found: %s", uav_path.c_str());
        file_uav.close();
    }

    auto tmp_uav = uav::load(uav_path, 0.25f, 0.3f);
    if (tmp_uav != nullptr) {
        this->uav_detector = tmp_uav;
        RCLCPP_INFO(this->get_logger(), "[INIT] ✓ UAV detector loaded successfully! conf=0.25, nms=0.3");
    } else {
        RCLCPP_ERROR(this->get_logger(), "[INIT] ❌ FAILED to load UAV detector from: %s", uav_path.c_str());
        RCLCPP_ERROR(this->get_logger(), "[INIT] Check if file exists, has correct format, or is corrupted");
    }

    this->flow_generator_ = std::make_shared<OpticalFlowGenerator>(1280, 1280);
    this->light_bar_detector_ = std::make_shared<LightBarDetector>();  // 初始化灯条检测器
    
    // 初始化异步光流相关变量
    flow_task_running_.store(false);
    frame_counter_.store(0);
    
#if OPTICAL_FLOW_MODE == 0
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Optical flow: DISABLED (zero flow mode)");
#elif OPTICAL_FLOW_MODE == 1
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Optical flow: ALWAYS ENABLED (synchronous mode)");
#elif OPTICAL_FLOW_MODE == 2
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Optical flow: LOW-FREQUENCY ASYNC MODE");
    RCLCPP_INFO(this->get_logger(), "[INIT]   - Compute interval: every %d frames", FLOW_COMPUTE_INTERVAL);
    RCLCPP_INFO(this->get_logger(), "[INIT]   - Most frames use ZERO flow for max speed");
    RCLCPP_INFO(this->get_logger(), "[INIT]   - Real flow used 1-2 frames after computation");
#endif
    
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Light bar detector initialized for UAV armor detection");

    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));

    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());

    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto img = cv_bridge::toCvShare(msg, "bgr8")->image;
    tdt_radar::Image image(img.data, img.cols, img.rows);
    
    // -----------------------------------------------------------
    // 1. 智能光流策略 (根据 OPTICAL_FLOW_MODE 配置)
    // -----------------------------------------------------------
    cv::Mat merged_flow_img;

#if OPTICAL_FLOW_MODE == 0
    // 模式0: 完全禁用光流 - 使用零光流占位符
    cv::Mat zero_flow = cv::Mat::zeros(1280, 1280, CV_32FC2);
    merged_flow_img = flow_generator_->mergeFlowToRGB(img, zero_flow);
    
#elif OPTICAL_FLOW_MODE == 1
    // 模式1: 总是计算光流 (原始模式)
    static cv::Mat prev_frame; 
    if (prev_frame.empty()) {
        prev_frame = img.clone();
        RCLCPP_INFO(this->get_logger(), "✓ Initialized prev_frame for optical flow");
    }
    cv::Mat flow = flow_generator_->computeOpticalFlow(prev_frame, img);
    merged_flow_img = flow_generator_->mergeFlowToRGB(img, flow);
    prev_frame = img.clone();
    
#elif OPTICAL_FLOW_MODE == 2
    // 模式2: 低频异步光流 - 大部分时间用零光流，定期后台计算真实光流
    {
        std::lock_guard<std::mutex> lock(flow_cache_mutex_);
        if (cached_flow_img_.empty()) {
            // 缓存为空，使用零光流
            cv::Mat zero_flow = cv::Mat::zeros(1280, 1280, CV_32FC2);
            merged_flow_img = flow_generator_->mergeFlowToRGB(img, zero_flow);
            RCLCPP_DEBUG(this->get_logger(), "[Flow] Using ZERO flow (cache empty)");
        } else {
            // 有缓存时，90%时间用零光流，10%时间用真实光流（当光流刚计算完）
            // 这样既保证了速度，又能周期性利用真实光流补偿漏检
            int current_frame = frame_counter_.load();
            
            // 如果是光流计算帧的后续几帧，使用缓存的真实光流
            // 否则使用零光流（更快）
            if ((current_frame % FLOW_COMPUTE_INTERVAL) < 2) {  // 计算后的1-2帧使用真实光流
                merged_flow_img = cached_flow_img_.clone();
                RCLCPP_DEBUG(this->get_logger(), "[Flow] Using CACHED real flow (frame %d)", current_frame);
            } else {
                // 其余帧使用零光流
                cv::Mat zero_flow = cv::Mat::zeros(1280, 1280, CV_32FC2);
                merged_flow_img = flow_generator_->mergeFlowToRGB(img, zero_flow);
                RCLCPP_DEBUG(this->get_logger(), "[Flow] Using ZERO flow (fast path, frame %d)", current_frame);
            }
        }
    }
    
    // 帧计数器递增
    frame_counter_.fetch_add(1);
#endif

    // 2. 异步并行推理
    auto future_car = std::async(std::launch::async, [&]() {
        return yolo->forward(image);
    });

    // 关键修复：克隆 merged_flow_img，确保在异步任务中有有效数据
    cv::Mat flow_img_copy = merged_flow_img.clone();
    
    // 调试：保存光流图像以验证质量
    static int frame_count = 0;
    if (frame_count++ % 30 == 0) {  // 每30帧保存一次
        cv::imwrite("/tmp/flow_debug_" + std::to_string(frame_count) + ".jpg", flow_img_copy);
        RCLCPP_DEBUG(this->get_logger(), "[FLOW] Saved flow image: %dx%d, channels=%d", flow_img_copy.cols, flow_img_copy.rows, flow_img_copy.channels());
    }
    
    // 条件编译开关：若不需要 UAV 检测可注释掉以加速
    #define ENABLE_UAV_DETECTION 1
    
    auto future_uav = std::async(std::launch::async, [&]() -> std::vector<yolo::Box> {
        #if !ENABLE_UAV_DETECTION
            return std::vector<yolo::Box>();
        #endif
        if (uav_detector == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "❌ [UAV] uav_detector is nullptr!");
            return std::vector<yolo::Box>();
        }
        if (flow_img_copy.empty()) {
            RCLCPP_ERROR(this->get_logger(), "❌ [UAV] flow image is empty!");
            return std::vector<yolo::Box>();
        }
        RCLCPP_DEBUG(this->get_logger(), "✓ [UAV] Calling uav_detector->forward with image %dx%d, channels=%d", flow_img_copy.cols, flow_img_copy.rows, flow_img_copy.channels());
        tdt_radar::Image flow_image(flow_img_copy.data, flow_img_copy.cols, flow_img_copy.rows);
        auto res = uav_detector->forward(flow_image);
        RCLCPP_DEBUG(this->get_logger(), "✓ [UAV] forward returned %zu boxes", res.size());
        return res;
    });

    auto result = future_car.get();
    auto uav_raw_result = future_uav.get();

    vision_interface::msg::DetectResult detect_result;
    
    // 【重要】初始化所有检测数组为 0，防止垃圾数据
    for (int i = 0; i < 7; i++) {
        detect_result.blue_x[i] = 0.0f;
        detect_result.blue_y[i] = 0.0f;
        detect_result.red_x[i] = 0.0f;
        detect_result.red_y[i] = 0.0f;
    }
    
    // 初始化UAV灯条检测数组
    detect_result.blue_uav_light_count = 0;
    detect_result.red_uav_light_count = 0;
    for (int i = 0; i < 2; i++) {
        detect_result.blue_uav_light_x[i] = 0.0f;
        detect_result.blue_uav_light_y[i] = 0.0f;
        detect_result.red_uav_light_x[i] = 0.0f;
        detect_result.red_uav_light_y[i] = 0.0f;
    }
    
    size_t array_limit = detect_result.blue_x.size(); 

    // -----------------------------------------------------------
    // 3. 处理地面车辆 (ID 1-6 -> Index 0-5) - 优化版本 <25ms
    // -----------------------------------------------------------
    std::vector<Car> cars;
    
    if (!result.empty() && result.size() <= MAX_CARS) {
        // 【优化1】过滤车辆并提取 ROI 在单个循环中完成
        std::vector<cv::Mat> car_imgs;
        std::vector<cv::Rect> car_rects;
        
        for (const auto& box : result) {
            if (box.class_label == 0 || box.class_label == 1) {
                cv::Rect temp_rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
                cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
                
                // 早期过滤：跳过过小的车辆（可能是误检）
                if (temp_car_rect.area() < 50 * 50) continue;
                
                car_imgs.push_back(img(temp_car_rect).clone());
                car_rects.push_back(temp_car_rect);
                
                Car car;
                car.car = box;
                cars.push_back(car);
            }
        }
        
        // 【优化2】早期退出：没有车辆则直接返回
        if (!cars.empty()) {
            // 【优化3】并行执行装甲板检测和后续处理
            std::vector<tdt_radar::Image> car_images;
            for (const auto& img_mat : car_imgs) {
                car_images.push_back(tdt_radar::Image(img_mat.data, img_mat.cols, img_mat.rows));
            }
            
            auto future_armor = std::async(std::launch::async, [&]() {
                return armor_yolo->forwards(car_images);
            });
            
            // 异步获取装甲板结果
            auto armor_boxes = future_armor.get();
            for (int i = 0; i < (int)armor_boxes.size() && i < (int)cars.size(); i++) {
                cars[i].armors = armor_boxes[i];
                cars[i].car_rect = car_rects[i];
            }
            
            // 【优化4】跳过无装甲板的车辆，减少分类计算
            std::vector<cv::Mat> armor_imgs;
            std::vector<std::pair<int, int>> armor_indices; // (car_idx, armor_idx)
            
            for (int car_idx = 0; car_idx < (int)cars.size(); car_idx++) {
                auto& car = cars[car_idx];
                if (car.armors.empty()) continue;
                
                for (int armor_idx = 0; armor_idx < (int)car.armors.size(); armor_idx++) {
                    const auto& armor = car.armors[armor_idx];
                    cv::Rect rect_img(armor.left + car.car.left, armor.top + car.car.top,
                                     armor.right - armor.left, armor.bottom - armor.top);
                    cv::Rect rect_safe = getSafeRect(img, rect_img);
                    
                    armor_imgs.push_back(img(rect_safe).clone());
                    armor_indices.push_back({car_idx, armor_idx});
                }
            }
            
            // 【优化5】仅分类实际检测到的装甲板
            if (!armor_imgs.empty()) {
                std::vector<classify::Image> armor_images;
                for (const auto& armor_img : armor_imgs) {
                    armor_images.push_back(classify::Image(armor_img.data, armor_img.cols, armor_img.rows));
                }
                
                auto armor_result = classifier->forwards(armor_images);
                for (size_t i = 0; i < armor_indices.size() && i < armor_result.size(); i++) {
                    int car_idx = armor_indices[i].first;
                    int armor_idx = armor_indices[i].second;
                    cars[car_idx].armors[armor_idx].class_label = armor_result[i];
                }
            }
            
            // 【优化6】简化数字匹配：只取置信度最高的有效装甲板
            for (auto& car : cars) {
                if (car.armors.empty()) continue;
                
                // 找到最高置信度的有效装甲板
                int best_armor_idx = -1;
                float best_score = -1.0f;
                
                for (int i = 0; i < (int)car.armors.size(); i++) {
                    const auto& armor = car.armors[i];
                    if (armor.class_label != 0 && armor.class_label != 5) {
                        float combined_score = armor.confidence * 0.8f + 
                                              (armor.class_label > 0 ? 0.2f : 0.0f);
                        if (combined_score > best_score) {
                            best_score = combined_score;
                            best_armor_idx = i;
                        }
                    }
                }
                
                if (best_armor_idx == -1) continue; // 没有有效装甲板
                
                const auto& best_armor = car.armors[best_armor_idx];
                cv::Rect max_rect(best_armor.left + car.car.left, best_armor.top + car.car.top,
                                 best_armor.right - best_armor.left, best_armor.bottom - best_armor.top);
                
                car.number = best_armor.class_label;
                auto safe_rect = getSafeRect(img, max_rect);
                cv::Mat armor_roi = img(safe_rect).clone();
                car.color = getColor(armor_roi);
                car.center = cv::Point2f(max_rect.x + max_rect.width / 2.0f, 
                                        max_rect.y + max_rect.height / 2.0f);
                
                int idx = car.number - 1;
                
                // 【重要】防崩溃检查
                if (idx >= 0 && idx < (int)array_limit) {
                    if (car.color == 0) { // Blue
                        detect_result.blue_x[idx] = car.center.x;
                        detect_result.blue_y[idx] = car.center.y;
                    } else if (car.color == 2) { // Red
                        detect_result.red_x[idx] = car.center.x;
                        detect_result.red_y[idx] = car.center.y;
                    }
                }
                
                if (debug) {
                    cv::Scalar draw_color = (car.color == 2) ? cv::Scalar(0,0,255) : cv::Scalar(255,0,0);
                    std::string label = (car.color == 2 ? "R" : "B") + std::to_string(car.number);
                    cv::rectangle(img, car.car_rect, draw_color, 3);
                    cv::putText(img, label, cv::Point(car.car_rect.x, car.car_rect.y), 
                               cv::FONT_HERSHEY_SIMPLEX, 1, draw_color, 2);
                }
            }
        }
    }

    // -----------------------------------------------------------
    // 4. 处理 UAV (ID 7 -> Index 6) - 添加追踪稳定机制
    // -----------------------------------------------------------
    float scale_x = (float)img.cols / 1280.0f;
    float scale_y = (float)img.rows / 1280.0f;
    float img_area = (float)img.cols * img.rows;

    // 【优化】UAV最多2架，按置信度排序，只保留top 2
    std::sort(uav_raw_result.begin(), uav_raw_result.end(), 
              [](const yolo::Box& a, const yolo::Box& b) { return a.confidence > b.confidence; });
    
    // 只处理置信度最高的2个UAV
    std::vector<yolo::Box> uav_filtered;
    for (size_t i = 0; i < uav_raw_result.size() && i < 2; i++) {
        uav_filtered.push_back(uav_raw_result[i]);
    }
    
    // ============ 新增：UAV追踪逻辑（防止闪动） ============
    // 1. 更新现有追踪目标的miss_count
    for (auto& tracked : tracked_uavs) {
        tracked.miss_count++;
    }
    
    // 2. 匹配当前检测到的UAV与追踪目标
    const float IOU_THRESHOLD = 0.3f;  // IOU阈值：匹配成功的最小重叠度
    
    auto calculate_iou = [](const cv::Rect& a, const cv::Rect& b) -> float {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0f;
        
        int inter_area = (x2 - x1) * (y2 - y1);
        int union_area = a.area() + b.area() - inter_area;
        
        return (float)inter_area / (float)union_area;
    };
    
    std::vector<bool> detection_matched(uav_filtered.size(), false);
    std::vector<bool> tracker_matched(tracked_uavs.size(), false);
    
    // 匹配检测框与追踪框
    for (size_t i = 0; i < uav_filtered.size(); i++) {
        auto& uav = uav_filtered[i];
        int x = int(uav.left * scale_x);
        int y = int(uav.top * scale_y);
        int w = int((uav.right - uav.left) * scale_x);
        int h = int((uav.bottom - uav.top) * scale_y);
        cv::Rect det_rect(x, y, w, h);
        
        float best_iou = 0.0f;
        int best_track_idx = -1;
        
        for (size_t j = 0; j < tracked_uavs.size(); j++) {
            if (tracker_matched[j]) continue;
            
            float iou = calculate_iou(det_rect, cv::Rect(
                tracked_uavs[j].box.left * scale_x,
                tracked_uavs[j].box.top * scale_y,
                (tracked_uavs[j].box.right - tracked_uavs[j].box.left) * scale_x,
                (tracked_uavs[j].box.bottom - tracked_uavs[j].box.top) * scale_y
            ));
            
            if (iou > best_iou && iou > IOU_THRESHOLD) {
                best_iou = iou;
                best_track_idx = j;
            }
        }
        
        if (best_track_idx >= 0) {
            // 匹配成功：更新追踪目标
            tracked_uavs[best_track_idx].box = uav;
            tracked_uavs[best_track_idx].hit_count++;
            tracked_uavs[best_track_idx].miss_count = 0;
            tracked_uavs[best_track_idx].confirmed = (tracked_uavs[best_track_idx].hit_count >= HIT_THRESHOLD);
            
            detection_matched[i] = true;
            tracker_matched[best_track_idx] = true;
        }
    }
    
    // 3. 为未匹配的检测创建新追踪目标
    for (size_t i = 0; i < uav_filtered.size(); i++) {
        if (!detection_matched[i]) {
            TrackedUav new_track;
            new_track.box = uav_filtered[i];
            new_track.hit_count = 1;
            new_track.miss_count = 0;
            new_track.confirmed = false;
            tracked_uavs.push_back(new_track);
        }
    }
    
    // 4. 删除长时间未匹配的追踪目标
    tracked_uavs.erase(
        std::remove_if(tracked_uavs.begin(), tracked_uavs.end(),
            [this](const TrackedUav& t) { return t.miss_count > MAX_MISS_COUNT; }),
        tracked_uavs.end()
    );
    
    // 5. 使用追踪目标进行检测（只使用确认的目标）
    int uav_passed = 0;
    for (auto& tracked : tracked_uavs) {
        if (!tracked.confirmed) continue;  // 只处理已确认的目标
        
        auto& uav = tracked.box;
        int x = int(uav.left * scale_x);
        int y = int(uav.top * scale_y);
        int w = int((uav.right - uav.left) * scale_x);
        int h = int((uav.bottom - uav.top) * scale_y);

        cv::Rect uav_rect_raw(x, y, w, h);
        cv::Rect uav_rect = getSafeRect(img, uav_rect_raw);

        // 与 Python 版保持一致的面积过滤：相对面积 0.0002~0.05
        float uav_area = static_cast<float>(uav_rect.area());
        float ratio = uav_area / img_area;
        // 面积过滤
        if (uav_area > img_area * 0.05f) {
            continue;
        }
        if (uav_area < img_area * 0.0002f) {
            continue;
        }
        uav_passed++;

        cv::Mat uav_roi = img(uav_rect);
        int uav_color = getColor(uav_roi);

        // ============ 新增：精细ROI策略 - 在UAV ROI中心区域检测灯条 ============
        // 计算精细ROI：取长边中心40%，短边上方40%区域
        int roi_width = uav_rect.width;
        int roi_height = uav_rect.height;
        
        // 确定长边和短边
        bool is_vertical = (roi_height > roi_width);  // 判断是竖直方向还是水平方向
        
        // 计算精细ROI的位置和大小
        cv::Rect fine_roi_local;  // 相对于uav_rect的局部坐标
        if (is_vertical) {
            // 竖直方向：长边是height，短边是width
            // 长边（height）中心40%：从30%位置开始，取40%长度
            // 短边（width）上方50%：从0开始，取50%宽度，然后居中
            int fine_height = static_cast<int>(roi_height * 0.4f);
            int fine_width = static_cast<int>(roi_width * 0.5f);
            int fine_y = static_cast<int>(roi_height * 0.3f);  // 从30%位置开始（中心向上偏移）
            int fine_x = (roi_width - fine_width) / 2;         // 宽度居中
            
            fine_roi_local = cv::Rect(fine_x, fine_y, fine_width, fine_height);
        } else {
            // 水平方向：长边是width，短边是height
            int fine_width = static_cast<int>(roi_width * 0.4f);
            int fine_height = static_cast<int>(roi_height * 0.4f);
            int fine_x = static_cast<int>(roi_width * 0.3f);   // 从30%位置开始（中心向左偏移）
            int fine_y = 0;                                     // 从顶部开始（上方40%）
            
            fine_roi_local = cv::Rect(fine_x, fine_y, fine_width, fine_height);
        }
        
        // 确保精细ROI在有效范围内
        fine_roi_local.x = std::max(0, fine_roi_local.x);
        fine_roi_local.y = std::max(0, fine_roi_local.y);
        fine_roi_local.width = std::min(fine_roi_local.width, roi_width - fine_roi_local.x);
        fine_roi_local.height = std::min(fine_roi_local.height, roi_height - fine_roi_local.y);
        
        // 提取精细ROI图像
        cv::Mat fine_roi = uav_roi(fine_roi_local);
        
        // 计算精细ROI的中心点（全局坐标）
        cv::Point2f fine_roi_center_global(
            uav_rect.x + fine_roi_local.x + fine_roi_local.width / 2.0f,
            uav_rect.y + fine_roi_local.y + fine_roi_local.height / 2.0f
        );
        
        // ============ 在精细ROI中检测灯条（上下配对） ============
        std::vector<LightBarDetector::LightBar> detected_lights;
        std::vector<LightBarDetector::UAVTarget> detected_uav_targets;
        cv::Point2f uav_center(uav_rect.x + uav_rect.width / 2.0f, uav_rect.y + uav_rect.height / 2.0f);
        cv::Point2f refined_center = fine_roi_center_global;  // 默认使用精细ROI中心
        
        if (light_bar_detector_ != nullptr && !fine_roi.empty()) {
            // 1. 在精细ROI中检测灯条
            detected_lights = light_bar_detector_->detectLights(fine_roi, uav_color);
            
            // 2. 匹配UAV目标（上下配对）
            if (detected_lights.size() >= 2) {
                detected_uav_targets = light_bar_detector_->matchUAVTargets(detected_lights);
                
                // 如果检测到灯条对，使用灯条中心；否则使用精细ROI中心
                if (!detected_uav_targets.empty()) {
                    // 选择最靠近精细ROI中心的目标
                    cv::Point2f fine_roi_center_local(fine_roi.cols / 2.0f, fine_roi.rows / 2.0f);
                    float min_dist = std::numeric_limits<float>::max();
                    int best_target_idx = 0;
                    
                    for (size_t k = 0; k < detected_uav_targets.size(); ++k) {
                        float dx = detected_uav_targets[k].center.x - fine_roi_center_local.x;
                        float dy = detected_uav_targets[k].center.y - fine_roi_center_local.y;
                        float dist = std::sqrt(dx * dx + dy * dy);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_target_idx = k;
                        }
                    }
                    
                    // 更新精细定位中心点（fine_roi局部坐标 → 全局坐标）
                    refined_center.x = uav_rect.x + fine_roi_local.x + detected_uav_targets[best_target_idx].center.x;
                    refined_center.y = uav_rect.y + fine_roi_local.y + detected_uav_targets[best_target_idx].center.y;
                    
                    RCLCPP_DEBUG(this->get_logger(), 
                                "[UAV-Light] Detected light pair in fine ROI, center: (%.1f, %.1f)",
                                refined_center.x, refined_center.y);
                } else {
                    // 未检测到灯条对，使用精细ROI中心作为fallback
                    RCLCPP_DEBUG(this->get_logger(), 
                                "[UAV-Light] No light pairs found, using fine ROI center: (%.1f, %.1f)",
                                refined_center.x, refined_center.y);
                }
            } else {
                // 灯条数量不足，使用精细ROI中心
                RCLCPP_DEBUG(this->get_logger(), 
                            "[UAV-Light] Insufficient lights (%zu), using fine ROI center",
                            detected_lights.size());
            }
        } else {
            // 检测器不可用或ROI为空，使用精细ROI中心
            RCLCPP_DEBUG(this->get_logger(), "[UAV-Light] Using fine ROI center (detector unavailable)");
        }
        // ============ 灯条检测结束 ============

        cv::Scalar color_bgr;
        int uav_id = UAV_ID; // 7
        int idx = uav_id - 1; // 6
        bool can_send = (idx < (int)array_limit);

        if (uav_color == 0) { // Blue
            color_bgr = cv::Scalar(255, 0, 0);
            if (can_send) {
                // 使用精细定位的中心点
                detect_result.blue_x[idx] = refined_center.x;
                detect_result.blue_y[idx] = refined_center.y;
            }
            
            // ============ 新增：发送蓝方UAV所有灯条对的中心点 (最多2个) ============
            size_t light_count = std::min(detected_uav_targets.size(), size_t(2));
            detect_result.blue_uav_light_count = static_cast<uint8_t>(light_count);
            for (size_t i = 0; i < light_count; i++) {
                // 坐标转换：fine_roi局部 → 全局
                detect_result.blue_uav_light_x[i] = uav_rect.x + fine_roi_local.x + detected_uav_targets[i].center.x;
                detect_result.blue_uav_light_y[i] = uav_rect.y + fine_roi_local.y + detected_uav_targets[i].center.y;
            }
            RCLCPP_DEBUG(this->get_logger(), "[UAV-Light] Blue team: %u light pairs sent", 
                        detect_result.blue_uav_light_count);
            // ===================================================================
            
        } else if (uav_color == 2) { // Red
            color_bgr = cv::Scalar(0, 0, 255);
            if (can_send) {
                // 使用精细定位的中心点
                detect_result.red_x[idx] = refined_center.x;
                detect_result.red_y[idx] = refined_center.y;
            }
            
            // ============ 新增：发送红方UAV所有灯条对的中心点 (最多2个) ============
            size_t light_count = std::min(detected_uav_targets.size(), size_t(2));
            detect_result.red_uav_light_count = static_cast<uint8_t>(light_count);
            for (size_t i = 0; i < light_count; i++) {
                // 坐标转换：fine_roi局部 → 全局
                detect_result.red_uav_light_x[i] = uav_rect.x + fine_roi_local.x + detected_uav_targets[i].center.x;
                detect_result.red_uav_light_y[i] = uav_rect.y + fine_roi_local.y + detected_uav_targets[i].center.y;
            }
            RCLCPP_DEBUG(this->get_logger(), "[UAV-Light] Red team: %u light pairs sent", 
                        detect_result.red_uav_light_count);
            // ===================================================================
        } else {
            color_bgr = cv::Scalar(255, 0, 255); 
        }

        if (!can_send) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "DetectResult array size %zu too small for UAV index %d; consider expanding msg arrays",
                                 array_limit, idx);
        }

        if (debug) {
            // 绘制UAV主框
            cv::rectangle(img, uav_rect, color_bgr, 3);
            
            // ============ 新增：绘制精细ROI框（用于灯条检测的区域） ============
            cv::Rect fine_roi_global(
                uav_rect.x + fine_roi_local.x,
                uav_rect.y + fine_roi_local.y,
                fine_roi_local.width,
                fine_roi_local.height
            );
            cv::rectangle(img, fine_roi_global, cv::Scalar(255, 255, 0), 2);  // 青色虚线框表示精细ROI
            
            // 绘制精细ROI中心点（十字标记）
            cv::circle(img, fine_roi_center_global, 3, cv::Scalar(255, 255, 0), -1);
            cv::line(img, fine_roi_center_global - cv::Point2f(8, 0), 
                    fine_roi_center_global + cv::Point2f(8, 0), cv::Scalar(255, 255, 0), 1);
            cv::line(img, fine_roi_center_global - cv::Point2f(0, 8), 
                    fine_roi_center_global + cv::Point2f(0, 8), cv::Scalar(255, 255, 0), 1);
            
            std::string label = "UAV(ID" + std::to_string(uav_id) + ")";
            if (!can_send) label += " MSG-FULL"; // 提示数组已满
            
            // ============ 绘制检测到的灯条和UAV目标（在精细ROI坐标系中） ============
            // 绘制所有灯条（注意坐标转换：fine_roi局部坐标 → 全局坐标）
            for (const auto& light : detected_lights) {
                cv::Point2f pts[4];
                light.rect.points(pts);
                for (int k = 0; k < 4; k++) {
                    pts[k].x += uav_rect.x + fine_roi_local.x;  // fine_roi局部 → 全局
                    pts[k].y += uav_rect.y + fine_roi_local.y;
                }
                for (int k = 0; k < 4; k++) {
                    cv::line(img, pts[k], pts[(k+1)%4], cv::Scalar(0, 255, 255), 2);  // 黄色灯条
                }
            }
            
            // 绘制UAV目标（上下灯条对）
            for (const auto& target : detected_uav_targets) {
                cv::Point2f target_center_global(
                    uav_rect.x + fine_roi_local.x + target.center.x,
                    uav_rect.y + fine_roi_local.y + target.center.y
                );
                cv::circle(img, target_center_global, 5, cv::Scalar(0, 255, 0), -1);  // 绿色目标中心
                
                // 绘制上下灯条连线（坐标转换：fine_roi局部 → 全局）
                cv::Point2f top_center_global(
                    uav_rect.x + fine_roi_local.x + target.top.center.x,
                    uav_rect.y + fine_roi_local.y + target.top.center.y
                );
                cv::Point2f bottom_center_global(
                    uav_rect.x + fine_roi_local.x + target.bottom.center.x,
                    uav_rect.y + fine_roi_local.y + target.bottom.center.y
                );
                cv::line(img, top_center_global, bottom_center_global, cv::Scalar(0, 255, 0), 2);  // 绿色连线
                
                // 绘制目标矩形框
                cv::Point2f top_left(target_center_global.x - target.width/2, top_center_global.y);
                cv::Point2f bottom_right(target_center_global.x + target.width/2, bottom_center_global.y);
                cv::rectangle(img, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);  // 绿色目标框
            }
            
            // 绘制精细定位中心点
            if (!detected_uav_targets.empty()) {
                cv::circle(img, refined_center, 8, cv::Scalar(0, 255, 0), 2);  // 绿色精细中心点
                cv::line(img, refined_center - cv::Point2f(10, 0), 
                        refined_center + cv::Point2f(10, 0), cv::Scalar(0, 255, 0), 2);
                cv::line(img, refined_center - cv::Point2f(0, 10), 
                        refined_center + cv::Point2f(0, 10), cv::Scalar(0, 255, 0), 2);
            }
            // ============ 灯条/UAV目标绘制结束 ============
            
            int baseline = 0;
            cv::Size txt_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
            cv::rectangle(img,
                        cv::Point(uav_rect.x, std::max(uav_rect.y - txt_size.height - 4, 0)),
                        cv::Point(uav_rect.x + txt_size.width, uav_rect.y + 2),
                        cv::Scalar(0, 0, 0), -1);
            
            cv::putText(img, label, cv::Point(uav_rect.x, std::max(uav_rect.y - 6, 0)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2);
        }
    }

    detect_result.header.stamp = msg->header.stamp;
    
    pub->publish(detect_result);

#if OPTICAL_FLOW_MODE == 2
    // ============ 低频异步光流计算 ============
    // 只在特定帧启动后台光流计算，大幅减少光流开销
    
    int current_frame = frame_counter_.load();
    bool should_compute_flow = (current_frame % FLOW_COMPUTE_INTERVAL == 0);
    
    if (should_compute_flow && !flow_task_running_.load()) {
        // 标记任务开始
        flow_task_running_.store(true);
        
        // 克隆当前帧和前一帧用于异步计算
        cv::Mat curr_frame = img.clone();
        cv::Mat prev_frame_copy;
        
        if (prev_frame_.empty()) {
            prev_frame_ = img.clone();
            prev_frame_copy = prev_frame_.clone();
        } else {
            prev_frame_copy = prev_frame_.clone();
        }
        
        // 更新前一帧
        prev_frame_ = curr_frame.clone();
        
        // 捕获必要的成员指针
        auto flow_gen = flow_generator_;
        auto logger = this->get_logger();
        
        // 启动异步光流计算任务
        flow_future_ = std::async(std::launch::async, [this, flow_gen, prev_frame_copy, curr_frame, logger, current_frame]() -> cv::Mat {
            auto start_time = std::chrono::steady_clock::now();
            
            // 计算光流
            cv::Mat flow = flow_gen->computeOpticalFlow(prev_frame_copy, curr_frame);
            cv::Mat merged = flow_gen->mergeFlowToRGB(curr_frame, flow);
            
            auto end_time = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            RCLCPP_INFO(logger, "[Flow] Background computation at frame %d completed in %.2fms", current_frame, elapsed_ms);
            
            // 更新缓存
            {
                std::lock_guard<std::mutex> lock(flow_cache_mutex_);
                cached_flow_img_ = merged.clone();
            }
            
            // 标记任务完成
            flow_task_running_.store(false);
            
            return merged;
        });
        
        RCLCPP_INFO(this->get_logger(), "[Flow] Started background flow task at frame %d (interval: %d)", 
                    current_frame, FLOW_COMPUTE_INTERVAL);
    } else if (should_compute_flow) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "[Flow] Skipped flow computation (previous task still running)");
    }
    
    // 定期更新prev_frame以保持连续性（即使不计算光流）
    if (current_frame % FLOW_COMPUTE_INTERVAL == (FLOW_COMPUTE_INTERVAL - 1)) {
        prev_frame_ = img.clone();
    }
#endif

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms" << std::endl;

    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    
    // 在窗口显示检测统计信息（显示过滤后的UAV数量）
    std::string status_text = "Cars: " + std::to_string(result.size()) + " | UAVs: " + std::to_string(uav_filtered.size());
    cv::putText(final_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    
    if (uav_filtered.empty()) {
        cv::putText(final_img, "NO UAV DETECTED!", cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    cv::imshow("detect", final_img);

    auto key = cv::waitKey(1);
    if (key == 'r') {
        debug = !debug;
    }
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)