#include "detect.h"
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#include <future> 

#define TDT_INFO(msg) std::cout << msg << std::endl
// 定义 UAV 的 ID 为 7
#define UAV_ID 7 
#define MAX_CARS 12
#define MAX_ARMORS 20

namespace tdt_radar {

unsigned int count_img = 0;

/**
 * @brief 光流生成器类
 */
class OpticalFlowGenerator {
public:
    OpticalFlowGenerator(int flow_size = 320, int merge_size = 1280)
        : flow_size_(flow_size), merge_size_(merge_size), flow_dtype_(CV_32F), 
          skip_frame_count_(0), skip_frame_interval_(2) {}

    // 跳帧处理：每 skip_frame_interval_ 帧计算一次光流
    cv::Mat computeOpticalFlow(const cv::Mat& prev_img, const cv::Mat& curr_img) {
        if (prev_img.empty() || curr_img.empty()) {
            return cv::Mat::zeros(flow_size_, flow_size_, CV_32FC2);
        }

        skip_frame_count_++;
        
        // 【优化1】跳帧：每 2 帧计算一次光流（减少 50% 光流计算）
        if (skip_frame_count_ % skip_frame_interval_ != 0) {
            // 跳帧，复用上一帧的光流
            return last_flow_.clone();
        }

        // 计算光流
        // 【优化2】低分辨率光流计算（640x640 代替 1280x1280，减少 75% 像素处理）
        cv::Mat prev_resized, curr_resized;
        cv::resize(prev_img, prev_resized, cv::Size(flow_size_, flow_size_));
        cv::resize(curr_img, curr_resized, cv::Size(flow_size_, flow_size_));

        cv::Mat prev_gray, curr_gray;
        cv::cvtColor(prev_resized, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(curr_resized, curr_gray, cv::COLOR_BGR2GRAY);

        // 【优化3】激进参数：winsize 8->5, levels 4->3（进一步减少计算）
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            prev_gray, curr_gray, flow,
            0.5, 2, 3, 2, 5, 1.2,  // levels=3（原8）, 金字塔层级减少 62%
            cv::OPTFLOW_FARNEBACK_GAUSSIAN
        );

        double flow_min, flow_max;
        cv::Mat flow_reshaped = flow.reshape(1); 
        cv::minMaxLoc(flow_reshaped, &flow_min, &flow_max);

        double div = flow_max - flow_min + 1e-8;
        flow = (flow - flow_min) / div;

        last_flow_ = flow.clone(); // 缓存本帧光流供下一次跳帧使用
        return flow; 
    }

    cv::Mat mergeFlowToRGB(const cv::Mat& raw_bgr_img, const cv::Mat& flow) {
        cv::Mat b_resized;
        cv::resize(raw_bgr_img, b_resized, cv::Size(merge_size_, merge_size_));

        cv::Mat flow_resized;
        if (flow.empty()) {
            flow_resized = cv::Mat::zeros(merge_size_, merge_size_, CV_32FC2);
        } else {
            // 【优化】从 640x640 上采样到 1280x1280，仅用于最终推理输入
            cv::resize(flow, flow_resized, cv::Size(merge_size_, merge_size_), 0, 0, cv::INTER_LINEAR);
        }

        std::vector<cv::Mat> bgr_channels;
        cv::split(b_resized, bgr_channels);  // [0]=B, [1]=G, [2]=R in OpenCV

        // 提取光流的 X 和 Y 分量（来自 CV_32FC2）
        cv::Mat flow_x_f32, flow_y_f32;
        if (flow.channels() == 2) {
            std::vector<cv::Mat> flow_channels;
            cv::split(flow_resized, flow_channels);
            flow_x_f32 = flow_channels[0];  // X 分量
            flow_y_f32 = flow_channels[1];  // Y 分量
        } else {
            flow_x_f32 = cv::Mat::zeros(merge_size_, merge_size_, CV_32F);
            flow_y_f32 = cv::Mat::zeros(merge_size_, merge_size_, CV_32F);
        }

        // 转换为 uint8（0-255）
        cv::Mat flow_x, flow_y;
        flow_x_f32.convertTo(flow_x, CV_8U, 255.0);
        flow_y_f32.convertTo(flow_y, CV_8U, 255.0);

        // 与 Python 保持一致：[B, FlowX, FlowY]
        // bgr_channels[0] 是 B (蓝色通道)
        cv::Mat merged;
        std::vector<cv::Mat> merged_channels = {bgr_channels[0], flow_x, flow_y};
        cv::merge(merged_channels, merged);
        return merged; 
    }

private:
    int flow_size_;      // 光流计算分辨率（640x640）
    int merge_size_;     // 合并输出分辨率（1280x1280）
    int flow_dtype_;
    cv::Mat last_flow_;  // 缓存的光流，用于跳帧补偿
    int skip_frame_count_;
    int skip_frame_interval_;
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
    // 1. 光流计算
    // -----------------------------------------------------------
    static cv::Mat prev_frame; 
    cv::Mat merged_flow_img;

    // 第一帧时用当前帧初始化 prev_frame，避免光流为零
    if (prev_frame.empty()) {
        prev_frame = img.clone();
        RCLCPP_INFO(this->get_logger(), "✓ Initialized prev_frame for optical flow");
    }

    cv::Mat flow = flow_generator_->computeOpticalFlow(prev_frame, img);
    merged_flow_img = flow_generator_->mergeFlowToRGB(img, flow);
    prev_frame = img.clone();

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
    
    // ==========================================
    // 修复编译错误：移除 resize()
    // 获取当前数组大小 (可能是6，也可能是8，取决于msg定义)
    // ==========================================
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
                
                // 找到最高置信度的有效装甲板（不是0或5）
                int best_armor_idx = -1;
                float best_score = -1.0f;
                
                for (int i = 0; i < (int)car.armors.size(); i++) {
                    const auto& armor = car.armors[i];
                    // 【简化】只要有分类结果且不是 0 或 5，就认可
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
    // 4. 处理 UAV (ID 7 -> Index 6)
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

    int uav_passed = 0;
    for (auto& uav : uav_filtered) {
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

        cv::Scalar color_bgr;
        int uav_id = UAV_ID; // 7
        int idx = uav_id - 1; // 6
        
        cv::Point2f center(uav_rect.x + uav_rect.width / 2.0f, uav_rect.y + uav_rect.height / 2.0f);

        // 自动适配消息长度：若 DetectResult 仍是长度6，则不会写入并提示
        bool can_send = (idx < (int)array_limit);

        if (uav_color == 0) { // Blue
            color_bgr = cv::Scalar(255, 0, 0);
            if (can_send) {
                detect_result.blue_x[idx] = center.x;
                detect_result.blue_y[idx] = center.y;
            }
        } else if (uav_color == 2) { // Red
            color_bgr = cv::Scalar(0, 0, 255);
            if (can_send) {
                detect_result.red_x[idx] = center.x;
                detect_result.red_y[idx] = center.y;
            }
        } else {
            color_bgr = cv::Scalar(255, 0, 255); 
        }

        if (!can_send) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "DetectResult array size %zu too small for UAV index %d; consider expanding msg arrays",
                                 array_limit, idx);
        }

        if (debug) {
            cv::rectangle(img, uav_rect, color_bgr, 3);
            std::string label = "UAV(ID" + std::to_string(uav_id) + ")";
            if (!can_send) label += " MSG-FULL"; // 提示数组已满
            
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
    
    // 【数据流追踪】检查发送的数据
    bool has_vehicle_data = false;
    std::string vehicle_log = "[Detect发送] detect_result: ";
    for (int i = 0; i < 7; i++) {
        if (detect_result.blue_x[i] != 0 || detect_result.blue_y[i] != 0) {
            vehicle_log += "B" + std::to_string(i+1) + "(";
            vehicle_log += std::to_string((int)detect_result.blue_x[i]) + ",";
            vehicle_log += std::to_string((int)detect_result.blue_y[i]) + ") ";
            has_vehicle_data = true;
        }
        if (detect_result.red_x[i] != 0 || detect_result.red_y[i] != 0) {
            vehicle_log += "R" + std::to_string(i+1) + "(";
            vehicle_log += std::to_string((int)detect_result.red_x[i]) + ",";
            vehicle_log += std::to_string((int)detect_result.red_y[i]) + ") ";
            has_vehicle_data = true;
        }
    }
    
    if (has_vehicle_data) {
        RCLCPP_WARN(this->get_logger(), "%s", vehicle_log.c_str());
    }
    
    pub->publish(detect_result);

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