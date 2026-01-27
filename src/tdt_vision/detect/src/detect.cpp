#include "detect.h"
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#include <future>
#include "uav_light_detector.h"

// ==================== 全局常量定义 ====================
#define TDT_INFO(msg) std::cout << msg << std::endl
#define UAV_ID 7          // UAV目标ID（在检测结果数组中的索引为6）
#define MAX_CARS 12       // 场上最大车辆数量
#define MAX_ARMORS 20     // 单车最大装甲板数量

namespace tdt_radar {

unsigned int count_img = 0;  // 全局图像帧计数器

/**
 * @brief 判断图像主色调（蓝色或红色）
 * @param img 输入图像
 * @return 0-蓝色, 2-红色, 1-无法判断
 * @details 通过B-R和R-B通道差值的平均值判断主色调
 */
int getColor(cv::Mat& img)
{
    if (img.empty()) return 1;
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    cv::Mat blueMinusRed = channels[0] - channels[2];  // B - R
    cv::Mat redMinusBlue = channels[2] - channels[0];  // R - B

    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);

    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0; // Blue（蓝色主导）
    } else {
        return 2; // Red（红色主导）
    }
}

/**
 * @brief 获取安全的ROI区域（防止越界）
 * @param image 原图
 * @param rect 原始矩形框
 * @return 裁剪后的安全矩形框
 * @details 确保rect在image范围内，防止访问越界导致崩溃
 */
cv::Rect getSafeRect(const cv::Mat& image, const cv::Rect& rect)
{
    // 边界矩形
    const int img_right = image.cols;
    const int img_bottom = image.rows;
    
    // 解包原始矩形
    int rx = rect.x;
    int ry = rect.y;
    int rw = rect.width;
    int rh = rect.height;
    
    // 如果宽度或高度为负，修正为正向矩形
    if (rw < 0) { rx += rw; rw = -rw; }
    if (rh < 0) { ry += rh; rh = -rh; }
    
    // 快速边界检查：完全在图像外
    if (rx >= img_right || ry >= img_bottom || 
        rx + rw <= 0 || ry + rh <= 0) {
        return cv::Rect();
    }
    
    // 计算交集边界（使用快速比较和选择）
    int x1 = rx < 0 ? 0 : rx;
    int y1 = ry < 0 ? 0 : ry;
    int x2 = rx + rw;
    int y2 = ry + rh;
    x2 = x2 > img_right ? img_right : x2;
    y2 = y2 > img_bottom ? img_bottom : y2;
    
    // 计算交集尺寸
    int width = x2 - x1;
    int height = y2 - y1;
    
    // 如果尺寸小于等于0，返回空矩形
    if (width <= 0 || height <= 0) {
        return cv::Rect();
    }
    
    // 检查相交面积是否超过原矩形30%（使用整数乘法避免浮点运算）
    int intersect_area = width * height;
    int original_area = rw * rh;
    
    // 30%阈值检查：intersect_area * 10 > original_area * 3
    if (intersect_area * 10 <= original_area * 3) {
        return cv::Rect();
    }
    
    return cv::Rect(x1, y1, width, height);
}

static cv::Mat resize_keep_ratio_blackpad(
    const cv::Mat& src,
    const cv::Size& target_size,
    float& out_scale,
    int& out_pad_x,
    int& out_pad_y
) {
    if (src.empty()) {
        out_scale = 1.0f;
        out_pad_x = 0;
        out_pad_y = 0;
        return cv::Mat();
    }

    const int src_w = src.cols;
    const int src_h = src.rows;
    const int dst_w = target_size.width;
    const int dst_h = target_size.height;

    // 1) 计算等比例缩放系数（取更小的那个，保证完整放入目标尺寸）
    float r = std::min(dst_w / (float)src_w, dst_h / (float)src_h);
    out_scale = r;

    // 2) 计算缩放后的新尺寸
    int new_w = (int)std::round(src_w * r);
    int new_h = (int)std::round(src_h * r);

    // 3) resize 到新尺寸（不拉伸）
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 4) 创建目标图并填充黑色
    cv::Mat dst(dst_h, dst_w, src.type(), cv::Scalar(0, 0, 0));

    // 5) 计算 padding（居中放置）
    int pad_x = (dst_w - new_w) / 2;  // left pad
    int pad_y = (dst_h - new_h) / 2;  // top pad
    out_pad_x = pad_x;
    out_pad_y = pad_y;

    // 6) 拷贝到目标图
    cv::Rect roi(pad_x, pad_y, new_w, new_h);
    resized.copyTo(dst(roi));

    return dst;  // 输出始终是 1280x1280
}

// ==================== 检测节点构造函数 ====================
/**
 * @brief Detect节点构造函数
 * @details 初始化流程：
 *   1. 检查CUDA环境可用性
 *   2. 加载配置文件（detect_params.yaml）获取模型路径
 *   3. 自动转换ONNX模型为TensorRT引擎（如不存在）
 *   4. 初始化4个深度学习模型：
 *      - yolo: 车辆检测（RM2025.onnx）
 *      - armor_yolo: 装甲板检测（armor_yolo.onnx）
 *      - classifier: 数字分类（classify.onnx）
 *      - uav_detector: UAV检测（best.onnx, conf=0.15）
 *   5. 初始化UAV跟踪系统：
 *      - uav_tracker_: DeepSORT跟踪器（IOU阈值0.7）
 *      - light_bar_detector_: 灯条精细检测器
 *   6. 创建ROS2订阅者和发布者
 */
Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    // 1. 初始化调试窗口
    debug = 1;  // 默认开启调试绘制
    cv::namedWindow("detect", cv::WINDOW_NORMAL);

    // 2. 检查CUDA环境
    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }

    // 3. 加载配置文件（模型路径）
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

    // 4. 自动转换ONNX为TensorRT引擎（如不存在）
    std::ifstream file1(yolo_path.c_str());
    if (!file1.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/RM2025.onnx --saveEngine=model/TensorRT/yolo.engine --minBatch 1 --optBatch 1 --maxBatch 2 --Shape=1280x1280 --input_name=images");
    
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/armor_yolo.onnx --saveEngine=model/TensorRT/armor_yolo.engine --minBatch 1 --optBatch 5 --maxBatch 12 --Shape=192x192 --input_name=images");

    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/classify.onnx --saveEngine=model/TensorRT/classify.engine --minBatch 1 --optBatch 10 --maxBatch 20 --Shape=224x224 --input_name=input");

    // 5. 加载深度学习模型
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading classifier model...");
    this->classifier = classify::load(classify_path, classify::Type::densenet121);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Classifier loaded");
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading armor_yolo model...");
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.55f);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Armor YOLO loaded");
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading main yolo model...");
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Main YOLO loaded");

    // 6. 加载UAV检测器
    std::ifstream file_uav(uav_path.c_str());
    if (!file_uav.good()) {
        RCLCPP_WARN(this->get_logger(), "[INIT] ⚠ UAV TRT engine file NOT found at: %s", uav_path.c_str());
        RCLCPP_INFO(this->get_logger(), "[INIT] Attempting ONNX to TensorRT conversion...");
        system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/best.onnx --saveEngine=model/TensorRT/uav_test.engine --minBatch 1 --optBatch 5 --maxBatch 12 --Shape=1280x1280 --input_name=images");
    } else {
        RCLCPP_INFO(this->get_logger(), "[INIT] ✓ UAV TRT engine file found: %s", uav_path.c_str());
        file_uav.close();
    }

    RCLCPP_INFO(this->get_logger(), "[INIT] Loading UAV detector from: %s", uav_path.c_str());
    auto tmp_uav = uav::load(uav_path, 0.3f, 0.45f);
    if (tmp_uav != nullptr) {
        this->uav_detector = tmp_uav;
        RCLCPP_INFO(this->get_logger(), "[INIT] ✓ UAV detector loaded successfully! conf=0.5, nms=0.45");
    } else {
        RCLCPP_ERROR(this->get_logger(), "[INIT] ❌ FAILED to load UAV detector from: %s", uav_path.c_str());
        RCLCPP_ERROR(this->get_logger(), "[INIT] Check if file exists, has correct format, or is corrupted");
    }

    // 7. 初始化UAV检测辅助模块
    RCLCPP_INFO(this->get_logger(), "[INIT] Initializing light bar detector...");
    this->light_bar_detector_ = std::make_shared<LightBarDetector>();  // 灯条精细检测器
    
    // 8. 初始化UAV DeepSORT跟踪器（结合卡尔曼滤波）
    RCLCPP_INFO(this->get_logger(), "[INIT] Initializing UAV DeepSORT tracker...");
    // 参数说明：
    //   max_iou_distance = 0.7   - IOU距离阈值（1-IOU < 0.7认为是同一目标）
    //   max_feature_distance = 0.3 - 特征距离阈值（暂未使用外观特征）
    //   feature_weight = 0.0     - 特征权重（0=纯IOU匹配，1=纯特征匹配）
    this->uav_tracker_ = std::make_shared<UAVDeepSORTTracker>(0.7f, 0.3f, 0.0f);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ UAV DeepSORT tracker initialized (with Kalman Filter)");
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Light bar detector initialized for UAV armor detection");

    // 9. 创建ROS2订阅者和发布者
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));

    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());

    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

// ==================== 主检测回调函数 ====================
/**
 * @brief 图像检测主回调函数
 * @param msg 输入图像消息
 * @details 处理流程：
 *   阶段1: 图像预处理 + 异步并行推理（车辆YOLO + UAV YOLO）
 *   阶段2: 处理车辆检测结果（ID 1-6）
 *     - 车辆检测 → 装甲板检测 → 数字分类 → 颜色判断
 *   阶段3: 处理UAV检测结果（ID 7）
 *     - DeepSORT跟踪 + 卡尔曼预测
 *     - 精细定位（灯条检测）
 *     - 绘制可视化结果
 *   阶段4: 发布检测结果 + 性能统计
 */
void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    // ==================== 阶段1: 图像预处理与异步推理 ====================
    auto img = cv_bridge::toCvShare(msg, "bgr8")->image;
    tdt_radar::Image image(img.data, img.cols, img.rows);
    
    // 1.1 准备UAV检测输入（RGB检测模式：直接resize到1280x1280）
    cv::Mat uav_input_img;
    float uav_scale = 1.0f;
    int uav_pad_x = 0, uav_pad_y = 0;

    uav_input_img = resize_keep_ratio_blackpad(img, cv::Size(1280, 1280),
                                            uav_scale, uav_pad_x, uav_pad_y);

    
    // 1.2 启动车辆检测异步任务
    auto future_car = std::async(std::launch::async, [&]() {
        return yolo->forward(image);
    });

    // 1.3 克隆图像防止异步任务中数据失效
    cv::Mat uav_input_copy = uav_input_img.clone();
    
    // 1.4 调试：每30帧保存一次UAV检测输入图像
    static int frame_count = 0;
    if (frame_count++ % 30 == 0) {
        cv::imwrite("/tmp/uav_input_debug_" + std::to_string(frame_count) + ".jpg", uav_input_copy);
        RCLCPP_DEBUG(this->get_logger(), "[UAV-INPUT] Saved UAV input image: %dx%d, channels=%d", 
                    uav_input_copy.cols, uav_input_copy.rows, uav_input_copy.channels());
    }
    
    // 1.5 启动UAV检测异步任务（可通过宏开关关闭）
    #define ENABLE_UAV_DETECTION 1
    
    auto future_uav = std::async(std::launch::async, [&]() -> std::vector<yolo::Box> {
        #if !ENABLE_UAV_DETECTION
            return std::vector<yolo::Box>();
        #endif
        if (uav_detector == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "❌ [UAV] uav_detector is nullptr!");
            return std::vector<yolo::Box>();
        }
        if (uav_input_copy.empty()) {
            RCLCPP_ERROR(this->get_logger(), "❌ [UAV] UAV input image is empty!");
            return std::vector<yolo::Box>();
        }
        RCLCPP_DEBUG(this->get_logger(), "✓ [UAV] Calling uav_detector->forward with image %dx%d, channels=%d", 
                    uav_input_copy.cols, uav_input_copy.rows, uav_input_copy.channels());
        tdt_radar::Image uav_input_image(uav_input_copy.data, uav_input_copy.cols, uav_input_copy.rows);
        auto res = uav_detector->forward(uav_input_image);
        RCLCPP_DEBUG(this->get_logger(), "✓ [UAV] forward returned %zu boxes", res.size());
        return res;
    });

    // 1.6 等待异步任务完成
    auto result = future_car.get();
    auto uav_raw_result = future_uav.get();

    // ==================== 阶段2: 初始化检测结果消息 ====================
    vision_interface::msg::DetectResult detect_result;
    
    // 2.1 初始化车辆检测数组（防止垃圾数据）
    for (int i = 0; i < 7; i++) {
        detect_result.blue_x[i] = 0.0f;
        detect_result.blue_y[i] = 0.0f;
        detect_result.red_x[i] = 0.0f;
        detect_result.red_y[i] = 0.0f;
    }
    
    // 2.2 初始化UAV灯条检测数组
    detect_result.blue_uav_light_count = 0;
    detect_result.red_uav_light_count = 0;
    for (int i = 0; i < 2; i++) {
        detect_result.blue_uav_light_x[i] = 0.0f;
        detect_result.blue_uav_light_y[i] = 0.0f;
        detect_result.red_uav_light_x[i] = 0.0f;
        detect_result.red_uav_light_y[i] = 0.0f;
    }
    
    size_t array_limit = detect_result.blue_x.size(); 

    // ==================== 阶段3: 处理地面车辆检测（ID 1-6）====================
    std::vector<Car> cars;
    
    if (!result.empty() && result.size() <= MAX_CARS) {
        // 3.1 过滤车辆并提取ROI（单循环完成）
        std::vector<cv::Mat> car_imgs;
        std::vector<cv::Rect> car_rects;
        
        for (const auto& box : result) {
            if (box.class_label == 0 || box.class_label == 1) {
                cv::Rect temp_rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
                cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
                
                // 过滤过小的检测框（可能是误检）
                if (temp_car_rect.area() < 50 * 50) continue;
                
                car_imgs.push_back(img(temp_car_rect).clone());
                car_rects.push_back(temp_car_rect);
                
                Car car;
                car.car = box;
                cars.push_back(car);
            }
        }
        
        // 3.2 如果没有车辆则跳过后续处理
        if (!cars.empty()) {
            // 3.3 批量装甲板检测（异步并行）
            std::vector<tdt_radar::Image> car_images;
            for (const auto& img_mat : car_imgs) {
                car_images.push_back(tdt_radar::Image(img_mat.data, img_mat.cols, img_mat.rows));
            }
            
            auto future_armor = std::async(std::launch::async, [&]() {
                return armor_yolo->forwards(car_images);
            });
            
            // 3.4 获取装甲板检测结果
            auto armor_boxes = future_armor.get();
            for (int i = 0; i < (int)armor_boxes.size() && i < (int)cars.size(); i++) {
                cars[i].armors = armor_boxes[i];
                cars[i].car_rect = car_rects[i];
            }
            
            // 3.5 提取所有装甲板图像（跳过无装甲板的车辆）
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
            
            // 3.6 批量数字分类（仅分类实际检测到的装甲板）
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
            
            // 3.7 数字匹配与结果发送（选择置信度最高的有效装甲板）
            for (auto& car : cars) {
                if (car.armors.empty()) continue;
                
                // 找到最高置信度的有效装甲板（排除class_label=0和5）
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
                
                if (best_armor_idx == -1) continue;
                
                // 计算车辆中心点和颜色
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
                
                // 填充检测结果数组（防崩溃检查）
                if (idx >= 0 && idx < (int)array_limit) {
                    if (car.color == 0) { // Blue
                        detect_result.blue_x[idx] = car.center.x;
                        detect_result.blue_y[idx] = car.center.y;
                    } else if (car.color == 2) { // Red
                        detect_result.red_x[idx] = car.center.x;
                        detect_result.red_y[idx] = car.center.y;
                    }
                }
                
                // 3.8 绘制调试信息
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

    // ==================== 阶段4: 处理UAV检测（ID 7）====================
    // letterbox 映射参数（来自 Stage1 的 resize_keep_ratio_blackpad 输出）
    const float uav_scale_inv = 1.0f / uav_scale;   // 用倒数避免重复除法
    const int pad_x = uav_pad_x;
    const int pad_y = uav_pad_y;

    float img_area = (float)img.cols * img.rows;


    // 4.1 UAV检测结果预处理（按置信度排序，只保留top 2）
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "[UAV] Raw detections: %zu", uav_raw_result.size());
    
    std::sort(uav_raw_result.begin(), uav_raw_result.end(), 
              [](const yolo::Box& a, const yolo::Box& b) { return a.confidence > b.confidence; });
    
    std::vector<yolo::Box> uav_filtered;
    for (size_t i = 0; i < uav_raw_result.size() && i < 2; i++) {
        uav_filtered.push_back(uav_raw_result[i]);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "[UAV] Filtered #%zu: conf=%.3f, box=[%.1f,%.1f,%.1f,%.1f]",
                            i, uav_raw_result[i].confidence,
                            uav_raw_result[i].left, uav_raw_result[i].top,
                            uav_raw_result[i].right, uav_raw_result[i].bottom);
    }
    
    // 4.2 准备DeepSORT跟踪器输入数据
    std::vector<cv::Rect> uav_bboxes;
    std::vector<float> uav_confidences;
    std::vector<cv::Mat> uav_features;  // 暂不使用外观特征（feature_weight=0.0）
    
    for (const auto& uav : uav_filtered) {
        // 坐标转换：1280x1280 → 原图尺寸
        // 1280x1280（含黑边）坐标 -> 原图坐标
        int x1 = (int)std::round((uav.left   - pad_x) * uav_scale_inv);
        int y1 = (int)std::round((uav.top    - pad_y) * uav_scale_inv);
        int x2 = (int)std::round((uav.right  - pad_x) * uav_scale_inv);
        int y2 = (int)std::round((uav.bottom - pad_y) * uav_scale_inv);

        // clamp 到图像范围（防止越界）
        x1 = std::max(0, std::min(x1, img.cols - 1));
        y1 = std::max(0, std::min(y1, img.rows - 1));
        x2 = std::max(0, std::min(x2, img.cols - 1));
        y2 = std::max(0, std::min(y2, img.rows - 1));

        int w = x2 - x1;
        int h = y2 - y1;
        if (w <= 0 || h <= 0) {
            continue;
        }

        // 长宽比过滤
        if (w < h || w > 4 * h) {
            continue;
        }
        
        // 面积过滤（排除异常大小的检测）
        float uav_area = static_cast<float>(w * h);
        float ratio = uav_area / img_area;
        if (ratio > 0.08f || ratio < 0.0002f) {
            continue;
        }
        
        cv::Rect bbox(x1, y1, w, h);
        uav_bboxes.push_back(bbox);
        uav_confidences.push_back(uav.confidence);
    }
    
    // 4.3 更新DeepSORT跟踪器（匹配 + 卡尔曼预测）
    std::vector<std::shared_ptr<UAVTrack>> confirmed_tracks;
    if (uav_tracker_ != nullptr) {
        confirmed_tracks = uav_tracker_->update(uav_bboxes, uav_confidences, uav_features);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "[UAV-Track] Detections: %zu, Confirmed tracks: %zu",
                            uav_bboxes.size(), confirmed_tracks.size());
    } else {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "[UAV-Track] Tracker not initialized!");
    }
    
    // 4.4 处理每个确认的跟踪目标（优先处理有检测的，跳过过旧的预测）
    int uav_passed = 0;
    // 只处理最多两个track
    std::vector<cv::Rect> kalman_rects; // 每帧局部变量
    int track_count = 0;
    for (const auto& track : confirmed_tracks) {
        if (track_count >= 2) break;
        // 快速过滤：跳过丢失超过UAV_MAX_TRACKING_AGE帧的轨迹（减少无效计算）
        if (track->getTimeSinceUpdate() > UAV_MAX_TRACKING_AGE) {
            RCLCPP_DEBUG(this->get_logger(), "[UAV] Skipping Track ID=%d (lost %d frames)",
                        track->getID(), track->getTimeSinceUpdate());
            continue;
        }
        // 4.4.1 判断当前帧是否有检测匹配
        //      time_since_update=0: 有YOLO检测匹配
        //      time_since_update>0: 仅卡尔曼预测
        bool has_detection = (track->getTimeSinceUpdate() == 0);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "[UAV] Track ID=%d, has_detection=%s, time_since_update=%d",
                            track->getID(), has_detection ? "YES" : "NO", 
                            track->getTimeSinceUpdate());

        // 4.4.2 根据检测状态选择边界框
        auto area_ratio_ok = [&](const cv::Rect& r) -> bool {
            if (r.width <= 0 || r.height <= 0) return false;
            if(r.width<r.height || r.width>4*r.height) return false;
            float ratio = (float)r.area() / img_area;
            return (ratio >= 0.0002f && ratio <= 0.05f);
        };

        // 选择 track 框（检测框或预测框）
        cv::Rect uav_rect_raw = has_detection ? track->getBox(): track->getPredictedBox();

        // ===== 在裁剪前对 track 框做面积过滤（关键）=====
        if (!area_ratio_ok(uav_rect_raw)) {
            // 这里两种策略二选一：
            // A) 仅跳过本帧处理（推荐：不会破坏 tracker 内部状态）
            // continue;

            // B) 同时让 track “视为丢失”（更激进，会加速删除异常轨迹）
            track->markMissed();
            continue;
        }

        // 再做安全裁剪（防越界）
        cv::Rect uav_rect = getSafeRect(img, uav_rect_raw);
        if (uav_rect.width <= 0 || uav_rect.height <= 0) continue;


        uav_passed++;
        track_count++;

        // 4.4.3 快速检查有效性（getSafeRect已确保边界安全，只需检查面积）
        if (uav_rect.width <= 0 || uav_rect.height <= 0) {
            continue;
        }

        // 收集kalman预测框用于重叠判断
        if (!has_detection) {
            kalman_rects.push_back(uav_rect);
        }
        
        cv::Mat uav_roi = img(uav_rect);
        int uav_color = getColor(uav_roi);

        // 4.4.4 精细检测（对所有有效轨迹执行，包括YOLO检测和卡尔曼预测）
        cv::Rect fine_roi_local;
        cv::Point2f fine_roi_center_global;
        std::vector<LightBarDetector::LightBar> detected_lights;
        std::vector<LightBarDetector::UAVTarget> detected_uav_targets;
        cv::Point2f refined_center;
        
        // 跳过time_since_update>UAV_MAX_TRACKING_AGE的轨迹（允许卡尔曼预测，提高跟踪鲁棒性）
        if (light_bar_detector_ != nullptr && track->getTimeSinceUpdate() <= UAV_MAX_TRACKING_AGE) {
            // 调用LightBarDetector执行精细检测流程（对检测框和预测框都执行）
            refined_center = light_bar_detector_->performFineDetection(
                uav_roi, uav_rect, uav_color,
                fine_roi_local, fine_roi_center_global,
                detected_lights, detected_uav_targets
            );
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "[UAV-Fine] %s: Detected %zu lights, %zu targets, center=(%.1f,%.1f)",
                                has_detection ? "YOLO" : "KALMAN",
                                detected_lights.size(), detected_uav_targets.size(),
                                refined_center.x, refined_center.y);
        } else {
            // 无检测时直接使用预测框中心
            refined_center = cv::Point2f(
                uav_rect.x + uav_rect.width / 2.0f,
                uav_rect.y + uav_rect.height / 2.0f
            );
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "[UAV-Fine] No fine detection, using %s center=(%.1f,%.1f)",
                                has_detection ? "YOLO" : "predicted",
                                refined_center.x, refined_center.y);
        }

        // 4.4.5 填充检测结果
        cv::Scalar color_bgr;
        int uav_id = UAV_ID; // 7
        int idx = uav_id - 1; // 6
        bool can_send = (idx < (int)array_limit);
        
        // UAV框偏中心点
        cv::Point2f uav_box_center(
            uav_rect.x + uav_rect.width *0.50f,
            uav_rect.y + uav_rect.height *0.60f
        );
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "[UAV-Box] uav_rect=[%d,%d,%d,%d], center=(%.1f,%.1f), refined_center=(%.1f,%.1f)",
                            uav_rect.x, uav_rect.y, uav_rect.width, uav_rect.height,
                            uav_box_center.x, uav_box_center.y,
                            refined_center.x, refined_center.y);

        if (uav_color == 0) { // Blue
            color_bgr = cv::Scalar(255, 0, 0);
            if (can_send) {
                if (light_bar_detector_ != nullptr && track->getTimeSinceUpdate() <= UAV_MAX_TRACKING_AGE) {
                    detect_result.blue_x[idx] = uav_box_center.x;
                    detect_result.blue_y[idx] = uav_box_center.y;
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                        "[UAV] Blue team sent: refined_center=(%.1f, %.1f)", 
                                        refined_center.x, refined_center.y);
                } else {
                    detect_result.blue_x[idx] = 0.0f;
                    detect_result.blue_y[idx] = 0.0f;
                }
            }
            // 发送蓝方UAV所有灯条对的中心点（最多2个）
            size_t light_count = std::min(detected_uav_targets.size(), size_t(2));
            detect_result.blue_uav_light_count = static_cast<uint8_t>(light_count);
            for (size_t i = 0; i < light_count; i++) {
                detect_result.blue_uav_light_x[i] = uav_rect.x + fine_roi_local.x + detected_uav_targets[i].center.x;
                detect_result.blue_uav_light_y[i] = uav_rect.y + fine_roi_local.y + detected_uav_targets[i].center.y;
            }
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "[UAV-Light] Blue team: %u light pairs sent", 
                                detect_result.blue_uav_light_count);
        } else if (uav_color == 2) { // Red
            color_bgr = cv::Scalar(0, 0, 255);
            if (can_send) {
                if (light_bar_detector_ != nullptr && track->getTimeSinceUpdate() <= UAV_MAX_TRACKING_AGE) {
                    detect_result.red_x[idx] = uav_box_center.x;
                    detect_result.red_y[idx] = uav_box_center.y;
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                        "[UAV] Red team sent: refined_center=(%.1f, %.1f)", 
                                        refined_center.x, refined_center.y);
                } else {
                    detect_result.red_x[idx] = 0.0f;
                    detect_result.red_y[idx] = 0.0f;
                }
            }
            // 发送红方UAV所有灯条对的中心点（最多2个）
            size_t light_count = std::min(detected_uav_targets.size(), size_t(2));
            detect_result.red_uav_light_count = static_cast<uint8_t>(light_count);
            for (size_t i = 0; i < light_count; i++) {
                detect_result.red_uav_light_x[i] = uav_rect.x + fine_roi_local.x + detected_uav_targets[i].center.x;
                detect_result.red_uav_light_y[i] = uav_rect.y + fine_roi_local.y + detected_uav_targets[i].center.y;
            }
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "[UAV-Light] Red team: %u light pairs sent", 
                                detect_result.red_uav_light_count);
        } else {
            color_bgr = cv::Scalar(255, 0, 255); 
        }

        if (!can_send) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "DetectResult array size %zu too small for UAV index %d; consider expanding msg arrays",
                                 array_limit, idx);
        }

        // 4.4.6 绘制可视化结果（调试模式）
        if (debug) {
            // 只做多个kalman预测框重叠时只保留最后一个
            bool draw_this = true;
            if (!has_detection) {
                // 只保留最后一个重叠的kalman预测框
                for (size_t k = kalman_rects.size() - 1; k > 0; --k) {
                    if (kalman_rects[k] == uav_rect) continue;
                    cv::Rect inter = kalman_rects[k] & uav_rect;
                    if (inter.area() > 0) {
                        draw_this = false;
                        break;
                    }
                }
            }
            if (draw_this) {
                cv::rectangle(img, uav_rect, color_bgr, 2);
                if (light_bar_detector_ != nullptr && track->getTimeSinceUpdate() <= UAV_MAX_TRACKING_AGE) {
                    light_bar_detector_->drawLightDetection(
                        img, uav_rect, fine_roi_local,
                        detected_lights, detected_uav_targets,
                        refined_center, fine_roi_center_global
                    );
                }
            }

            // 第二步:如果执行了精细检测,绘制灯条检测结果（对检测框和预测框都绘制）
            if (light_bar_detector_ != nullptr && track->getTimeSinceUpdate() <= UAV_MAX_TRACKING_AGE) {
                light_bar_detector_->drawLightDetection(
                    img, uav_rect, fine_roi_local,
                    detected_lights, detected_uav_targets,
                    refined_center, fine_roi_center_global
                );
            }
            
            // 第三步:绘制UAV标签
            std::string label = "UAV(ID" + std::to_string(uav_id) + ")";
            if (!has_detection) {
                label += " [KALMAN]";
            }
            if (!can_send) {
                label += " MSG-FULL";
            }
            
            int baseline = 0;
            cv::Size txt_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            cv::rectangle(img,
                        cv::Point(uav_rect.x, std::max(uav_rect.y - txt_size.height - 4, 0)),
                        cv::Point(uav_rect.x + txt_size.width, uav_rect.y + 2),
                        cv::Scalar(0, 0, 0), -1);
            
            cv::Scalar label_color = has_detection ? color_bgr : cv::Scalar(200, 200, 255);
            cv::putText(img, label, cv::Point(uav_rect.x, std::max(uav_rect.y - 6, 0)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2);
        }
    }

    // ==================== 阶段5: 发布结果与性能统计 ====================

    RCLCPP_INFO(this->get_logger(), "[DETECT-RESULT] header.stamp: sec=%u, nsec=%u", detect_result.header.stamp.sec, detect_result.header.stamp.nanosec);
    // 输出UAV相关坐标日志，便于调试
    int uav_id = UAV_ID; // 7
    int idx = uav_id - 1; // 6
    RCLCPP_INFO(this->get_logger(), "[DETECT-RESULT] blue_x[%d]=%.2f, blue_y[%d]=%.2f, red_x[%d]=%.2f, red_y[%d]=%.2f", 
        idx, detect_result.blue_x[idx], idx, detect_result.blue_y[idx], idx, detect_result.red_x[idx], idx, detect_result.red_y[idx]);

    // 5.1 发布检测结果到话题
    pub->publish(detect_result);

    // 5.2 计算检测耗时
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms" << std::endl;

    // 5.3 可视化调试输出
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    
    // 显示检测统计信息（显示过滤后的UAV数量）
    std::string status_text = "Cars: " + std::to_string(result.size()) + " | UAVs: " + std::to_string(uav_filtered.size());
    cv::putText(final_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    
    if (uav_filtered.empty()) {
        cv::putText(final_img, "NO UAV DETECTED!", cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    // 5.4 显示调试窗口
    cv::imshow("detect", final_img);

    // 5.5 处理键盘输入（r键切换调试模式）
    auto key = cv::waitKey(1);
    if (key == 'r') {
        debug = !debug;
    }
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)