#include "detect.h"
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <filesystem>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#include <future>
#include <sstream>
#include "uav_light_detector.h"

// ==================== 全局常量定义 ====================
#define TDT_INFO(msg) std::cout << msg << std::endl
#define UAV_ID 7          // UAV目标ID（在检测结果数组中的索引为6）
#define MAX_CARS 12       // 场上最大车辆数量
#define MAX_ARMORS 20     // 单车最大装甲板数量

namespace tdt_radar {

namespace {

std::string shellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

bool fileExists(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

int buildTensorRTEngine(
    const std::string& onnx_path,
    const std::string& engine_path,
    int min_batch,
    int opt_batch,
    int max_batch,
    const std::string& shape,
    const std::string& input_name,
    bool fp16,
    const std::vector<std::string>& extra_args = {})
{
    std::ostringstream command;
    command << "python3 src/utils/onnx2trt.py"
            << " --onnx " << shellQuote(onnx_path)
            << " --saveEngine " << shellQuote(engine_path)
            << " --Shape " << shellQuote(shape)
            << " --minBatch " << min_batch
            << " --optBatch " << opt_batch
            << " --maxBatch " << max_batch
            << " --input_name " << shellQuote(input_name);

    if (fp16) {
        command << " --fp16";
    }

    for (const auto& arg : extra_args) {
        command << " --extraArg " << shellQuote(arg);
    }

    return std::system(command.str().c_str());
}

static float rectIou(const cv::Rect& lhs, const cv::Rect& rhs)
{
    if (lhs.area() <= 0 || rhs.area() <= 0) {
        return 0.0f;
    }
    const cv::Rect intersection = lhs & rhs;
    const float intersection_area = static_cast<float>(intersection.area());
    const float union_area =
        static_cast<float>(lhs.area() + rhs.area()) - intersection_area;
    if (union_area <= 1.0f) {
        return 0.0f;
    }
    return intersection_area / union_area;
}

static float rectCenterDistance(const cv::Rect& lhs, const cv::Rect& rhs)
{
    const cv::Point2f lhs_center(
        lhs.x + lhs.width * 0.5f,
        lhs.y + lhs.height * 0.5f);
    const cv::Point2f rhs_center(
        rhs.x + rhs.width * 0.5f,
        rhs.y + rhs.height * 0.5f);
    const float dx = lhs_center.x - rhs_center.x;
    const float dy = lhs_center.y - rhs_center.y;
    return std::sqrt(dx * dx + dy * dy);
}

static float rectDiag(const cv::Rect& rect)
{
    if (rect.width <= 0 || rect.height <= 0) {
        return 0.0f;
    }
    return std::sqrt(static_cast<float>(rect.width * rect.width +
                                        rect.height * rect.height));
}

}  // namespace

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
static inline cv::Rect getSafeRect(const cv::Mat& image, const cv::Rect& rect)
{
    if (image.empty()) return cv::Rect();

    int rx = rect.x, ry = rect.y, rw = rect.width, rh = rect.height;

    // 归一化负宽高
    if (rw < 0) { rx += rw; rw = -rw; }
    if (rh < 0) { ry += rh; rh = -rh; }

    if (rw <= 0 || rh <= 0) return cv::Rect();

    // 完全在外
    const int img_right = image.cols;
    const int img_bottom = image.rows;
    if (rx >= img_right || ry >= img_bottom || rx + rw <= 0 || ry + rh <= 0)
        return cv::Rect();

    // clamp 到图像内
    int x1 = std::max(0, rx);
    int y1 = std::max(0, ry);
    int x2 = std::min(img_right,  rx + rw);
    int y2 = std::min(img_bottom, ry + rh);

    int w = x2 - x1;
    int h = y2 - y1;
    if (w <= 0 || h <= 0) return cv::Rect();

    return cv::Rect(x1, y1, w, h);
}


cv::Rect getSafeRectUAV(const cv::Mat& image, const cv::Rect& rect)
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
    if (intersect_area * 10 <= original_area *3) {
        return cv::Rect();
    }
    
    return cv::Rect(x1, y1, width, height);
}

static float lowPass1st(float input, float& state, bool& inited, float alpha)
{
    if (!inited) {
        state = input;
        inited = true;
        return state;
    }

    state = alpha * input + (1.0f - alpha) * state;
    return state;
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

struct UavDetectionCandidate {
    cv::Rect bbox;
    cv::Rect roi;
    cv::Point2f roi_center;
    float confidence{0.0f};
    int color{1};
};

static cv::Rect build_uav_upper_middle_roi(const cv::Rect& bbox)
{
    if (bbox.width <= 0 || bbox.height <= 0) {
        return cv::Rect();
    }

    const int roi_x = bbox.x + static_cast<int>(std::round(bbox.width * 0.30f));
    const int roi_y = bbox.y;
    const int roi_w = std::max(1, static_cast<int>(std::round(bbox.width * 0.40f)));
    const int roi_h = std::max(1, static_cast<int>(std::round(bbox.height * 0.40f)));
    return cv::Rect(roi_x, roi_y, roi_w, roi_h);
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
 *      - uav_detector: UAV检测（best.onnx, conf=0.5）
 *   5. 初始化UAV跟踪系统：
 *      - uav_tracker_: DeepSORT跟踪器（IOU阈值0.7）
 *      - light_bar_detector_: 灯条精细检测器
 *   6. 创建ROS2订阅者和发布者
 */
Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    // 1. 初始化调试窗口
    debug = this->declare_parameter<int>("debug", 1);  // 默认开启调试绘制
    show_debug_window_ =
        this->declare_parameter<bool>("show_debug_window", true);
    publish_debug_image_ =
        this->declare_parameter<bool>("publish_debug_image", true);
    debug_image_topic_ = this->declare_parameter<std::string>(
        "debug_image_topic", "/detect/debug_image");
    if (show_debug_window_) {
        cv::namedWindow("detect", cv::WINDOW_NORMAL);
    }

    // 2. 检查CUDA环境
    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }

    // 3. 加载配置文件（模型路径）
    cv::FileStorage fs;
    fs.open("/home/user/code/T-DT_Radar-main/config/detect_params.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "[INIT] ❌ Failed to open config file: ./config/detect_params.yaml");
    }
    fs["yolo_path"] >> yolo_path;
    fs["armor_path"] >> armor_path;
    fs["classify_path"] >> classify_path;
    fs["uav_path"] >> uav_path;
    double uav_conf_threshold = 0.5;
    if (!fs["uav_conf_threshold"].empty()) {
        fs["uav_conf_threshold"] >> uav_conf_threshold;
    }
    fs.release();
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Config loaded: yolo=%s, armor=%s, classify=%s, uav=%s, uav_conf=%.2f",
                yolo_path.c_str(), armor_path.c_str(), classify_path.c_str(), uav_path.c_str(),
                uav_conf_threshold);

    // 4. 自动转换ONNX为TensorRT引擎（如不存在）
    std::ifstream file1(yolo_path.c_str());
    // if (!file1.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/self_car.onnx --saveEngine=model/TensorRT/self_car.engine --minBatch 1 --optBatch 1 --maxBatch 2 --Shape=1280x1280 --input_name=images");
    // if (!file1.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/self_car.onnx --saveEngine=model/TensorRT/self_car.engine");
    if (!file1.good())
    system("python3 src/utils/onnx2trt.py "
           "--onnx=model/ONNX/car_merged_200.onnx "
           "--saveEngine=model/TensorRT/car_merged_200_fp16.engine "
           "--minBatch 1 --optBatch 1 --maxBatch 4 "
           "--Shape=1280x1280 --input_name=images");
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/armor_yolo.onnx --saveEngine=model/TensorRT/armor_yolo.engine --minBatch 1 --optBatch 5 --maxBatch 12 --Shape=192x192 --input_name=images");

    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) system("python3 src/utils/onnx2trt.py --onnx=model/ONNX/classify.onnx --saveEngine=model/TensorRT/classify.engine --minBatch 1 --optBatch 10 --maxBatch 20 --Shape=224x224 --input_name=input");

    // 5. 加载深度学习模型
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading classifier model...");
    this->classifier = classify::load(classify_path, classify::Type::densenet121);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Classifier loaded");
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading armor_yolo model...");
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.3f, 0.45f);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Armor YOLO loaded");
    
    RCLCPP_INFO(this->get_logger(), "[INIT] Loading main yolo model...");
    this->yolo = yolo::load(yolo_path, yolo::Type::V26, 0.45f, 0.35f);
    RCLCPP_INFO(this->get_logger(), "[INIT] ✓ Main YOLO loaded");


    RCLCPP_INFO(this->get_logger(), "[CAR-MODEL] yolo_path = %s", yolo_path.c_str());
RCLCPP_INFO(this->get_logger(), "[CAR-MODEL] yolo ptr = %p", (void*)yolo.get());


    // 6. 加载UAV检测器
    const float kUavConfThreshold = static_cast<float>(uav_conf_threshold);
    constexpr float kUavNmsThreshold = 0.25f;
    constexpr int kUavMinBatch = 1;
    constexpr int kUavOptBatch = 4;
    constexpr int kUavMaxBatch = 8;
    const std::string uav_onnx_path = "model/ONNX/uav.onnx";
    if (!fileExists(uav_path)) {
        if (!fileExists(uav_onnx_path)) {
            RCLCPP_ERROR(
                this->get_logger(),
                "[INIT] ❌ UAV engine missing and ONNX not found. engine=%s onnx=%s",
                uav_path.c_str(), uav_onnx_path.c_str());
        } else {
            RCLCPP_WARN(
                this->get_logger(),
                "[INIT] UAV engine missing, rebuilding with FP16 dynamic batch profile %d/%d/%d: %s",
                kUavMinBatch, kUavOptBatch, kUavMaxBatch, uav_path.c_str());
            const int build_ret = buildTensorRTEngine(
                uav_onnx_path,
                uav_path,
                kUavMinBatch,
                kUavOptBatch,
                kUavMaxBatch,
                "1280x1280",
                "images",
                true,
                {"--profilingVerbosity=detailed"});
            if (build_ret != 0 || !fileExists(uav_path)) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "[INIT] ❌ Failed to build UAV engine. return=%d target=%s",
                    build_ret, uav_path.c_str());
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "[INIT] Loading UAV detector from: %s", uav_path.c_str());
    auto tmp_uav = uav::load(uav_path, kUavConfThreshold, kUavNmsThreshold);
    if (tmp_uav != nullptr) {
        this->uav_detector = tmp_uav;
        RCLCPP_INFO(
            this->get_logger(),
            "[INIT] ✓ UAV detector loaded successfully! conf=%.2f, nms=%.2f",
            kUavConfThreshold, kUavNmsThreshold);
    } else {
        RCLCPP_ERROR(this->get_logger(), "[INIT] ❌ FAILED to load UAV detector from: %s", uav_path.c_str());
        RCLCPP_ERROR(this->get_logger(), "[INIT] Check if file exists, has correct format, or is corrupted");
    }

    // UAV 老版本灯条/DeepSORT 逻辑不再接入当前定位链路
    this->light_bar_detector_.reset();
    this->uav_tracker_.reset();
    RCLCPP_INFO(this->get_logger(), "[INIT] UAV pipeline uses raw YOLO boxes only (DeepSORT/light detector disabled)");

    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera_image");

    // 9. 创建ROS2订阅者和发布者
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));

    this->declare_parameter<int>("uav_intersection_timeout_ms", 120);
    uav_intersection_timeout_ms_ =
        this->get_parameter("uav_intersection_timeout_ms").as_int();
    this->declare_parameter<int>("semantic_carry_ms", 500);
    semantic_carry_ms_ =
        std::max(0, static_cast<int>(
                        this->get_parameter("semantic_carry_ms").as_int()));
    uav_intersection_debug_sub_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/uav_intersection_debug", rclcpp::SensorDataQoS(),
            std::bind(&Detect::uavIntersectionDebugCallback, this, std::placeholders::_1));
    uav_cluster_debug_sub_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/uav_cluster_debug", rclcpp::SensorDataQoS(),
            std::bind(&Detect::uavClusterDebugCallback, this, std::placeholders::_1));

    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());
    if (publish_debug_image_) {
        debug_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            debug_image_topic_, rclcpp::QoS(1));
    }

    RCLCPP_INFO(
        this->get_logger(),
        "Detect node has been started. show_debug_window=%d publish_debug_image=%d debug_image_topic=%s",
        show_debug_window_ ? 1 : 0,
        publish_debug_image_ ? 1 : 0,
        debug_image_topic_.c_str());
}

void Detect::uavIntersectionDebugCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
        std::lock_guard<std::mutex> lk(uav_intersection_mutex_);
        uav_intersection_points_.clear();
        uav_intersection_stamp_ =
            (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                ? this->now()
                : rclcpp::Time(msg->header.stamp);
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::vector<cv::Point2f> points;
    points.reserve(cloud.points.size());
    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        points.emplace_back(p.x, p.y);
    }

    std::lock_guard<std::mutex> lk(uav_intersection_mutex_);
    uav_intersection_points_ = std::move(points);
    uav_intersection_stamp_ =
        (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
            ? this->now()
            : rclcpp::Time(msg->header.stamp);
}

void Detect::uavClusterDebugCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
        std::lock_guard<std::mutex> lk(uav_intersection_mutex_);
        uav_cluster_points_.clear();
        uav_cluster_stamp_ =
            (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                ? this->now()
                : rclcpp::Time(msg->header.stamp);
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::vector<cv::Point2f> points;
    points.reserve(cloud.points.size());
    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        points.emplace_back(p.x, p.y);
    }

    std::lock_guard<std::mutex> lk(uav_intersection_mutex_);
    uav_cluster_points_ = std::move(points);
    uav_cluster_stamp_ =
        (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
            ? this->now()
            : rclcpp::Time(msg->header.stamp);
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
void Detect::callback(sensor_msgs::msg::Image::UniquePtr msg)
{
    cv::Mat raw;
    try {
        if (msg->encoding == "bgr8" || msg->encoding == "rgb8") {
            raw = cv::Mat(
                static_cast<int>(msg->height),
                static_cast<int>(msg->width),
                CV_8UC3,
                msg->data.data(),
                static_cast<size_t>(msg->step));
        } else if (msg->encoding == "mono8") {
            raw = cv::Mat(
                static_cast<int>(msg->height),
                static_cast<int>(msg->width),
                CV_8UC1,
                msg->data.data(),
                static_cast<size_t>(msg->step));
        } else {
            auto shared_msg = std::make_shared<sensor_msgs::msg::Image>(*msg);
            auto cv_ptr = cv_bridge::toCvCopy(shared_msg, msg->encoding);
            raw = cv_ptr->image;
        }

        static int dump = 0;
        if (dump++ % 60 == 0 && !raw.empty()) {
            cv::imwrite("/tmp/cam_raw.jpg", raw);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[IMG] cv_bridge exception: %s", e.what());
        return;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // ==================== 阶段1: 图像预处理与异步推理 ====================
    cv::Mat img;
    if (msg->encoding == "bgr8") {
        img = raw;
    } else if (msg->encoding == "rgb8") {
        cv::cvtColor(raw, img, cv::COLOR_RGB2BGR);
    } else if (msg->encoding == "mono8") {
        cv::cvtColor(raw, img, cv::COLOR_GRAY2BGR);
    } else {
        try {
            auto shared_msg = std::make_shared<sensor_msgs::msg::Image>(*msg);
            img = cv_bridge::toCvCopy(shared_msg, "bgr8")->image;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[IMG] unsupported encoding %s: %s",
                         msg->encoding.c_str(), e.what());
            return;
        }
    }
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


    // ==================== DEBUG: print raw YOLO outputs ====================
// RCLCPP_INFO(this->get_logger(), "[CAR] raw result.size() = %zu", result.size());

for (int i = 0; i < (int)std::min<size_t>(result.size(), 20); ++i) {
    const auto& b = result[i];
    // RCLCPP_INFO(this->get_logger(),
    //     "[CAR-RAW] i=%d cls=%d conf=%.4f box=[%.1f %.1f %.1f %.1f] wh=[%.1f %.1f]",
    //     i, (int)b.class_label, (double)b.confidence,
    //     (double)b.left, (double)b.top, (double)b.right, (double)b.bottom,
    //     (double)(b.right - b.left), (double)(b.bottom - b.top));
}

// 统计分布：NaN/Inf、负宽高、低置信度
int nan_cnt = 0, badwh_cnt = 0, low_cnt = 0;
float min_conf = 1e9f, max_conf = -1e9f;
for (const auto& b : result) {
    if (!std::isfinite(b.confidence) || !std::isfinite(b.left) || !std::isfinite(b.top) ||
        !std::isfinite(b.right) || !std::isfinite(b.bottom)) {
        nan_cnt++;
        continue;
    }
    float w = b.right - b.left;
    float h = b.bottom - b.top;
    if (w <= 0 || h <= 0) badwh_cnt++;
    min_conf = std::min(min_conf, b.confidence);
    max_conf = std::max(max_conf, b.confidence);
    if (b.confidence < 0.01f) low_cnt++;
}
// RCLCPP_INFO(this->get_logger(),
//     "[CAR-STATS] nan=%d bad_wh=%d low_conf(<0.01)=%d conf_range=[%.4f, %.4f]",
//     nan_cnt, badwh_cnt, low_cnt, (double)min_conf, (double)max_conf);


    // ==================== 阶段2: 初始化检测结果消息 ====================
    vision_interface::msg::DetectResult detect_result;
    detect_result.header = msg->header;
    
    // 2.1 初始化车辆检测数组（防止垃圾数据）
    for (int i = 0; i < 7; i++) {
        detect_result.blue_x[i] = 0.0f;
        detect_result.blue_y[i] = 0.0f;
        detect_result.red_x[i] = 0.0f;
        detect_result.red_y[i] = 0.0f;
        detect_result.blue_car_box_x1[i] = 0.0f;
        detect_result.blue_car_box_y1[i] = 0.0f;
        detect_result.blue_car_box_x2[i] = 0.0f;
        detect_result.blue_car_box_y2[i] = 0.0f;
        detect_result.red_car_box_x1[i] = 0.0f;
        detect_result.red_car_box_y1[i] = 0.0f;
        detect_result.red_car_box_x2[i] = 0.0f;
        detect_result.red_car_box_y2[i] = 0.0f;
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

    detect_result.blue_uav_bbox_x = 0.0f;
    detect_result.blue_uav_bbox_y = 0.0f;
    detect_result.blue_uav_bbox_w = 0.0f;
    detect_result.blue_uav_bbox_h = 0.0f;
    detect_result.blue_uav_conf = 0.0f;
    detect_result.red_uav_bbox_x = 0.0f;
    detect_result.red_uav_bbox_y = 0.0f;
    detect_result.red_uav_bbox_w = 0.0f;
    detect_result.red_uav_bbox_h = 0.0f;
    detect_result.red_uav_conf = 0.0f;
    
    size_t array_limit = detect_result.blue_x.size(); 
    rclcpp::Time frame_time = msg->header.stamp;
    if (frame_time.nanoseconds() == 0) {
        frame_time = this->now();
    }

    auto write_car_detection = [&](int color,
                                   int idx,
                                   const cv::Point2f& center,
                                   const cv::Rect& car_rect) -> bool {
        if (idx < 0 || idx >= static_cast<int>(array_limit)) {
            return false;
        }
        if (color == 0) {
            detect_result.blue_x[idx] = center.x;
            detect_result.blue_y[idx] = center.y;
            detect_result.blue_car_box_x1[idx] =
                static_cast<float>(car_rect.x);
            detect_result.blue_car_box_y1[idx] =
                static_cast<float>(car_rect.y);
            detect_result.blue_car_box_x2[idx] =
                static_cast<float>(car_rect.x + car_rect.width);
            detect_result.blue_car_box_y2[idx] =
                static_cast<float>(car_rect.y + car_rect.height);
            return true;
        }
        if (color == 2) {
            detect_result.red_x[idx] = center.x;
            detect_result.red_y[idx] = center.y;
            detect_result.red_car_box_x1[idx] =
                static_cast<float>(car_rect.x);
            detect_result.red_car_box_y1[idx] =
                static_cast<float>(car_rect.y);
            detect_result.red_car_box_x2[idx] =
                static_cast<float>(car_rect.x + car_rect.width);
            detect_result.red_car_box_y2[idx] =
                static_cast<float>(car_rect.y + car_rect.height);
            return true;
        }
        return false;
    };

    auto slot_has_detection = [&](int color, int idx) -> bool {
        if (idx < 0 || idx >= static_cast<int>(array_limit)) {
            return true;
        }
        if (color == 0) {
            return detect_result.blue_x[idx] != 0.0f ||
                   detect_result.blue_y[idx] != 0.0f;
        }
        if (color == 2) {
            return detect_result.red_x[idx] != 0.0f ||
                   detect_result.red_y[idx] != 0.0f;
        }
        return true;
    };

    // ==================== 阶段3: 处理地面车辆检测（ID 1-6）====================
    std::vector<Car> cars;
    
    if (!result.empty()) {
        // 3.0 先过滤再截断（解决 “Cars=300” 与 “超过MAX_CARS就不处理”）
    // 主车YOLO在 init 里 load 的阈值是 0.65f，这里保持一致做二次保险过滤
    constexpr float CAR_CONF_THRES = 0.65f;

    std::vector<yolo::Box> car_filtered;
    car_filtered.reserve(result.size());

    for (const auto& b : result) {
        // 丢掉 NaN/Inf
        if (!std::isfinite(b.confidence) ||
            !std::isfinite(b.left) || !std::isfinite(b.top) ||
            !std::isfinite(b.right) || !std::isfinite(b.bottom)) {
            continue;
        }

        // 只保留车类别（你原来就是 0/1）
        if (!(b.class_label == 0 || b.class_label == 1)) continue;

        // 置信度阈值过滤（避免 TopK 全进来）
        if (b.confidence < CAR_CONF_THRES) continue;

        // 合法框（宽高必须正）
        if ((b.right - b.left) <= 2.0f || (b.bottom - b.top) <= 2.0f) continue;

        car_filtered.push_back(b);
    }

    // 置信度从高到低排序，然后只取 MAX_CARS 个
    std::sort(car_filtered.begin(), car_filtered.end(),
              [](const yolo::Box& a, const yolo::Box& b) { return a.confidence > b.confidence; });

    if ((int)car_filtered.size() > MAX_CARS) car_filtered.resize(MAX_CARS);

    // 3.1 过滤车辆并提取ROI（单循环完成）
    std::vector<cv::Mat> car_imgs;
    std::vector<cv::Rect> car_rects;

    for (const auto& box : car_filtered) {

        // RCLCPP_INFO(this->get_logger(),
        //     "Detected CAR: ID=%d, Conf=%.3f, Rect=[%.1f,%.1f,%.1f,%.1f]",
        //     box.class_label, box.confidence, box.left, box.top, box.right, box.bottom);

        int l = (int)std::round(box.left);
        int t = (int)std::round(box.top);
        int r = (int)std::round(box.right);
        int b = (int)std::round(box.bottom);

        cv::Rect temp_rect(l, t, r - l, b - t);
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);

        // 过滤过小的检测框
        if (temp_car_rect.area() < 10 * 10) continue;

        // 【关键点：提前绘制】
        if (debug) {
            cv::rectangle(img, temp_car_rect, cv::Scalar(0, 255, 255), 2);
            cv::putText(img, "Car-Detected",
                        cv::Point(temp_car_rect.x, temp_car_rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.75,
                        cv::Scalar(0, 255, 255), 1);
        }

        car_imgs.push_back(img(temp_car_rect).clone());
        car_rects.push_back(temp_car_rect);

        Car car;
        car.car = box;
        car.car_rect = temp_car_rect;
        cars.push_back(car);
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
            
            auto try_carry_semantic = [&](const Car& car,
                                          int& out_color,
                                          int& out_number,
                                          int& out_idx) -> bool {
                if (semantic_carry_ms_ <= 0 || car.car_rect.area() <= 0) {
                    return false;
                }

                float best_match_score = -1.0f;
                for (int color : {0, 2}) {
                    for (int idx = 0; idx < 6 && idx < static_cast<int>(array_limit); ++idx) {
                        const auto& cache = last_semantic_tracks_[color][idx];
                        if (!cache.valid || cache.car_rect.area() <= 0 ||
                            slot_has_detection(color, idx)) {
                            continue;
                        }

                        const double dt_ms =
                            (frame_time - cache.last_real_armor_time).seconds() * 1000.0;
                        if (dt_ms < 0.0 ||
                            dt_ms > static_cast<double>(semantic_carry_ms_)) {
                            continue;
                        }

                        const float area_ratio =
                            static_cast<float>(car.car_rect.area()) /
                            static_cast<float>(std::max(1, cache.car_rect.area()));
                        if (area_ratio < 0.5f || area_ratio > 2.0f) {
                            continue;
                        }

                        const float iou = rectIou(car.car_rect, cache.car_rect);
                        const float center_distance =
                            rectCenterDistance(car.car_rect, cache.car_rect);
                        const float diag =
                            std::max(rectDiag(car.car_rect), rectDiag(cache.car_rect));
                        const bool center_ok =
                            diag > 1.0f && center_distance <= 0.75f * diag;
                        if (iou <= 0.25f && !center_ok) {
                            continue;
                        }

                        const float center_score =
                            center_ok ? std::max(0.0f,
                                                 1.0f - center_distance /
                                                            std::max(1.0f, 0.75f * diag))
                                      : 0.0f;
                        const float score = iou + center_score;
                        if (score > best_match_score) {
                            best_match_score = score;
                            out_color = cache.color;
                            out_number = cache.number;
                            out_idx = cache.index;
                        }
                    }
                }

                return best_match_score >= 0.0f;
            };

            // 3.7 数字匹配与结果发送；无装甲板时短时继承最近语义
            for (auto& car : cars) {
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

                if (best_armor_idx == -1) {
                    int carried_color = 1;
                    int carried_number = 0;
                    int carried_idx = -1;
                    if (!try_carry_semantic(car, carried_color, carried_number, carried_idx)) {
                        continue;
                    }

                    car.color = carried_color;
                    car.number = carried_number;
                    car.center = cv::Point2f(
                        car.car_rect.x + car.car_rect.width * 0.5f,
                        car.car_rect.y + car.car_rect.height * 0.5f);
                    write_car_detection(car.color, carried_idx, car.center, car.car_rect);

                    if (debug) {
                        cv::Scalar draw_color =
                            (car.color == 2) ? cv::Scalar(0, 80, 255)
                                             : cv::Scalar(255, 120, 0);
                        std::string label =
                            (car.color == 2 ? "R" : "B") +
                            std::to_string(car.number) + "-carry";
                        cv::rectangle(img, car.car_rect, draw_color, 2);
                        cv::putText(img, label,
                                    cv::Point(car.car_rect.x, car.car_rect.y),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.85, draw_color, 2);
                    }
                    continue;
                }

                // 计算车辆中心点和颜色
                const auto& best_armor = car.armors[best_armor_idx];
                cv::Rect best_rect(
                    best_armor.left + car.car.left,
                    best_armor.top + car.car.top,
                    best_armor.right - best_armor.left,
                    best_armor.bottom - best_armor.top
                );

                // 车号仍然沿用“最佳装甲板”的分类结果
                car.number = best_armor.class_label;

                // 颜色仍然沿用“最佳装甲板ROI”判断
                auto safe_rect = getSafeRect(img, best_rect);
                cv::Mat armor_roi = img(safe_rect).clone();
                car.color = getColor(armor_roi);

                // ------------------------------
                // 新的最终定位定义：
                // x = 车框中心
                // y = 该车所有装甲板中心y的平均值
                // 前提：这辆车存在 best_armor_idx（即已有有效最佳装甲板）
                // ------------------------------

                float final_center_x = car.car_rect.x + car.car_rect.width / 2.0f;

                float sum_center_y = 0.0f;
                int valid_armor_count = 0;
                for (const auto& armor : car.armors) {
                    if (armor.class_label == 0 || armor.class_label == 5) {
                        continue;
                    }

                    float armor_center_y =
                        (armor.top + car.car.top) +
                        (armor.bottom - armor.top) / 2.0f;

                    sum_center_y += armor_center_y;
                    valid_armor_count++;
                }

                float final_center_y = best_rect.y + best_rect.height / 2.0f;
                if (valid_armor_count > 0) {
                    final_center_y = sum_center_y / static_cast<float>(valid_armor_count);
                }

                float filtered_center_y = final_center_y;
                int idx = car.number - 1;
                if (idx >= 0 && idx < 7) {
                    if (car.color == 0) {
                        filtered_center_y = lowPass1st(
                            final_center_y,
                            blue_y_lpf_[idx],
                            blue_y_lpf_init_[idx],
                            y_lpf_alpha_);
                    } else if (car.color == 2) {
                        filtered_center_y = lowPass1st(
                            final_center_y,
                            red_y_lpf_[idx],
                            red_y_lpf_init_[idx],
                            y_lpf_alpha_);
                    }
                }

                car.center = cv::Point2f(final_center_x, filtered_center_y);

                if (write_car_detection(car.color, idx, car.center, car.car_rect) &&
                    idx >= 0 && idx < 6 && (car.color == 0 || car.color == 2)) {
                    auto& cache = last_semantic_tracks_[car.color][idx];
                    cache.color = car.color;
                    cache.number = car.number;
                    cache.index = idx;
                    cache.car_rect = car.car_rect;
                    cache.center = car.center;
                    cache.last_real_armor_time = frame_time;
                    cache.valid = true;
                }

                // 3.8 绘制调试信息
                if (debug) {
                    cv::Scalar draw_color = (car.color == 2) ? cv::Scalar(0,0,255) : cv::Scalar(255,0,0);
                    std::string label = (car.color == 2 ? "R" : "B") + std::to_string(car.number);
                    cv::rectangle(img, car.car_rect, draw_color, 3);
                    cv::putText(img, label, cv::Point(car.car_rect.x, car.car_rect.y),
                                   cv::FONT_HERSHEY_SIMPLEX, 1.15, draw_color, 2);
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
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                     "[UAV] Raw detections: %zu", uav_raw_result.size());
    
    std::sort(uav_raw_result.begin(), uav_raw_result.end(), 
              [](const yolo::Box& a, const yolo::Box& b) { return a.confidence > b.confidence; });
    
    std::vector<yolo::Box> uav_filtered;
    for (size_t i = 0; i < uav_raw_result.size() && i < 2; i++) {
        uav_filtered.push_back(uav_raw_result[i]);
        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        //                     "[UAV] Filtered #%zu: conf=%.3f, box=[%.1f,%.1f,%.1f,%.1f]",
        //                     i, uav_raw_result[i].confidence,
        //                     uav_raw_result[i].left, uav_raw_result[i].top,
        //                     uav_raw_result[i].right, uav_raw_result[i].bottom);
    }
    
    std::vector<UavDetectionCandidate> uav_candidates;
    uav_candidates.reserve(uav_filtered.size());

    auto area_ratio_ok = [&](const cv::Rect& r) -> bool {
        if (r.width <= 0 || r.height <= 0) return false;
        if (r.width < r.height || r.width > 4 * r.height) return false;
        const float ratio = static_cast<float>(r.area()) / img_area;
        return ratio >= 0.0002f && ratio <= 0.08f;
    };

    for (const auto& uav : uav_filtered) {
        int x1 = static_cast<int>(std::round((uav.left - pad_x) * uav_scale_inv));
        int y1 = static_cast<int>(std::round((uav.top - pad_y) * uav_scale_inv));
        int x2 = static_cast<int>(std::round((uav.right - pad_x) * uav_scale_inv));
        int y2 = static_cast<int>(std::round((uav.bottom - pad_y) * uav_scale_inv));

        x1 = std::max(0, std::min(x1, img.cols - 1));
        y1 = std::max(0, std::min(y1, img.rows - 1));
        x2 = std::max(0, std::min(x2, img.cols - 1));
        y2 = std::max(0, std::min(y2, img.rows - 1));

        cv::Rect bbox(x1, y1, x2 - x1, y2 - y1);
        bbox = getSafeRectUAV(img, bbox);
        if (!area_ratio_ok(bbox)) {
            continue;
        }

        cv::Mat bbox_roi = img(bbox);
        const int uav_color = getColor(bbox_roi);

        cv::Rect roi = build_uav_upper_middle_roi(bbox);
        roi = getSafeRect(img, roi);
        if (roi.width <= 0 || roi.height <= 0) {
            roi = bbox;
        }

        UavDetectionCandidate candidate;
        candidate.bbox = bbox;
        candidate.roi = roi;
        candidate.roi_center = cv::Point2f(
            roi.x + roi.width * 0.5f,
            roi.y + roi.height * 0.5f);
        candidate.confidence = uav.confidence;
        candidate.color = uav_color;
        uav_candidates.push_back(candidate);
    }

    int uav_id = UAV_ID;
    int idx = uav_id - 1;
    bool can_send = (idx < static_cast<int>(array_limit));
    if (!can_send) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "DetectResult array size %zu too small for UAV index %d; consider expanding msg arrays",
                             array_limit, idx);
    }

    auto choose_best_candidate = [&](int color) -> const UavDetectionCandidate* {
        const UavDetectionCandidate* best = nullptr;
        for (const auto& candidate : uav_candidates) {
            if (candidate.color != color) {
                continue;
            }
            if (best == nullptr || candidate.confidence > best->confidence) {
                best = &candidate;
            }
        }
        return best;
    };

    const UavDetectionCandidate* best_blue = choose_best_candidate(0);
    const UavDetectionCandidate* best_red = choose_best_candidate(2);

    if (can_send && best_blue != nullptr) {
        detect_result.blue_x[idx] = best_blue->roi_center.x;
        detect_result.blue_y[idx] = best_blue->roi_center.y;
        detect_result.blue_uav_bbox_x = static_cast<float>(best_blue->bbox.x);
        detect_result.blue_uav_bbox_y = static_cast<float>(best_blue->bbox.y);
        detect_result.blue_uav_bbox_w = static_cast<float>(best_blue->bbox.width);
        detect_result.blue_uav_bbox_h = static_cast<float>(best_blue->bbox.height);
        detect_result.blue_uav_conf = best_blue->confidence;
    }

    if (can_send && best_red != nullptr) {
        detect_result.red_x[idx] = best_red->roi_center.x;
        detect_result.red_y[idx] = best_red->roi_center.y;
        detect_result.red_uav_bbox_x = static_cast<float>(best_red->bbox.x);
        detect_result.red_uav_bbox_y = static_cast<float>(best_red->bbox.y);
        detect_result.red_uav_bbox_w = static_cast<float>(best_red->bbox.width);
        detect_result.red_uav_bbox_h = static_cast<float>(best_red->bbox.height);
        detect_result.red_uav_conf = best_red->confidence;
    }

    if (debug) {
        for (const auto& candidate : uav_candidates) {
            cv::Scalar color_bgr = cv::Scalar(255, 0, 255);
            std::string label = "UAV";
            if (candidate.color == 0) {
                color_bgr = cv::Scalar(255, 0, 0);
                label = "UAV-B";
            } else if (candidate.color == 2) {
                color_bgr = cv::Scalar(0, 0, 255);
                label = "UAV-R";
            }

            cv::rectangle(img, candidate.bbox, color_bgr, 2);
            cv::rectangle(img, candidate.roi, cv::Scalar(0, 255, 255), 2);
            cv::circle(img, candidate.roi_center, 4, cv::Scalar(0, 255, 0), -1);
            cv::putText(img, label + cv::format(" %.2f", candidate.confidence),
                        cv::Point(candidate.bbox.x, std::max(candidate.bbox.y - 6, 0)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.95, color_bgr, 2);
        }

        const rclcpp::Time image_stamp =
            (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                ? this->now()
                : rclcpp::Time(msg->header.stamp);

        std::vector<cv::Point2f> debug_points;
        std::vector<cv::Point2f> debug_cluster_points;
        {
            std::lock_guard<std::mutex> lk(uav_intersection_mutex_);
            if (uav_intersection_stamp_.nanoseconds() != 0) {
                const double dt_ms =
                    std::fabs((image_stamp - uav_intersection_stamp_).seconds()) * 1000.0;
                if (dt_ms <= static_cast<double>(uav_intersection_timeout_ms_)) {
                    debug_points = uav_intersection_points_;
                }
            }
            if (uav_cluster_stamp_.nanoseconds() != 0) {
                const double dt_ms =
                    std::fabs((image_stamp - uav_cluster_stamp_).seconds()) * 1000.0;
                if (dt_ms <= static_cast<double>(uav_intersection_timeout_ms_)) {
                    debug_cluster_points = uav_cluster_points_;
                }
            }
        }

        for (const auto& pt : debug_points) {
            cv::circle(img, pt, 1, cv::Scalar(0, 255, 255), -1);
        }
        for (const auto& pt : debug_cluster_points) {
            cv::circle(img, pt, 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    // ==================== 阶段5: 发布结果与性能统计 ====================

    // RCLCPP_INFO(this->get_logger(), "[DETECT-RESULT] header.stamp: sec=%u, nsec=%u", detect_result.header.stamp.sec, detect_result.header.stamp.nanosec);
    // 输出UAV相关坐标日志，便于调试
    // RCLCPP_INFO(this->get_logger(), "[DETECT-RESULT] blue_x[%d]=%.2f, blue_y[%d]=%.2f, red_x[%d]=%.2f, red_y[%d]=%.2f", 
    //     idx, detect_result.blue_x[idx], idx, detect_result.blue_y[idx], idx, detect_result.red_x[idx], idx, detect_result.red_y[idx]);

    // 5.1 设置header.stamp为当前时间并发布检测结果到话题
    //detect_result.header.stamp = this->now();
    pub->publish(detect_result);

    // 5.2 计算检测耗时
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms" << std::endl;

    // 5.3 可视化调试输出
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1080, 960));
    
    // 显示检测统计信息（显示过滤后的UAV数量）
    std::string status_text = "Cars: " + std::to_string(cars.size()) +
                          " | UAVs: " + std::to_string(uav_candidates.size());

    cv::putText(final_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    
    if (uav_candidates.empty()) {
        cv::putText(final_img, "NO UAV DETECTED!", cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    if (debug_img_pub_) {
        auto debug_msg =
            cv_bridge::CvImage(msg->header, "bgr8", final_img).toImageMsg();
        debug_img_pub_->publish(*debug_msg);
    }

    // 5.4 显示调试窗口
    if (show_debug_window_) {
        cv::imshow("detect", final_img);

        // 5.5 处理键盘输入（r键切换调试模式）
        auto key = cv::waitKey(1);
        if (key == 'r') {
            debug = !debug;
        }
    }
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)
