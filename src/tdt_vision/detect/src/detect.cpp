// 包含自定义检测模块头文件
#include "detect.h"

// 引入文件系统操作（检查文件是否存在）
#include <filesystem>
// OpenCV图像处理模块（如矩形操作、图像缩放）
#include <opencv2/imgproc.hpp>
// ROS 2日志工具
#include <rclcpp/logging.hpp>
// ROS 2节点关闭工具
#include <rclcpp/utilities.hpp>

// 自定义信息输出宏（替代std::cout，统一输出格式）
#define TDT_INFO(msg) std::cout << msg << std::endl
// 最大车辆检测数量限制（避免检测结果过多导致后续处理压力）
#define MAX_CARS 12
// 最大装甲板检测数量限制（同理，控制处理规模）
#define MAX_ARMORS 20

// 雷达相关功能的命名空间（避免命名冲突）
namespace tdt_radar {
// 全局变量：用于装甲板/车辆图像保存的计数（调试/训练数据收集用）
unsigned int count_img = 0;

/**
 * @brief 基于RGB通道差异判断图像颜色（蓝色或红色）
 * @param img 输入图像（ROI区域，通常为装甲板区域）
 * @return 0-蓝色，2-红色（与业务中颜色定义对应）
 */
int getColor(cv::Mat& img)
{
    // 拆分RGB通道（OpenCV默认BGR顺序：channels[0]=B, 1=G, 2=R）
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // 计算蓝通道-红通道、红通道-蓝通道的差值矩阵（突出颜色差异）
    cv::Mat blueMinusRed =
        channels[0] - channels[2];  // B-R：蓝色区域差值为正
    cv::Mat redMinusBlue =
        channels[2] - channels[0];  // R-B：红色区域差值为正

    // 计算差值矩阵的均值（整体颜色倾向）
    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);

    // 比较两个均值：B-R均值更大则为蓝色（返回0），否则为红色（返回2）
    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0;
    } else {
        return 2;
    }
}

/**
 * @brief 判断小矩形是否完全在大矩形内部（用于过滤嵌套目标）
 * @param small 待判断的小矩形
 * @param big 作为边界的大矩形
 * @return true-小矩形在大矩形内，false-否则
 */
bool isRectInside(const cv::Rect& small, const cv::Rect& big)
{
    // 检查小矩形的四个顶点是否都在大矩形内
    bool topLeftInside = big.contains(small.tl());  // 左上角
    bool topRightInside =
        big.contains(cv::Point(small.x + small.width, small.y));  // 右上角
    bool bottomLeftInside =
        big.contains(cv::Point(small.x, small.y + small.height));  // 左下角
    bool bottomRightInside = big.contains(cv::Point(
        small.x + small.width, small.y + small.height));  // 右下角

    // 四个顶点均在内部才返回true
    return (topLeftInside && topRightInside && bottomLeftInside &&
            bottomRightInside);
}

/**
 * @brief 封装yolo::Box类型，判断小Box是否在大Box内部（调用isRectInside）
 * @param small 待判断的小Box（yolo检测结果格式）
 * @param big 作为边界的大Box
 * @return true-小Box在大Box内，false-否则
 */
bool isBoxInside(const yolo::Box& small, const yolo::Box& big)
{
    // 将yolo::Box转换为cv::Rect（left/top为左上角，width=right-left，height=bottom-top）
    cv::Rect small_rect(small.left, small.top, small.right - small.left,
                        small.bottom - small.top);
    cv::Rect big_rect(big.left, big.top, big.right - big.left,
                      big.bottom - big.top);

    // 复用矩形判断逻辑
    return isRectInside(small_rect, big_rect);
}

/**
 * @brief 确保矩形完全在图像范围内（避免越界访问内存）
 * @param image 参考图像（用于获取边界）
 * @param rect 输入矩形（可能越界）
 * @return 调整后的安全矩形（完全在图像内）
 */
cv::Rect getSafeRect(cv::Mat& image, cv::Rect& rect)
{
    cv::Rect save_rect;
    // 左上角x坐标：最小为0（左边界），最大为输入rect.x（避免左越界）
    save_rect.x = std::max(0, rect.x);
    // 左上角y坐标：最小为0（上边界），最大为输入rect.y（避免上越界）
    save_rect.y = std::max(0, rect.y);
    // 宽度：最大为“图像右边界 -
    // 安全x坐标”（避免右越界），最小为输入rect.width
    save_rect.width = std::min(image.cols - save_rect.x, rect.width);
    // 高度：最大为“图像下边界 -
    // 安全y坐标”（避免下越界），最小为输入rect.height
    save_rect.height = std::min(image.rows - save_rect.y, rect.height);
    return save_rect;
}

/**
 * @brief Detect类构造函数（初始化节点、检测器、ROS通信）
 * @param node_options ROS 2节点选项（如参数配置）
 */
Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node",
           node_options)  // 初始化ROS 2节点，节点名"radar_detect_node"
{
    // 创建OpenCV显示窗口（"detect"窗口，支持自由缩放）
    cv::namedWindow("detect", cv::WINDOW_NORMAL);

    // 检查CUDA是否可用（TensorRT检测器依赖GPU）
    std::cout << "Checking CUDA with nvidia-smi...\n";
    if (system("nvidia-smi") ==
        0) {  // 执行nvidia-smi命令，返回0表示成功（有CUDA）
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {  // 无CUDA则无法运行TensorRT，关闭节点
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }

    // 读取检测配置文件（detect_params.yaml）
    cv::FileStorage fs;
    fs.open("./config/detect_params.yaml", cv::FileStorage::READ);
    fs["yolo_path"] >> yolo_path;    // 车辆检测TensorRT引擎路径
    fs["armor_path"] >> armor_path;  // 装甲板检测TensorRT引擎路径
    fs["classify_path"] >> classify_path;  // 装甲板分类TensorRT引擎路径
    fs.release();                          // 释放文件资源

    // 检查车辆检测TRT引擎是否存在，不存在则从ONNX转换
    std::ifstream file1(yolo_path.c_str());
    if (!file1.good()) {  // 文件不存在
        // 调用Python脚本：将ONNX模型转为TRT引擎（指定批次、输入尺寸、输入名）
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/RM2025.onnx "  // 输入ONNX模型
               "--saveEngine=model/TensorRT/yolo.engine "  // 输出TRT引擎
               "--minBatch 1 "                             // 最小批次
               "--optBatch 1 "                             // 最优批次
               "--maxBatch 2 "                             // 最大批次
               "--Shape=1280x1280 "     // 输入图像尺寸
               "--input_name=images");  // 模型输入名
    } else {                            // 文件存在，直接加载
        TDT_INFO("Load yolo engine!");
    }

    // 检查装甲板检测TRT引擎是否存在，不存在则转换
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/armor_yolo.onnx "
               "--saveEngine=model/TensorRT/armor_yolo.engine "
               "--minBatch 1 "
               "--optBatch 5 "
               "--maxBatch 12 "
               "--Shape=192x192 "  // 装甲板检测输入尺寸
               "--input_name=images");
    } else {
        TDT_INFO("Load armor_yolo engine!");
    }

    // 检查装甲板分类TRT引擎是否存在，不存在则转换
    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/classify.onnx "
               "--saveEngine=model/TensorRT/classify.engine "
               "--minBatch 1 "
               "--optBatch 10 "
               "--maxBatch 20 "
               "--Shape=224x224 "      // 分类模型输入尺寸
               "--input_name=input");  // 分类模型输入名
    } else {
        TDT_INFO("Load classify engine!");
    }

    // 打印加载的模型路径（调试用）
    std::cout << "yolo_path:" << yolo_path << "\n";
    std::cout << "armor_path:" << armor_path << "\n";
    std::cout << "classify_path:" << classify_path << "\n";

    // 初始化装甲板分类器（densenet121模型）
    this->classifier =
        classify::load(classify_path, classify::Type::densenet121);
    TDT_INFO("Load classify engine success!");

    // 初始化装甲板检测器（YOLOv8，置信度阈值0.4，IOU阈值0.45）
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.45f);
    TDT_INFO("Load armor_yolo engine success!");

    // 初始化车辆检测器（YOLOv8，置信度阈值0.65，IOU阈值0.45）
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);
    TDT_INFO("Load yolo engine success!");

    // 创建ROS
    // 2图像订阅者：订阅"camera_image"话题（传感器数据QoS，确保实时性）
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this,
                  std::placeholders::_1));  // 绑定图像处理回调

    // 创建ROS 2检测结果发布者：发布"detect_result"话题（自定义消息类型）
    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());

    // 打印节点启动成功日志
    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

/**
 * @brief 图像回调函数（核心检测逻辑：车辆→装甲板→分类→颜色判断→结果发布）
 * @param msg 订阅到的ROS图像消息（sensor_msgs::msg::Image）
 */
void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    // 记录检测开始时间（用于计算总耗时）
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "Detecting..." << std::endl;

    // 将ROS图像消息转换为OpenCV格式（BGR8编码，共享内存避免拷贝）
    auto img = cv_bridge::toCvShare(msg, "bgr8")->image;
    // 转换为自定义Image类型（适配检测器输入）
    tdt_radar::Image image(img.data, img.cols, img.rows);

    // 1. 第一步：车辆检测（用yolo检测器检测整图中的车辆）
    auto result = yolo->forward(image);
    // 无车辆检测结果→打印日志并返回
    if (result.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "No Car!");
        return;
    }
    // 检测车辆数量超过最大值→打印日志并返回（避免处理压力）
    else if (result.size() > MAX_CARS) {
        RCLCPP_INFO(this->get_logger(), "Too Many Car!");
        return;
    }

    // 存储车辆ROI图像（用于后续装甲板检测）
    std::vector<tdt_radar::Image> images;
    std::vector<cv::Mat>          car_imgs;
    // 存储车辆信息（检测框、ROI、装甲板等）
    std::vector<Car> cars;

    // 筛选车辆检测结果（仅保留class_label=0或1的目标，视为有效车辆）
    for (auto& box : result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;  // 保存车辆检测框
            cars.push_back(car);
        }
    }

    // 提取每个车辆的安全ROI（避免越界），并保存车辆图像
    for (auto& car : cars) {
        // 从车辆检测框构造cv::Rect
        auto temp_rect =
            cv::Rect(car.car.left, car.car.top,
                     car.car.right - car.car.left,  // 宽度=右边界-左边界
                     car.car.bottom - car.car.top  // 高度=下边界-上边界
            );
        // 确保ROI在图像内
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
        // 提取车辆ROI图像
        auto car_img = img(temp_car_rect);
        car_imgs.push_back(
            car_img.clone());  // 克隆保存（避免原图像修改影响）
        car.car_rect = temp_car_rect;  // 保存车辆安全ROI
    }

    // 将车辆ROI转换为检测器输入格式（tdt_radar::Image）
    for (auto& car_img : car_imgs) {
        auto image =
            tdt_radar::Image(car_img.data, car_img.cols, car_img.rows);
        images.push_back(image);
    }

    // 2. 第二步：装甲板检测（批量检测所有车辆ROI中的装甲板）
    auto armor_boxes = armor_yolo->forwards(images);
    bool has_armor = false;  // 标记是否检测到装甲板
    // 将装甲板检测结果对应到每辆车
    for (int i = 0; i < armor_boxes.size(); i++) {
        if (armor_boxes[i].size() == 0) {  // 该车辆无装甲板
            continue;
        } else {  // 该车辆有装甲板→保存并标记
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }
    // 无任何装甲板→打印日志并返回
    if (!has_armor) {
        RCLCPP_INFO(this->get_logger(), "No Armor!");
        return;
    }

    // ------------------------------
    // （注释代码）调试用：保存装甲板图像到本地（用于标注或训练数据）
    // for(int i=0;i<cars.size();i++){
    //   if(cars[i].armors.size()==0){continue;}
    //   for(auto &armor:cars[i].armors){
    //     cv::Rect rect_img(
    //       armor.left+cars[i].car.left,  //
    //       装甲板在整图中的左坐标（车辆ROI偏移+装甲板在ROI内坐标）
    //       armor.top+cars[i].car.top,
    //       armor.right-armor.left,
    //       armor.bottom-armor.top);
    //     cv::imwrite("/home/tdt/Label/armor_detect/"+std::to_string(count_img++)+".jpg",img(rect_img));
    //     }
    // }
    // ------------------------------

    // 存储装甲板ROI图像（用于后续分类）
    std::vector<cv::Mat>         armor_imgs;
    std::vector<classify::Image> armor_images;

    // 提取每个装甲板的安全ROI（整图坐标系）
    for (auto& car : cars) {
        if (car.armors.size() == 0) {  // 无装甲板的车辆跳过
            continue;
        }
        for (auto& armor : car.armors) {
            // 计算装甲板在整图中的坐标（车辆ROI左上角 +
            // 装甲板在ROI内的坐标）
            cv::Rect rect_img_1(
                armor.left + car.car.left, armor.top + car.car.top,
                armor.right - armor.left, armor.bottom - armor.top);
            // 确保装甲板ROI在图像内
            cv::Rect rect_img = getSafeRect(img, rect_img_1);
            // 提取装甲板ROI图像
            auto armor_img = img(rect_img);
            armor_imgs.push_back(armor_img.clone());
        }
    }

    // 将装甲板ROI转换为分类器输入格式（classify::Image）
    for (auto& armor_img : armor_imgs) {
        auto image =
            classify::Image(armor_img.data, armor_img.cols, armor_img.rows);
        armor_images.push_back(image);
    }

    // 3. 第三步：装甲板分类（批量分类装甲板，获取类别编号）
    auto armor_result = classifier->forwards(armor_images);

    // ------------------------------
    // （注释代码）调试用：按分类结果保存装甲板图像（用于训练数据分类）
    // for(int i=0;i<armor_imgs.size();i++){
    //   cv::imwrite("/home/mozihe/Label/fenqu_armor/"+std::to_string(armor_result[i])+"/"+std::to_string(count_img++)+".jpg",armor_imgs[i]);
    // }
    // ------------------------------

    // 将分类结果分配给对应的装甲板（按检测顺序一一对应）
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            armor.class_label =
                armor_result[0];  // 取第一个分类结果（顺序对应）
            armor_result.erase(
                armor_result.begin());  // 删除已分配的结果，避免重复
            // debug模式：在图像上绘制装甲板类别编号
            if (debug) {
                cv::putText(
                    img, std::to_string(armor.class_label),
                    cv::Point(armor.left + car.car.left,
                              armor.top +
                                  car.car.top),  // 文本位置（装甲板左上角）
                    cv::FONT_HERSHEY_SIMPLEX, 1,  // 字体、字号
                    cv::Scalar(255, 255, 255), 2  // 白色文本，线宽2
                );
            }
        }
    }

    // 4. 第四步：构造检测结果消息（vision_interface::msg::DetectResult）
    vision_interface::msg::DetectResult detect_result;
    for (auto& car : cars) {
        if (car.armors.size() == 0) {  // 无装甲板的车辆跳过
            continue;
        }

        cv::Rect max_rect;            // 置信度最高的装甲板矩形
        float    max_confidence = 0;  // 最高置信度
        // 筛选当前车辆中“有效装甲板”（class_label≠0且≠5，视为有效编号）
        for (auto& armor : car.armors) {
            if (armor.class_label != 0 && armor.class_label != 5 &&
                armor.confidence > max_confidence) {
                // 更新最高置信度的装甲板信息
                max_rect = cv::Rect(
                    armor.left + car.car.left, armor.top + car.car.top,
                    armor.right - armor.left, armor.bottom - armor.top);
                max_confidence = armor.confidence;
                car.number = armor.class_label;  // 车辆编号=装甲板分类编号
            }
        }

        // 无有效装甲板→debug模式绘制车辆框，跳过后续处理
        if (max_confidence == 0) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);  // 白色框
            continue;
        }

        // 确保最高置信度装甲板ROI安全（避免越界）
        auto safe_rect = getSafeRect(img, max_rect);
        auto max_mat = img(safe_rect);

        // 5. 第五步：车辆颜色判断（基于最高置信度装甲板的颜色）
        car.color = getColor(max_mat);
        // 计算车辆中心坐标（基于最高置信度装甲板的中心）
        car.center = cv::Point2f(
            max_rect.x + max_rect.width / 2,  // 中心x=左坐标+宽度/2
            max_rect.y + max_rect.height / 2  // 中心y=上坐标+高度/2
        );

        // 根据颜色填充检测结果消息
        if (car.color == 0) {  // 蓝色车辆
            // 车辆编号-1作为数组索引（假设编号从1开始）
            detect_result.blue_x[car.number - 1] = car.center.x;
            detect_result.blue_y[car.number - 1] = car.center.y;
            // 异常检查：中心坐标为0且编号非0→打印错误日志
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: blue car center is 0");
            }
            // debug模式：绘制蓝色车辆框+置信度文本
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 0, 0),
                              2);  // 蓝色框
                cv::putText(
                    img,
                    std::to_string(car.car.confidence),  // 车辆检测置信度
                    cv::Point(car.car.left,
                              car.car.top),  // 文本位置（车辆左上角）
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255),
                    2);
            }
        }
        if (car.color == 2) {  // 红色车辆
            detect_result.red_x[car.number - 1] = car.center.x;
            detect_result.red_y[car.number - 1] = car.center.y;
            // 异常检查：中心坐标为0且编号非0→打印错误日志
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: red car center is 0");
            }
            // debug模式：绘制红色车辆框+置信度文本
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(0, 0, 255),
                              2);  // 红色框
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        if (car.color == 1) {  // 颜色未识别（异常）→debug模式绘制白色框
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
        }
    }

    // 6. 第六步：发布检测结果（设置消息时间戳为图像时间戳，确保同步）
    detect_result.header.stamp = msg->header.stamp;
    pub->publish(detect_result);

    // 计算并打印检测总耗时（毫秒）
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                  begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms"
              << std::endl;

    // 调整图像大小并显示（适配窗口）
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    cv::imshow("detect", final_img);

    // 处理键盘事件：按'r'键切换debug模式（显示/隐藏绘制信息）
    auto key = cv::waitKey(1);  // 等待1ms，避免阻塞
    if (key == 'r') {
        debug = !debug;
    }
}
}  // namespace tdt_radar

// ROS 2节点注册宏：将Detect类注册为组件，使ROS能发现并加载该节点
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)
