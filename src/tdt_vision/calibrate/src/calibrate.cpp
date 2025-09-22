#include "calibrate.h"
#include <string>

// 雷达相关功能的命名空间
namespace tdt_radar {

/**
 * @brief Calibrate类构造函数，初始化校准节点及相关资源
 * @param options ROS节点选项
 */
Calibrate::Calibrate(const rclcpp::NodeOptions& options)
    : Node("radar_calibrate_node",
           options)  // 初始化ROS节点，节点名为"radar_calibrate_node"
{
    std::cout << "Calibrate start" << std::endl;

    // 创建并配置OpenCV窗口：主校准窗口
    cv::namedWindow("calibrate", cv::WINDOW_AUTOSIZE);  // 创建窗口
    cv::resizeWindow("calibrate", 1440, 900);  // 调整窗口大小
    cv::moveWindow("calibrate", 1920 - 1440,
                   1080 - 900);  // 移动窗口到指定位置（右上角）

    // 创建并配置ROI预览窗口（用于精细调整选点）
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("ROI", 400, 400);
    cv::moveWindow("ROI", 0, 0);  // 移动窗口到左上角

    // 设置鼠标回调函数，处理"calibrate"窗口的鼠标事件
    cv::setMouseCallback("calibrate", mousecallback, 0);

    // 读取相机内参和畸变系数（从yaml配置文件）
    cv::FileStorage fs;
    fs.open("./config/camera_params.yaml",
            cv::FileStorage::READ);        // 打开配置文件
    fs["camera_matrix"] >> camera_matrix;  // 读取相机内参矩阵
    fs["dist_coeffs"] >> dist_coeffs;      // 读取畸变系数
    fs.release();                          // 释放文件资源

    // 初始化真实世界坐标点（校准用的参考点，如己方堡垒、己方高塔等）
    real_points.push_back(self_FORTRESS);  // 己方堡垒
    real_points.push_back(self_Tower);     // 己方高塔
    real_points.push_back(enemy_Base);     // 敌方基地
    real_points.push_back(enemy_Tower);    // 敌方高塔
    real_points.push_back(enemy_High);     // 敌方高地

    // 初始化解析器（用于处理矩阵变换等）
    parser_ = new parser();

    // 打印初始化成功的日志信息（ASCII艺术字）
    RCLCPP_INFO(this->get_logger(), "\n"
                                    "  ______    ____  ______\n"
                                    " /_  __/   / __ \\/_  __/\n"
                                    "  / /_____/ / / / / /   \n"
                                    " / /_____/ /_/ / / /    \n"
                                    "/_/     /_____/ /_/     ");

    // 创建普通图像订阅者，订阅"camera_image"话题
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image",
        rclcpp::SensorDataQoS(),  // 话题名和QoS设置（传感器数据类型）
        std::bind(&Calibrate::callback, this,
                  std::placeholders::_1));  // 绑定回调函数

    // 创建压缩图像订阅者，订阅"compressed_image"话题
    compressed_image_sub =
        this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "compressed_image", rclcpp::SensorDataQoS(),
            std::bind(&Calibrate::compressed_callback, this,
                      std::placeholders::_1));  // 绑定压缩图像回调函数

    std::cout << "Calibrate end" << std::endl;
}

/**
 * @brief 普通图像消息回调函数，处理未压缩的相机图像
 * @param msg 图像消息指针
 */
void Calibrate::callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 将ROS图像消息转换为OpenCV格式（BGR8编码）
    auto    img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    cv::Mat calib_img;
    cv::resize(img, calib_img,
               cv::Size(1536, 1125));  // 调整图像大小用于校准
    cvimage_ = calib_img;  // 保存到成员变量，供鼠标回调使用

    if (is_calibrating) {  // 如果处于校准模式
        // 显示已选择的点数量
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);
        // 提示按'n'键确认选点
        cv::putText(img, "Press 'n' to add good point", cv::Point(50, 400),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);

        // 如果选点数量等于真实参考点数量，执行校准计算
        if (pick_points.size() == real_points.size()) {
            solve();                   // 计算旋转和平移矩阵
            parser_->Change_Matrix();  // 更新解析器中的矩阵
        }
    } else {                    // 非校准模式
        parser_->draw_ui(img);  // 绘制UI界面
        // 提示按Enter进入校准模式
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }

    auto temp = img.clone();  // 临时克隆图像（未使用，可能用于预留扩展）
    cv::resize(img, img, cv::Size(1536, 1125));  // 调整图像大小用于显示
    cv::imshow("calibrate", img);  // 在"calibrate"窗口显示图像

    // 处理键盘事件（等待10ms）
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:                    // Enter键ASCII码
        is_calibrating = true;  // 进入校准模式
        break;
    default:
        break;
    }
}

/**
 * @brief 压缩图像消息回调函数，处理压缩格式的相机图像
 * @param msg 压缩图像消息指针
 */
void Calibrate::compressed_callback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    // 解码压缩图像为OpenCV格式
    auto    img = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    cv::Mat calib_img;
    cv::resize(img, calib_img,
               cv::Size(1536, 1125));  // 调整图像大小用于校准
    cvimage_ = calib_img;  // 保存到成员变量，供鼠标回调使用

    if (is_calibrating) {  // 校准模式
        // 显示已选择的点数量
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);

        // 选点数量足够时执行校准
        if (pick_points.size() == real_points.size()) {
            solve();
            parser_->Change_Matrix();
        }
    } else {                    // 非校准模式
        parser_->draw_ui(img);  // 绘制UI
        // 提示按Enter进入校准
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }

    auto temp = img.clone();                     // 临时克隆图像
    cv::resize(img, img, cv::Size(1536, 1125));  // 调整显示大小
    cv::imshow("calibrate", img);                // 显示图像

    // 处理键盘事件
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:                    // Enter键
        is_calibrating = true;  // 进入校准模式
        break;
    default:
        break;
    }
}

/**
 * @brief 鼠标回调函数，处理"calibrate"窗口的鼠标事件（选点和预览）
 * @param event 鼠标事件类型
 * @param x 鼠标x坐标
 * @param y 鼠标y坐标
 * @param flags 事件标志
 * @param userdata 用户数据（未使用）
 */
void mousecallback(int event, int x, int y, int flags, void* userdata)
{
    int temp_key = 0;  // 临时存储键盘输入

    switch (event) {
    case cv::EVENT_LBUTTONDOWN:  // 鼠标左键按下事件
        if (is_calibrating) {    // 仅在校准模式下处理
            // 循环等待键盘输入（WASD微调，n确认）
            do {
                temp_key = cv::waitKey(10);
                switch (temp_key) {
                case 'w':  // 上移
                    y -= 1;
                    break;
                case 'a':  // 左移
                    x -= 1;
                    break;
                case 's':  // 下移
                    y += 1;
                    break;
                case 'd':  // 右移
                    x += 1;
                    break;
                }
                // 限制坐标范围，避免超出图像边界
                x = std::max(50, std::min(x, cvimage_.cols - 50));
                y = std::max(50, std::min(y, cvimage_.rows - 50));

                // 提取鼠标位置周围100x100的ROI区域（用于精细预览）
                cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
                cv::Mat dst;
                cv::resize(roi, dst,
                           cv::Size(400, 400));  // 放大到400x400显示
                // 绘制十字准线
                cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                         cv::Scalar(0, 0, 255), 1);  // 竖线
                cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                         cv::Scalar(0, 0, 255), 1);  // 横线
                cv::imshow("ROI", dst);  // 在ROI窗口显示预览

            } while (temp_key != 'n');  // 按'n'键确认选点

            // 坐标缩放（根据图像显示比例调整到原始尺寸）
            x *= 1.3333333333 * 2;
            y *= 1.3333333333 * 2;
            std::cout << "x:" << x << " y:" << y
                      << std::endl;                    // 打印选点坐标
            pick_points.push_back(cv::Point2f(x, y));  // 保存选点到列表
        }
        break;

    case cv::EVENT_MOUSEMOVE:  // 鼠标移动事件
        // 限制在图像有效区域内（避免越界）
        if (x > cvimage_.cols - 50 || y > cvimage_.rows - 50 || x < 50 ||
            y < 50)
            break;
        // 实时显示鼠标周围ROI区域
        cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
        cv::Mat dst;
        cv::resize(roi, dst, cv::Size(400, 400));  // 放大显示
        // 绘制十字准线
        cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                 cv::Scalar(0, 0, 255), 1);
        cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                 cv::Scalar(0, 0, 255), 1);
        cv::imshow("ROI", dst);  // 更新ROI窗口
        break;
    }
}

/**
 * @brief 执行校准计算，求解相机外参（旋转向量和平移向量）
 */
void Calibrate::solve()
{
    // 使用PnP算法求解旋转向量(rvec)和平移向量(tvec)
    // 参数：真实世界点、图像点、相机内参、畸变系数、输出旋转向量、输出平移向量、是否使用初始值、求解方法
    cv::solvePnP(real_points, pick_points, camera_matrix, dist_coeffs, rvec,
                 tvec, 0, cv::SOLVEPNP_EPNP);

    // 打印计算结果
    std::cout << "rvec:" << rvec << std::endl;  // 旋转向量
    std::cout << "tvec:" << tvec << std::endl;  // 平移向量

    // 将外参保存到yaml文件
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml",
            cv::FileStorage::WRITE);  // 打开输出文件
    fs << "world_rvec" << rvec;       // 写入旋转向量
    fs << "world_tvec" << tvec;       // 写入平移向量
    fs.release();                     // 释放资源

    // 重置校准状态
    pick_points.clear();     // 清空选点列表
    is_calibrating = false;  // 退出校准模式
}
}  // namespace tdt_radar

// 注册ROS 2节点组件，使节点可被发现和加载
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Calibrate);
