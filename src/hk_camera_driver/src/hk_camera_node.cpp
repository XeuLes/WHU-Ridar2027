// 1. 导入ROS2核心库（类比雷达的rclcpp）
#include "rclcpp/rclcpp.hpp"
// 导入ROS2图像消息类型（发布图像需要）
#include "sensor_msgs/msg/image.hpp"
// 导入cv_bridge：转换OpenCV图像和ROS2消息（关键工具）
#include "cv_bridge/cv_bridge.h"
// 导入OpenCV：处理图像（显示、格式转换）
#include <opencv2/opencv.hpp>
// 导入海康MVS SDK头文件（连接相机必须）
#include "MvCameraControl.h"

// 2. 定义相机节点类（继承自ROS2的Node类，类比雷达的LivoxRosDriver类）
class HkCameraNode : public rclcpp::Node {
public:
    // 构造函数：节点初始化（启动时会执行）
    HkCameraNode() : Node("hk_camera_node") {
        // 3. 读取参数文件里的配置（类比雷达读取lidar_ip）
        // 3.1 声明参数（如果参数文件里没有，就用默认值）
        this->declare_parameter("camera_ip", "192.168.1.64");  // 相机IP（默认值，参数文件会覆盖）
        this->declare_parameter("resolution", std::vector<int64_t>{1280, 720});  // 分辨率
        this->declare_parameter("frame_id", "camera_link");  // 坐标系
        this->declare_parameter("publish_topic", "/camera/image_raw");  // 发布话题

        // 3.2 从参数文件中获取实际值（覆盖默认值）
        camera_ip_ = this->get_parameter("camera_ip").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        auto resolution = this->get_parameter("resolution").as_integer_array();
        width_ = resolution[0];  // 图像宽度（1280）
        height_ = resolution[1]; // 图像高度（720）

        // 4. 创建图像发布者（类比雷达创建PointCloud2发布者）
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            this->get_parameter("publish_topic").as_string(),  // 发布到哪个话题
            10  // 消息队列长度（满了就丢旧的，新手不用改）
        );

        // 5. 连接海康相机（类比雷达连接硬件）
        if (!connect_to_camera()) {
            // 连接失败就报错，退出节点
            RCLCPP_ERROR(this->get_logger(), "Failed to connect Hikvision camera! Check IP or SDK!");
            rclcpp::shutdown();  // 关闭ROS2节点
        }

        // 6. 启动取图线程（类比雷达的UDP数据接收线程）
        // 为什么用线程？因为取图要一直循环，不能卡主节点初始化
        grab_image_thread_ = std::thread(&HkCameraNode::grab_and_publish_image, this);
    }

    // 析构函数：节点关闭时执行（释放资源）
    ~HkCameraNode() {
        // 断开相机连接（必须做，否则下次可能连不上）
        if (camera_handle_ != nullptr) {
            MV_CC_StopGrabbing(camera_handle_);  // 停止取流
            MV_CC_CloseDevice(camera_handle_);   // 关闭设备
            MV_CC_DestroyHandle(&camera_handle_); // 销毁句柄
        }
        // 等待取图线程结束
        grab_image_thread_.join();
    }

private:
    // 函数1：连接相机（核心：调用海康MVS SDK接口）
    bool connect_to_camera() {
        // 1. 枚举所有可用相机（SDK标准流程）
        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(device_list));

        // 枚举网口（GigE）和USB相机
        if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list) != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to search cameras! Check SDK or hardware!");
            return false;
        }

        if (device_list.nDeviceNum == 0) {
            RCLCPP_ERROR(this->get_logger(), "No camera found! Please connect camera first!");
            return false;
        }

        // 2. 遍历设备列表，找到目标IP的相机（修正：循环变量用unsigned int，解决signed/unsigned警告）
        int target_index = -1;  // 改回int类型，避免和-1比较的类型不匹配
        for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {  // i改为unsigned int
            // 只处理网口相机（GigE）
            if (device_list.pDeviceInfo[i]->nTLayerType != MV_GIGE_DEVICE) {
                continue;
            }

            // 修正：从nCurrentIp（32位整数）拆出点分十进制IP（替代不存在的nCurrentIpAddr）
            unsigned int ip_int = device_list.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp;
            char camera_ip[16] = {0};
            sprintf(camera_ip, "%d.%d.%d.%d",
                    (ip_int >> 24) & 0xFF,  // 取IP的第1段（最高8位）
                    (ip_int >> 16) & 0xFF,  // 取IP的第2段
                    (ip_int >> 8) & 0xFF,   // 取IP的第3段
                    ip_int & 0xFF);         // 取IP的第4段（最低8位）

            // 对比IP是否与参数一致
            if (strcmp(camera_ip, camera_ip_.c_str()) == 0) {
                target_index = i;
                break;
            }
        }

        if (target_index == -1) {
            RCLCPP_ERROR(this->get_logger(), "Camera with IP %s not found! Check IP in config file!", camera_ip_.c_str());
            return false;
        }

        // 3. 创建相机句柄
        if (MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[target_index]) != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create camera handle!");
            return false;
        }

        // 4. 打开相机（修正：第三个参数传0，替代nullptr，符合short unsigned int类型）
        if (MV_CC_OpenDevice(camera_handle_, MV_ACCESS_Exclusive, 0) != MV_OK) {  // 第三个参数改为0
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera! Maybe it's used by other program?");
            MV_CC_DestroyHandle(&camera_handle_);
            return false;
        }

        // 5. 开始连续取流（修正：用MV_CONTINUOUS_GRAB，替代不存在的MV_GRAB_MODE_CONTINUOUS）
        // 如果MV_CONTINUOUS_GRAB仍报错，直接用数值1（SDK中连续取流模式值通常为1）
        if (MV_CC_StartGrabbing(camera_handle_) != MV_OK) {  // 1 = 连续取流模式（固定值）
            RCLCPP_ERROR(this->get_logger(), "Failed to start grabbing image!");
            MV_CC_CloseDevice(camera_handle_);
            MV_CC_DestroyHandle(&camera_handle_);
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Successfully connected to camera! IP: %s", camera_ip_.c_str());
        return true;
    }


    // 函数2：循环取图并发布（核心：图像数据处理）
    void grab_and_publish_image() {
        MV_FRAME_OUT frame_data;  // SDK的图像数据结构
        memset(&frame_data, 0, sizeof(MV_FRAME_OUT));  // 初始化数据

        // 一直循环取图（直到节点关闭）
        while (rclcpp::ok()) {
            // 1. 从相机取一帧图像（超时1秒，取不到就重试）
            if (MV_CC_GetImageBuffer(camera_handle_, &frame_data, 1000) != MV_OK) {
                continue;  // 取图失败，跳过这次
            }

            // 2. 转换SDK图像到OpenCV格式（SDK输出是BGR8，OpenCV能直接用）
            // frame_data.pBufAddr：图像数据的内存地址
            // width_/height_：分辨率（从参数来）
            // CV_8UC3：8位无符号整数，3通道（BGR）
            cv::Mat opencv_image(height_, width_, CV_8UC3, frame_data.pBufAddr);

            // 3. 转换OpenCV图像到ROS2消息（关键一步，类比雷达转PointCloud2）
            // cv_bridge::CvImage：把OpenCV图像包装成ROS2能认的格式
            // "bgr8"：图像编码（BGR，8位）
            auto ros_image_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(),  // 消息头（包含时间戳、坐标系）
                "bgr8",                   // 图像编码
                opencv_image              // OpenCV图像
            ).toImageMsg();  // 转换成ROS2的Image消息

            // 4. 给消息加时间戳和坐标系（必须加，多传感器融合要用）
            ros_image_msg->header.frame_id = frame_id_;  // 坐标系（和参数文件一致）
            ros_image_msg->header.stamp = this->now();   // 当前时间戳（ROS2的系统时间）

            // 5. 发布图像消息（类比雷达发布点云）
            image_publisher_->publish(*ros_image_msg);

            // 6. 释放SDK的图像缓存（必须做，否则内存会炸）
            MV_CC_FreeImageBuffer(camera_handle_, &frame_data);
        }
    }

    // 成员变量（类比雷达驱动的全局变量，存节点需要的状态）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;  // 图像发布者
    void* camera_handle_ = nullptr;  // 相机句柄（SDK用）
    std::string camera_ip_;          // 相机IP（从参数来）
    std::string frame_id_;           // 坐标系名称（从参数来）
    int width_, height_;             // 图像分辨率（从参数来）
    std::thread grab_image_thread_;  // 取图线程（后台跑）
};

// 3. 主函数：启动节点（类比雷达的main函数）
int main(int argc, char* argv[]) {
    // 初始化ROS2
    rclcpp::init(argc, argv);
    // 创建相机节点并运行
    rclcpp::spin(std::make_shared<HkCameraNode>());
    // 关闭ROS2
    rclcpp::shutdown();
    return 0;
}

