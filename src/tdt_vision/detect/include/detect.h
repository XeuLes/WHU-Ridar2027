#ifndef RADAR_DETECT_H
#define RADAR_DETECT_H

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "classify.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "NvidiaInterface.hpp"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "vision_interface/msg/detect_result.hpp"
#include "yolos.hpp"
#include "BaseInfer.hpp"
#include "uav.hpp"
#include "uav_tracker.h"
#include "uav_light_detector.h"
#include <fstream>
namespace tdt_radar {

// --- 新增：无人机结构体 ---
class Uav {
public:
    yolo::Box   box;        // 第一层检测到的原始框
    cv::Rect    uav_rect;   // 裁剪后的 ROI
    cv::Point2f center;     // 预留：第二层精细定位的中心点坐标
    int color = 1;          // 颜色：0-蓝色，2-红色，1-未知
    int track_id = -1;      // 追踪ID
};

// SORT追踪器 - 简化版本（无卡尔曼滤波）
struct SortTrack {
    int id;                          // 追踪ID
    yolo::Box box;                   // 当前框
    int age = 0;                     // 追踪年龄（帧数）
    int hits = 0;                    // 连续命中次数
    int miss = 0;                    // 连续丢失次数
    bool active = false;             // 是否活跃
    int color = 1;                   // 颜色
};

struct TrackedUav {
    yolo::Box box;          // 当前检测框
    int hit_count = 0;      // 连续命中的帧数
    int miss_count = 0;     // 连续丢失的帧数
    bool confirmed = false; // 是否已激活（通过闪烁过滤）
    int color = 1;          // 颜色：0-蓝色，2-红色，1-未知
    int track_id = -1;      // 追踪ID
};

class Detect final : public rclcpp::Node {
public:
    explicit Detect(const rclcpp::NodeOptions& options);
    void callback(const std::shared_ptr<sensor_msgs::msg::Image> msg);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
        compressed_image_sub;

private:
    std::shared_ptr<Infer<yolo::BoxArray>>     yolo;
    std::shared_ptr<Infer<yolo::BoxArray>>     armor_yolo;
    std::shared_ptr<Infer<yolo::BoxArray>>     uav_detector;
    std::shared_ptr<Infer<int>> classifier;
    std::string image_topic_;
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr pub;

    bool        if_rosbag = false;
    int         EnemyColor;
    int         debug;
    std::string yolo_path;
    std::string armor_path;
    std::string classify_path;
    std::string uav_path;

    std::vector<TrackedUav> tracked_uavs;
    const int HIT_THRESHOLD = 3;   // 连续3帧出现才认为有效
    const int MAX_MISS_COUNT = 5;  // 连续丢失5帧才删除
    
    // UAV DeepSORT跟踪器（结合卡尔曼滤波）
    std::shared_ptr<UAVDeepSORTTracker> uav_tracker_;
    
    // UAV跟踪参数
    const int UAV_MAX_TRACKING_AGE = 20;  // UAV卡尔曼预测框最大保持帧数
    
    // 灯条检测器
    std::shared_ptr<class LightBarDetector> light_bar_detector_;
};
class Car {
public:
    cv::Rect       car_rect;
    yolo::Box      car;
    yolo::BoxArray armors;
    cv::Point2f    center;
    cv::Rect       center_rect;
    int            number = 0;
    int            color = 1;
};


}  // namespace tdt_radar

#endif
