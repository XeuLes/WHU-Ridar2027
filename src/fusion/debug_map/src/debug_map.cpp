#include <cv_bridge/cv_bridge.h>
#include <cstring>  // 新增：用于memset初始化数组
#include <fstream>
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>  // 新增：cv_bridge转换需要Header类
#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>
#include <vision_interface/msg/radar2_sentry.hpp>
#include <vision_interface/msg/radar_warn.hpp>
// 新增：包含sensor_msgs/PointCloud2的定义
#include <sensor_msgs/msg/point_cloud2.hpp>
// 新增：包含PCL与ROS点云转换的工具（用于plane_callback中fromROSMsg）
#include <pcl_conversions/pcl_conversions.h>
// 新增：包含PCL点云类型（用于存储转换后的飞机坐标）
#include <pcl/point_types.h>

namespace debug_map {
class DebugMap : public rclcpp::Node {
public:
    // 构造函数：初始化列表中显式初始化所有成员变量（解决随机值问题）
    explicit DebugMap(const rclcpp::NodeOptions& options)
        : Node("debug_map_node", options)  // 节点名与Launch一致
          ,
          hero_count1(0)  // 初始化英雄状态计数1
          ,
          hero_count2(0)  // 初始化英雄状态计数2
          ,
          count(0)
    {
        // 1. 声明并读取Launch传递的地图路径参数（解决硬编码问题）
        this->declare_parameter<std::string>("map_path",
                                             "");  // 声明参数，默认空
        std::string map_path = this->get_parameter("map_path").as_string();

        // 2. 加载地图（带日志提示，方便调试）
        if (map_path.empty()) {
            RCLCPP_ERROR(this->get_logger(),
                         "地图路径参数map_path未设置！");
            map = cv::Mat();  // 空图像，避免后续崩溃
        } else {
            map = cv::imread(map_path);
            if (map.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                             "地图文件加载失败！路径：%s",
                             map_path.c_str());
            } else {
                // 打印原始尺寸（确认加载的图像是否有效）
                RCLCPP_INFO(this->get_logger(), "原始地图尺寸：%dx%d（cols×rows）", map.cols, map.rows);

                // 计算目标尺寸并校验（核心排查：缩放参数是否合法）
                int target_w = 28 * 25;  // 700
                int target_h = 15 * 25;  // 375
                if (target_w <= 0 || target_h <= 0) {
                    RCLCPP_ERROR(this->get_logger(), "无效缩放尺寸：width=%d, height=%d（必须为正数）", target_w, target_h);
                    map = cv::Mat();
                } else {
                    // 执行缩放并检查结果
                    cv::resize(map, map, cv::Size(target_w, target_h));
                    if (map.empty()) {
                        RCLCPP_ERROR(this->get_logger(), "缩放失败！原始尺寸：%dx%d → 目标尺寸：%dx%d", map.cols, map.rows, target_w, target_h);
                    } else {
                        RCLCPP_INFO(this->get_logger(), "地图加载并缩放成功，最终尺寸：%dx%d", map.cols, map.rows);
                    }
                }
            }
        }

        // 3. 初始化数组变量（避免随机值导致的逻辑错误）
        memset(blue_time, 0, sizeof(blue_time));  // 蓝方时间戳初始化为0
        memset(red_time, 0, sizeof(red_time));  // 红方时间戳初始化为0
        memset(relax, false, sizeof(relax));  // 放松状态初始化为false
        memset(relax_time, 0, sizeof(relax_time));  // 放松时间初始化为0
        memset(blue_update, 0,
               sizeof(blue_update));  // 蓝方更新时间初始化为0
        memset(red_update, 0, sizeof(red_update));  // 红方更新时间初始化为0
        // 坐标点初始化为(0,0)（循环赋值，避免随机坐标）
        for (int i = 0; i < 6; i++) {
            blue_point[i] = cv::Point2f(0, 0);
            red_point[i] = cv::Point2f(0, 0);
        }
        // 初始化飞机变量（避免随机值）
        memset(plane_update, 0, sizeof(plane_update));
        plane_point[0] = cv::Point2f(0, 0);  // 初始化为(0,0)
        plane_point[1] = cv::Point2f(0, 0);  // 初始化为(0,0)

        // 4. 创建订阅者（保持原逻辑，QoS与话题匹配）
        detect_result_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/kalman_detect", 10,
                std::bind(&DebugMap::callback, this,
                          std::placeholders::_1));

        camera_detect_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/resolve_result", rclcpp::SensorDataQoS(),
                std::bind(&DebugMap::camera_callback, this,
                          std::placeholders::_1));

        match_info_sub =
            this->create_subscription<vision_interface::msg::MatchInfo>(
                "/match_info", 10,
                std::bind(&DebugMap::save_match_info, this,
                          std::placeholders::_1));

        // 新增：订阅飞机坐标话题（lidar_other_cluster）
        plane_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar_other_cluster",  // 飞机坐标发布的话题
            rclcpp::SensorDataQoS(),
            std::bind(&DebugMap::plane_callback, this, std::placeholders::_1)
        );

        // 5. 创建发布者（保持原逻辑）
        radar_warn_pub =
            this->create_publisher<vision_interface::msg::RadarWarn>(
                "/hero_state", 10);
        debug_map_pub =
            this->create_publisher<sensor_msgs::msg::Image>("/map_2d", 10);
        radar2sentry_pub =
            this->create_publisher<vision_interface::msg::Radar2Sentry>(
                "/Radar2Sentry", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "DebugMap组件初始化完成！");
    }

    // 保存比赛信息（原逻辑不变，增加日志）
    void save_match_info(
        const std::shared_ptr<vision_interface::msg::MatchInfo> msg)
    {
        this->match_info = *msg;
        if (msg->self_color == 1) {
            match_info.self_color = 2;
            RCLCPP_INFO(this->get_logger(), "自身颜色修正：从1改为2");
        }
        RCLCPP_DEBUG(this->get_logger(), "当前自身颜色：%d",
                     match_info.self_color);
    }

    // 地图可视化与发布（原逻辑不变，增加空地图判断）
    void show_map()
    {
        // 避免空地图导致崩溃
        if (map.empty()) {
            RCLCPP_WARN(this->get_logger(), "地图为空，跳过可视化");
            return;
        }

        auto   now_time = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now_time.time_since_epoch())
                          .count() /
                      1000.0;
        auto clone_map = map.clone();

        // 绘制蓝/红方目标（原逻辑不变）
        for (int i = 0; i < 6; i++) {
            int number = i + 1;
            if (number == 6)
                number++;  // 跳过6号（原逻辑保留）

            // 绘制蓝方目标
            if (blue_point[i].x * blue_point[i].y != 0 &&
                time - blue_update[i] < 0.5) {
                cv::Point2f point(clone_map.cols * blue_point[i].x / 28,
                                  clone_map.rows * (15 - blue_point[i].y) /
                                      15);
                cv::circle(clone_map, point, 10, cv::Scalar(200, 0, 0), -1);
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }

            // 绘制红方目标
            if (red_point[i].x * red_point[i].y != 0 &&
                time - red_update[i] < 0.5) {
                cv::Point2f point(clone_map.cols * red_point[i].x / 28,
                                  clone_map.rows * (15 - red_point[i].y) /
                                      15);
                cv::circle(clone_map, point, 10, cv::Scalar(0, 0, 200), -1);
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }
        }

        // 绘制2架飞机
        auto plane_now_time = std::chrono::system_clock::now();
        double plane_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                plane_now_time.time_since_epoch())
                                .count() / 1000.0;

        for (int i = 0; i < 2; ++i) {  // 循环2次，绘制2架飞机
            // 检查第i架飞机坐标是否有效（500ms内更新过）
            if (plane_time - plane_update[i] < 0.5) {
                // 转换为图像坐标
                cv::Point2f point(
                    clone_map.cols * plane_point[i].x / 28,
                    clone_map.rows * (15 - plane_point[i].y) / 15
                );

                // 越界检查
                if (point.x >= 0 && point.x < clone_map.cols &&
                    point.y >= 0 && point.y < clone_map.rows) {
                    // 绘制黄色实心圆（两架飞机用相同颜色，通过文字区分）
                    cv::circle(clone_map, point, 12, cv::Scalar(0, 255, 255), -1);
                    // 绘制带ID的文字（如"P1"、"P2"）
                    std::string label = "P" + std::to_string(i+1);  // P1=第1架，P2=第2架
                    cv::putText(clone_map, label, 
                                cv::Point(point.x - 8, point.y + 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));  // 黑色文字
                }
            }
        }

        // 发布地图图像（原逻辑不变）
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "map";  // 与Launch中地图坐标系一致

        try {
            sensor_msgs::msg::Image::SharedPtr map_msg =
                cv_bridge::CvImage(header, "bgr8", clone_map).toImageMsg();
            debug_map_pub->publish(*map_msg);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge转换失败：%s",
                         e.what());
        }

        // 显示窗口（组件模式下建议保留，方便调试；若崩溃可注释）
        cv::imshow("Debug Map", clone_map);
        cv::waitKey(1);  // 必须加，否则窗口无响应
    }

    // 相机检测结果回调（原逻辑不变）
    void camera_callback(
        const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;

        for (int i = 0; i < 6; i++) {
            // 更新蓝方目标（5秒间隔）
            if (msg->blue_x[i] * msg->blue_y[i] != 0) {
                if (time - blue_time[i] > 5) {
                    blue_point[i] =
                        cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                    if (!match_info.self_color) {  // 自身颜色为0时翻转坐标
                        blue_point[i] = cv::Point2f(28 - msg->blue_x[i],
                                                    15 - msg->blue_y[i]);
                    }
                    blue_update[i] = time;
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "相机回调：更新蓝方%d号目标坐标(%.1f, %.1f)", i + 1,
                        blue_point[i].x, blue_point[i].y);
                }
            }

            // 更新红方目标（5秒间隔）
            if (msg->red_x[i] * msg->red_y[i] != 0) {
                if (time - red_time[i] > 5) {
                    red_point[i] =
                        cv::Point2f(msg->red_x[i], msg->red_y[i]);
                    if (!match_info.self_color) {  // 自身颜色为0时翻转坐标
                        red_point[i] = cv::Point2f(28 - msg->red_x[i],
                                                   15 - msg->red_y[i]);
                    }
                    red_update[i] = time;
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "相机回调：更新红方%d号目标坐标(%.1f, %.1f)", i + 1,
                        red_point[i].x, red_point[i].y);
                }
            }
        }

        show_map();  // 触发地图更新
    }

    // 卡尔曼滤波结果回调（原逻辑不变，增加计数边界保护）
    void
    callback(const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;

        // 更新目标坐标（无时间间隔限制，卡尔曼数据更可靠）
        for (int i = 0; i < 6; i++) {
            if (msg->blue_x[i] * msg->blue_y[i] != 0) {
                blue_point[i] = cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                blue_time[i] = time;
                blue_update[i] = time;
            }
            if (msg->red_x[i] * msg->red_y[i] != 0) {
                red_point[i] = cv::Point2f(msg->red_x[i], msg->red_y[i]);
                red_time[i] = time;
                red_update[i] = time;
            }
        }

        show_map();  // 触发地图更新

        // 发布英雄状态（增加计数边界保护，避免负数值）
        vision_interface::msg::RadarWarn radar_warn;
        hero_count1 = std::max(0, hero_count1);  // 计数不小于0
        hero_count2 = std::max(0, hero_count2);
        if (hero_count1 > 10) {
            radar_warn.hero_state = 1;
            radar_warn_pub->publish(radar_warn);
            RCLCPP_INFO(this->get_logger(),
                        "发布英雄状态：1（hero_count1=%d）", hero_count1);
        } else if (hero_count2 > 10) {
            radar_warn.hero_state = 2;
            radar_warn_pub->publish(radar_warn);
            RCLCPP_INFO(this->get_logger(),
                        "发布英雄状态：2（hero_count2=%d）", hero_count2);
        }

        // 英雄状态计数逻辑（原逻辑不变）
        if (match_info.self_color == 0) {  // 自身为蓝方
            if (red_point[0].x > (28 - 8.668)) {
                hero_count1++;
                hero_count2--;
            } else if (red_point[0].x < (28 - 20.3) &&
                       red_point[0].x > (28 - 25.075) &&
                       red_point[0].y < 15 && red_point[0].y > 10.3) {
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        } else if (match_info.self_color == 2) {  // 自身为红方
            if (blue_point[0].x < 8.668) {
                hero_count1++;
                hero_count2--;
            } else if (blue_point[0].x > 20.3 && blue_point[0].x < 25.075 &&
                       blue_point[0].y > 0 &&
                       blue_point[0].y < (15 - 10.3)) {
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        }

        // 发布雷达到哨兵的目标坐标（原逻辑不变）
        vision_interface::msg::Radar2Sentry radar2sentry;
        if (match_info.self_color == 0) {  // 自身蓝方：发布红方目标
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else if (time - red_update[i] < 0.5) {
                        radar2sentry.radar_enemy_x[i] = red_point[i].x;
                        radar2sentry.radar_enemy_y[i] = red_point[i].y;
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    }
                }
            }
        } else if (match_info.self_color == 2) {  // 自身红方：发布蓝方目标
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else if (time - blue_update[i] < 0.5) {
                        radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                        radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    }
                }
            }
        }
        radar2sentry_pub->publish(radar2sentry);
    }

    // 处理飞机坐标话题（lidar_other_cluster）
    void plane_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 1. 将ROS点云消息转换为PCL点云（获取飞机坐标）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_INFO(this->get_logger(), "飞机点云为空，不更新坐标");
            return;
        }

        // 2. 提取飞机坐标（假设点云中的每个点是一架飞机的聚类中心）
        auto now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count() / 1000.0;

        // 处理前2架飞机（点云中的前2个点）
        for (int i = 0; i < 2; ++i) {  // 循环2次，对应2架飞机
            if (i >= cloud->points.size()) {
                // 若点云数量不足2个，跳过超出的飞机（保持上次坐标）
                RCLCPP_WARN(this->get_logger(), "飞机点云数量不足2个（仅%zu个），第%d架飞机不更新",
                            cloud->points.size(), i+1);
                continue;
            }

            // 提取第i架飞机的坐标
            pcl::PointXYZ plane_pcl = cloud->points[i];
            cv::Point2f plane_map(plane_pcl.x, plane_pcl.y);

            // 2. 根据自身颜色判断是否翻转（核心：与红蓝方逻辑一致）
            if (match_info.self_color == 0) {  // 自身为蓝方，需要翻转坐标
                plane_map.x = 28.0f - plane_map.x;  // x坐标翻转（同红蓝方）
                plane_map.y = 15.0f - plane_map.y;  // y坐标翻转（同红蓝方）
                // RCLCPP_INFO(this->get_logger(), "飞机%d 原始坐标(%.2f, %.2f) → 翻转后(%.2f, %.2f)",
                //             i+1, plane_pcl.x, plane_pcl.y, plane_map.x, plane_map.y);
            }

            // 坐标范围校验
            plane_map.x = std::clamp(plane_map.x, 0.0f, 28.0f);
            plane_map.y = std::clamp(plane_map.y, 0.0f, 15.0f);

            // 更新第i架飞机的坐标和时间戳
            plane_point[i] = plane_map;
            plane_update[i] = time;

            // RCLCPP_INFO(this->get_logger(), "收到第%d架飞机坐标：(%.2f, %.2f)",
            //             i+1, plane_map.x, plane_map.y);
        }

        // 触发地图更新
        show_map();
    }

private:
    // 订阅者（原逻辑不变，改为private成员更规范）
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_result_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        camera_detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr 
        plane_sub;  // 飞机坐标订阅者

    // 发布者（原逻辑不变）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_map_pub;
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr
        radar_warn_pub;
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar2sentry_pub;

    // 目标时间戳（已初始化）
    double blue_time[6];
    double red_time[6];

    // 放松状态（已初始化）
    bool   relax[6];
    double relax_time[6];

    // 目标更新时间（已初始化）
    double blue_update[6];
    double red_update[6];

    // 英雄状态计数（已初始化）
    int hero_count1;
    int hero_count2;

    // 目标坐标（已初始化）
    cv::Point2f blue_point[6];
    cv::Point2f red_point[6];

    // 飞机数据（假设最多2架飞机，可根据实际需求调整数量）
    cv::Point2f plane_point[2];  // 飞机坐标（地图坐标系，x:0-28, y:0-15）
    double      plane_update[2]; // 飞机最新更新时间（用于判断是否过期）

    // 比赛信息与地图（已处理空值）
    vision_interface::msg::MatchInfo match_info;
    cv::Mat                          map;
    int                              count;
};
}  // namespace debug_map

// 【关键修改】删除main函数：组件由容器加载，不需要独立入口（避免编译冲突）
#include "rclcpp_components/register_node_macro.hpp"

// 注册组件宏 - 这是解决插件找不到问题的关键
RCLCPP_COMPONENTS_REGISTER_NODE(debug_map::DebugMap)