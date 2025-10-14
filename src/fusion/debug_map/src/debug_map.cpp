#include <cv_bridge/cv_bridge.h>
#include <cstring>  // 新增：用于memset初始化数组
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
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
                cv::resize(map, map,
                           cv::Size(28 * 25, 15 * 25));  // 缩放地图
                RCLCPP_INFO(this->get_logger(), "地图加载成功！尺寸：%dx%d",
                            map.cols, map.rows);
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

private:
    // 订阅者（原逻辑不变，改为private成员更规范）
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_result_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        camera_detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;

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