#include <cv_bridge/cv_bridge.h>
#include <cstring>  // memset
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <radar_decision_comm/msg/referee_robot_identity.hpp>

#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>
#include <vision_interface/msg/radar2_sentry.hpp>
#include <vision_interface/msg/radar_warn.hpp>

namespace debug_map {

namespace
{

int normalize_self_color(int raw_color, int fallback)
{
    if (raw_color == 0) {
        return 0;
    }
    if (raw_color == 1 || raw_color == 2) {
        return 2;
    }
    return (fallback == 0) ? 0 : 2;
}

}  // namespace

class DebugMap : public rclcpp::Node {
public:
    explicit DebugMap(const rclcpp::NodeOptions& options)
        : Node("debug_map_node", options),
          hero_count1(0),
          hero_count2(0),
          count(0)
    {
        // 1) 地图路径参数
        this->declare_parameter<std::string>("map_path", "");
        this->declare_parameter<bool>("use_match_info_color", true);
        this->declare_parameter<int>("self_color_override", 0);
        this->declare_parameter<bool>("use_referee_identity_color", true);
        std::string map_path = this->get_parameter("map_path").as_string();
        use_match_info_color_ =
            this->get_parameter("use_match_info_color").as_bool();
        self_color_override_ =
            normalize_self_color(this->get_parameter("self_color_override").as_int(), 0);
        use_referee_identity_color_ =
            this->get_parameter("use_referee_identity_color").as_bool();

        // 2) 加载地图
        if (map_path.empty()) {
            RCLCPP_ERROR(this->get_logger(),
                         "地图路径参数map_path未设置！");
            map = cv::Mat();
        } else {
            map = cv::imread(map_path);
            if (map.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                             "地图文件加载失败！路径：%s",
                             map_path.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(),
                            "原始地图尺寸：%dx%d（cols×rows）", map.cols,
                            map.rows);

                int target_w = 28 * 25;  // 700
                int target_h = 15 * 25;  // 375
                if (target_w <= 0 || target_h <= 0) {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "无效缩放尺寸：width=%d, height=%d（必须为正数）",
                        target_w, target_h);
                    map = cv::Mat();
                } else {
                    cv::resize(map, map, cv::Size(target_w, target_h));
                    if (map.empty()) {
                        RCLCPP_ERROR(
                            this->get_logger(),
                            "缩放失败！原始尺寸：%dx%d → 目标尺寸：%dx%d",
                            map.cols, map.rows, target_w, target_h);
                    } else {
                        RCLCPP_INFO(this->get_logger(),
                                    "地图加载并缩放成功，最终尺寸：%dx%d",
                                    map.cols, map.rows);
                    }
                }
            }
        }

        // 3) 初始化数组变量
        memset(blue_time, 0, sizeof(blue_time));
        memset(red_time, 0, sizeof(red_time));
        memset(relax, false, sizeof(relax));
        memset(relax_time, 0, sizeof(relax_time));
        memset(blue_update, 0, sizeof(blue_update));
        memset(red_update, 0, sizeof(red_update));

        // 【修复】循环7次以初始化索引0-6（支持7号车/UAV）
        for (int i = 0; i < 7; i++) {
            blue_point[i] = cv::Point2f(0, 0);
            red_point[i] = cv::Point2f(0, 0);
        }

        // UAV（最终融合结果）初始化
        uav_point[0] = cv::Point2f(0, 0);
        uav_point[1] = cv::Point2f(0, 0);
        uav_update[0] = 0.0;
        uav_update[1] = 0.0;

        // 4) 订阅者
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

        referee_identity_sub =
            this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
                "/radar/referee/robot_identity", 10,
                std::bind(&DebugMap::save_referee_identity, this,
                          std::placeholders::_1));

        // 新增：订阅最终融合的UAV三维坐标（只用xy画地图）
        uav_sub = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/uav_positions", rclcpp::SensorDataQoS(),
            std::bind(&DebugMap::uav_callback, this,
                      std::placeholders::_1));

        // 5) 发布者
        radar_warn_pub =
            this->create_publisher<vision_interface::msg::RadarWarn>(
                "/hero_state", 10);
        debug_map_pub =
            this->create_publisher<sensor_msgs::msg::Image>("/map_2d", 10);
        radar2sentry_pub =
            this->create_publisher<vision_interface::msg::Radar2Sentry>(
                "/Radar2Sentry", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "DebugMap组件初始化完成！");
        match_info.self_color = self_color_override_;
    }

    void save_match_info(
        const std::shared_ptr<vision_interface::msg::MatchInfo> msg)
    {
        this->match_info = *msg;
        if (use_referee_identity_color_ && has_referee_identity_color_) {
            match_info.self_color = referee_identity_color_;
        } else if (use_match_info_color_) {
            match_info.self_color =
                normalize_self_color(msg->self_color, self_color_override_);
        } else {
            match_info.self_color = self_color_override_;
        }
    }

    void save_referee_identity(
        const std::shared_ptr<radar_decision_comm::msg::RefereeRobotIdentity> msg)
    {
        if (!use_referee_identity_color_ || !msg || !msg->valid) {
            return;
        }
        has_referee_identity_color_ = true;
        referee_identity_color_ =
            normalize_self_color(msg->self_color, self_color_override_);
        match_info.self_color = referee_identity_color_;
    }

    // 订阅最终融合UAV结果：poses[0]=blue UAV, poses[1]=red UAV
    // position=NaN 表示无效，不更新绘制点
    void uav_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;

        if (msg->poses.size() < 2) {
            // 仍然允许，但不更新
            return;
        }

        auto update_one = [&](int idx) {
            const auto& p = msg->poses[idx].position;

            // NaN 检查：只要任一分量非有限，就视作无效
            if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                !std::isfinite(p.z)) {
                // 不更新，保持上次值；为了避免残留旧点一直画，可将更新时间置0
                // 这里选择“置0”，保证无效时不绘制
                uav_update[idx] = 0.0;
                uav_point[idx] = cv::Point2f(0, 0);
                return;
            }

            cv::Point2f pt(static_cast<float>(p.x),
                           static_cast<float>(p.y));

            // 与车辆显示逻辑一致：self_color==0 时翻转坐标
            if (match_info.self_color == 0) {
                pt.x = 28.0f - pt.x;
                pt.y = 15.0f - pt.y;
            }

            // clamp
            pt.x = std::clamp(pt.x, 0.0f, 28.0f);
            pt.y = std::clamp(pt.y, 0.0f, 15.0f);

            uav_point[idx] = pt;
            uav_update[idx] = time;
        };

        update_one(0);
        update_one(1);
    }

    void show_map()
    {
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

        // =========================
        // 车辆绘制（保持原逻辑）
        // =========================
        for (int i = 0; i < 6; i++) {
            int number = i + 1;

            if ((blue_point[i].x != 0 || blue_point[i].y != 0) &&
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

            if ((red_point[i].x != 0 || red_point[i].y != 0) &&
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

        // =========================
        // UAV 绘制（改为只根据最终融合 /uav_positions）
        // =========================
        // poses[0]=蓝UAV
        if ((uav_point[0].x != 0.0f || uav_point[0].y != 0.0f) &&
            (time - uav_update[0] < 0.5)) {
            cv::Point2f point(clone_map.cols * uav_point[0].x / 28,
                              clone_map.rows * (15 - uav_point[0].y) / 15);
            cv::circle(clone_map, point, 12, cv::Scalar(200, 0, 0), -1);
            cv::putText(
                clone_map, "UAV-B", cv::Point(point.x - 20, point.y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }

        // poses[1]=红UAV
        if ((uav_point[1].x != 0.0f || uav_point[1].y != 0.0f) &&
            (time - uav_update[1] < 0.5)) {
            cv::Point2f point(clone_map.cols * uav_point[1].x / 28,
                              clone_map.rows * (15 - uav_point[1].y) / 15);
            cv::circle(clone_map, point, 12, cv::Scalar(0, 0, 200), -1);
            cv::putText(
                clone_map, "UAV-R", cv::Point(point.x - 20, point.y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }

        // 发布地图图像
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "map";

        try {
            sensor_msgs::msg::Image::SharedPtr map_msg =
                cv_bridge::CvImage(header, "bgr8", clone_map).toImageMsg();
            debug_map_pub->publish(*map_msg);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge转换失败：%s",
                         e.what());
        }

        cv::imshow("Debug Map", clone_map);
        cv::waitKey(1);
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

        for (int i = 0; i < 7; i++) {
            if (msg->blue_x[i] != 0 || msg->blue_y[i] != 0) {
                if (time - blue_time[i] > 5) {
                    cv::Point2f blue_raw(msg->blue_x[i], msg->blue_y[i]);
                    blue_point[i] = blue_raw;

                    if (!match_info.self_color) {
                        blue_point[i] = cv::Point2f(28 - msg->blue_x[i],
                                                    15 - msg->blue_y[i]);
                    }
                    blue_update[i] = time;
                }
            }

            if (msg->red_x[i] != 0 || msg->red_y[i] != 0) {
                if (time - red_time[i] > 5) {
                    cv::Point2f red_raw(msg->red_x[i], msg->red_y[i]);
                    red_point[i] = red_raw;

                    if (!match_info.self_color) {
                        red_point[i] = cv::Point2f(28 - msg->red_x[i],
                                                   15 - msg->red_y[i]);
                    }
                    red_update[i] = time;
                }
            }
        }

        show_map();
    }

    // 卡尔曼融合回调（原逻辑不变）
    void
    callback(const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;

        for (int i = 0; i < 7; i++) {
            if (msg->blue_x[i] != 0 || msg->blue_y[i] != 0) {
                blue_point[i] = cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                blue_time[i] = time;
                blue_update[i] = time;
            }
            if (msg->red_x[i] != 0 || msg->red_y[i] != 0) {
                red_point[i] = cv::Point2f(msg->red_x[i], msg->red_y[i]);
                red_time[i] = time;
                red_update[i] = time;
            }
        }

        show_map();

        // 英雄状态发布（原逻辑不变）
        vision_interface::msg::RadarWarn radar_warn;
        hero_count1 = std::max(0, hero_count1);
        hero_count2 = std::max(0, hero_count2);
        if (hero_count1 > 10) {
            radar_warn.hero_state = 1;
            radar_warn_pub->publish(radar_warn);
        } else if (hero_count2 > 10) {
            radar_warn.hero_state = 2;
            radar_warn_pub->publish(radar_warn);
        }

        if (match_info.self_color == 0) {
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
        } else if (match_info.self_color == 2) {
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

        // Radar2Sentry（原逻辑不变）
        vision_interface::msg::Radar2Sentry radar2sentry;
        if (match_info.self_color == 0) {
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
        } else if (match_info.self_color == 2) {
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
    // subscriptions
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_result_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        camera_detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;
    rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr
        referee_identity_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr uav_sub;

    // publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_map_pub;
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr
        radar_warn_pub;
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar2sentry_pub;

    // timestamps
    double blue_time[7];
    double red_time[7];

    bool   relax[7];
    double relax_time[7];

    double blue_update[7];
    double red_update[7];

    int hero_count1;
    int hero_count2;

    cv::Point2f blue_point[7];
    cv::Point2f red_point[7];

    // UAV fused positions from /uav_positions (poses[0]=blue, poses[1]=red)
    cv::Point2f uav_point[2];
    double      uav_update[2];

    vision_interface::msg::MatchInfo match_info;
    bool use_match_info_color_{true};
    bool use_referee_identity_color_{true};
    bool has_referee_identity_color_{false};
    int self_color_override_{0};
    int referee_identity_color_{0};
    cv::Mat                          map;
    int                              count;
};

}  // namespace debug_map

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(debug_map::DebugMap)
