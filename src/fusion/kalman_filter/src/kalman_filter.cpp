// 卡尔曼滤波节点实现文件
// 该文件实现了一个基于卡尔曼滤波的多目标跟踪系统，用于处理雷达点云数据

#include "kalman_filter.h"                    // 卡尔曼滤波器头文件
#include <pcl_conversions/pcl_conversions.h>  // PCL点云与ROS消息转换库
#include "filter_plus.h"                      // 扩展滤波器头文件
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace tdt_radar {  // 命名空间定义

int KalmanFilter::effective_self_color() const
{
    if (use_referee_identity_color_ && has_referee_identity_color_) {
        return referee_identity_color_;
    }
    if (use_match_info_color_) {
        return normalized_self_color(match_info.self_color);
    }
    return normalized_self_color(self_color_override_);
}

int KalmanFilter::normalized_self_color(int raw_color) const
{
    if (raw_color == 0) {
        return 0;
    }
    if (raw_color == 2 || raw_color == 1) {
        return 2;
    }
    return normalized_self_color(self_color_override_);
}

bool KalmanFilter::load_blind_guess_points(const std::string& config_path)
{
    guess_points_.clear();
    std::ifstream file(config_path);
    if (!file.good()) {
        RCLCPP_WARN(this->get_logger(),
                    "[BlindGuess] config not found: %s; blind guess disabled",
                    config_path.c_str());
        return false;
    }

    try {
        const YAML::Node root = YAML::LoadFile(config_path);
        const YAML::Node points = root["guess_points"];
        if (!points || !points.IsMap()) {
            RCLCPP_WARN(this->get_logger(),
                        "[BlindGuess] invalid config, missing guess_points: %s",
                        config_path.c_str());
            return false;
        }

        for (auto it = points.begin(); it != points.end(); ++it) {
            const std::string key = it->first.as<std::string>();
            const YAML::Node list = it->second;
            if (!list || !list.IsSequence()) {
                continue;
            }

            std::vector<pcl::PointXY> candidates;
            for (const auto& item : list) {
                if (!item.IsSequence() || item.size() < 2) {
                    continue;
                }
                pcl::PointXY point;
                point.x = item[0].as<float>();
                point.y = item[1].as<float>();
                if (std::isfinite(point.x) && std::isfinite(point.y)) {
                    candidates.push_back(point);
                }
            }
            if (!candidates.empty()) {
                guess_points_[key] = std::move(candidates);
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(),
                    "[BlindGuess] failed to load %s: %s; blind guess disabled",
                    config_path.c_str(), e.what());
        guess_points_.clear();
        return false;
    }

    if (guess_points_.empty()) {
        RCLCPP_WARN(this->get_logger(),
                    "[BlindGuess] no valid guess_points in %s; blind guess disabled",
                    config_path.c_str());
        return false;
    }

    RCLCPP_INFO(this->get_logger(),
                "[BlindGuess] loaded %zu robot guess point groups from %s",
                guess_points_.size(), config_path.c_str());
    return true;
}

std::string KalmanFilter::robot_key(const FixedTrackSlot& track) const
{
    const char color = track.color_id == 2 ? 'R' : 'B';
    return std::string(1, color) + std::to_string(track.number_id + 1);
}

pcl::PointXY KalmanFilter::mirror_point_if_needed(const pcl::PointXY& point,
                                                  bool mirror) const
{
    if (!mirror) {
        return point;
    }
    pcl::PointXY mirrored;
    mirrored.x = 28.0f - point.x;
    mirrored.y = 15.0f - point.y;
    return mirrored;
}

std::vector<pcl::PointXY> KalmanFilter::get_guess_points_for_track(
    const FixedTrackSlot& track,
    int effective_self_color) const
{
    (void)effective_self_color;

    const std::string key = robot_key(track);
    const auto direct = guess_points_.find(key);
    if (direct != guess_points_.end()) {
        return direct->second;
    }

    const char other_color = track.color_id == 2 ? 'B' : 'R';
    const std::string other_key =
        std::string(1, other_color) + std::to_string(track.number_id + 1);
    const auto other = guess_points_.find(other_key);
    if (other == guess_points_.end()) {
        return {};
    }

    std::vector<pcl::PointXY> mirrored;
    mirrored.reserve(other->second.size());
    for (const auto& point : other->second) {
        mirrored.push_back(mirror_point_if_needed(point, true));
    }
    return mirrored;
}

double KalmanFilter::lost_duration_ms(const FixedTrackSlot& track,
                                      const rclcpp::Time& time) const
{
    const rclcpp::Time base_time =
        track.last_match_update.nanoseconds() != 0 ? track.last_match_update
                                                   : track.last_state_change;
    if (base_time.nanoseconds() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::fabs((time - base_time).seconds()) * 1000.0;
}

bool KalmanFilter::compute_blind_guess(FixedTrackSlot& track,
                                       const rclcpp::Time& time,
                                       int effective_self_color)
{
    const auto candidates =
        get_guess_points_for_track(track, effective_self_color);
    if (candidates.empty()) {
        track.has_blind_guess_point = false;
        track.output_mode = "NO_OUTPUT";
        track.blind_guess_score = 0.0;
        return false;
    }

    const auto& params = track_sm_.params();
    const pcl::PointXY last_pos =
        track.has_last_good_point ? track.last_good_point : track.predict_point;
    const double speed =
        std::sqrt(track.velocity.x * track.velocity.x +
                  track.velocity.y * track.velocity.y);

    double best_score = -std::numeric_limits<double>::infinity();
    pcl::PointXY best_point = candidates.front();
    for (const auto& candidate : candidates) {
        const double dx = static_cast<double>(candidate.x - last_pos.x);
        const double dy = static_cast<double>(candidate.y - last_pos.y);
        const double distance = std::sqrt(dx * dx + dy * dy);
        const double distance_score =
            std::exp(-distance * params.blind_guess_d_factor);

        double cos_sim = 0.0;
        if (speed >= params.blind_guess_min_speed_mps &&
            distance > 1e-6) {
            cos_sim =
                (static_cast<double>(track.velocity.x) * dx +
                 static_cast<double>(track.velocity.y) * dy) /
                (speed * distance + 1e-8);
        }

        const double score =
            params.blind_guess_distance_weight * distance_score +
            params.blind_guess_direction_weight * cos_sim;
        if (score > best_score) {
            best_score = score;
            best_point = candidate;
        }
    }

    const std::string previous_mode = track.output_mode;
    track.blind_guess_point = best_point;
    track.has_blind_guess_point = true;
    track.blind_guess_update = time;
    track.blind_guess_score = best_score;
    track.output_mode = "BLIND_GUESS";

    if (previous_mode != "BLIND_GUESS") {
        const pcl::PointXY last_pos_for_log =
            track.has_last_good_point ? track.last_good_point : track.predict_point;
        RCLCPP_WARN(
            this->get_logger(),
            "[BlindGuess] key=%s state=LOST last=(%.2f,%.2f) vel=(%.2f,%.2f) guess=(%.2f,%.2f) score=%.3f",
            robot_key(track).c_str(), last_pos_for_log.x, last_pos_for_log.y,
            track.velocity.x, track.velocity.y, best_point.x, best_point.y,
            best_score);
    }
    return true;
}

void KalmanFilter::update_track_output_modes(const rclcpp::Time& time)
{
    const int self_color = effective_self_color();
    auto update_one = [&](FixedTrackSlot& track) {
        if (track.state == TrackState::INACTIVE) {
            track.output_mode = "NO_OUTPUT";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }
        if (track.state == TrackState::TENTATIVE) {
            track.output_mode = "NO_OUTPUT";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }
        if (track.state == TrackState::CONFIRMED) {
            track.output_mode = "REAL";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }
        if (track.state != TrackState::LOST || !track.semantic_locked) {
            track.output_mode = "NO_OUTPUT";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }

        const double lost_ms = lost_duration_ms(track, time);
        const auto& params = track_sm_.params();
        if (lost_ms <= static_cast<double>(params.kf_predict_hold_ms)) {
            track.output_mode = "KF_PREDICT";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }

        if (lost_ms > static_cast<double>(params.blind_guess_hold_ms)) {
            track.output_mode = "NO_OUTPUT";
            track.has_blind_guess_point = false;
            track.blind_guess_score = 0.0;
            return;
        }

        if (params.blind_guess_enabled &&
            lost_ms >= static_cast<double>(params.blind_guess_after_ms) &&
            (track.has_last_good_point || track.kf)) {
            if (compute_blind_guess(track, time, self_color)) {
                return;
            }
        }

        track.output_mode = "NO_OUTPUT";
        track.has_blind_guess_point = false;
        track.blind_guess_score = 0.0;
    };

    track_sm_.for_each_track(update_one);
}

// 卡尔曼滤波器构造函数
KalmanFilter::KalmanFilter(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("kalman_filter_node", node_options)  // ROS2节点初始化
{
    // 创建订阅器：订阅原始点云数据
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10,  // 话题名称和队列大小
        std::bind(&KalmanFilter::callback, this,
                  std::placeholders::_1));  // 回调函数绑定

    // 创建发布器：发布滤波后的点云
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_kalman", 10);

    // 创建发布器：发布雷达数据到哨兵
    radar_pub_ =
        this->create_publisher<vision_interface::msg::Radar2Sentry>(
            "/radar2sentry", 10);

    // 创建发布器：发布检测结果
    radar_detect_pub_ =
        this->create_publisher<vision_interface::msg::DetectResult>(
            "/kalman_detect", 10);

    track_status_pub_ =
        this->create_publisher<std_msgs::msg::String>(
            "/kalman_track_status", 10);

    // 创建订阅器：订阅检测结果（来自 Resolve 节点的坐标转换后数据）
    sub_detect_ =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "resolve_result",
            rclcpp::SensorDataQoS(),  // 使用传感器数据质量服务
            std::bind(&KalmanFilter::detect_callback, this,
                      std::placeholders::_1));

    // 创建订阅器：订阅激光雷达检测结果
    sub_lidar_ =
        this->create_subscription<vision_interface::msg::RadarWarn>(
            "/lidar_detect", 10,
            std::bind(&KalmanFilter::lidar_callback, this,
                      std::placeholders::_1));

    // 创建订阅器：订阅匹配信息
    sub_match_ =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "/match_info", 10,
            std::bind(&KalmanFilter::match_callback, this,
                      std::placeholders::_1));

    sub_referee_identity_ =
        this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
            "/radar/referee/robot_identity", 10,
            std::bind(&KalmanFilter::referee_identity_callback, this,
                      std::placeholders::_1));
    

    this->declare_parameter<int>("kf_available_timeout_ms", 300);
    this->declare_parameter<bool>("use_match_info_color", true);
    this->declare_parameter<int>("self_color_override", 2);
    this->declare_parameter<bool>("use_referee_identity_color", true);
    this->declare_parameter<bool>("semantic_tracking_enabled", true);
    this->declare_parameter<int>("semantic_history_size", 20);
    this->declare_parameter<int>("semantic_confirm_min_hits", 2);
    this->declare_parameter<int>("semantic_switch_min_hits", 3);
    this->declare_parameter<int>("fixed_track_confirm_hits", 2);
    this->declare_parameter<int>("pending_seed_confirm_hits", 2);
    this->declare_parameter<int>("semantic_vote_history_size", 6);
    this->declare_parameter<int>("tentative_miss_limit", 2);
    this->declare_parameter<int>("lost_miss_limit", 8);
    this->declare_parameter<int>("lost_prediction_publish_ms", 0);
    this->declare_parameter<int>("camera_publish_hold_ms", 120);
    this->declare_parameter<int>("lidar_publish_hold_ms", 100);
    this->declare_parameter<int>("recent_camera_constraint_ms", 500);
    this->declare_parameter<int>("frame_sync_ms", 50);
    this->declare_parameter<int>("box_iou_history_size", 6);
    this->declare_parameter<bool>("visual_bootstrap_on_first_frame", true);
    this->declare_parameter<bool>("tentative_accept_same_id_camera", true);
    this->declare_parameter<bool>("car_lidar_update_enabled", false);
    this->declare_parameter<double>("lidar_match_gate_m", 1.2);
    this->declare_parameter<double>("lost_lidar_match_gate_m", 2.0);
    this->declare_parameter<double>("camera_match_gate_m", 1.2);
    this->declare_parameter<double>("camera_dynamic_speed_gain", 1.0);
    this->declare_parameter<double>("vision_noise_margin_m", 0.8);
    this->declare_parameter<double>("box_consistency_gate_relax_m", 2.0);
    this->declare_parameter<double>("pending_seed_gate_m", 1.2);
    this->declare_parameter<double>("box_iou_confirm_threshold", 0.25);
    this->declare_parameter<double>("box_center_diag_ratio", 0.75);
    this->declare_parameter<bool>("publish_lost_prediction_to_decision", false);
    this->declare_parameter<bool>("blind_guess_enabled", true);
    this->declare_parameter<std::string>("blind_guess_config_path",
                                         "config/guess_pts.yaml");
    this->declare_parameter<int>("blind_guess_after_ms", 250);
    this->declare_parameter<int>("blind_guess_hold_ms", 1200);
    this->declare_parameter<int>("kf_predict_hold_ms", 150);
    this->declare_parameter<double>("max_predict_step_m", 0.8);
    this->declare_parameter<double>("max_predict_speed_mps", 5.0);
    this->declare_parameter<double>("blind_guess_distance_weight", 0.6);
    this->declare_parameter<double>("blind_guess_direction_weight", 0.4);
    this->declare_parameter<double>("blind_guess_d_factor", 0.15);
    this->declare_parameter<double>("blind_guess_min_speed_mps", 0.2);
    this->declare_parameter<bool>("blind_guess_publish_to_detect", false);
    this->declare_parameter<bool>("blind_guess_publish_to_sentry", false);
    kf_available_timeout_ms_ =
        this->get_parameter("kf_available_timeout_ms").as_int();
    use_match_info_color_ =
        this->get_parameter("use_match_info_color").as_bool();
    self_color_override_ =
        normalized_self_color(this->get_parameter("self_color_override").as_int());
    use_referee_identity_color_ =
        this->get_parameter("use_referee_identity_color").as_bool();
    auto& sm_params = track_sm_.params();
    sm_params.semantic_tracking_enabled =
        this->get_parameter("semantic_tracking_enabled").as_bool();
    sm_params.semantic_history_size =
        std::max(1, static_cast<int>(
                        this->get_parameter("semantic_history_size").as_int()));
    sm_params.semantic_confirm_min_hits =
        std::max(2, static_cast<int>(
                        this->get_parameter("semantic_confirm_min_hits").as_int()));
    sm_params.semantic_switch_min_hits =
        std::max(1, static_cast<int>(
                        this->get_parameter("semantic_switch_min_hits").as_int()));
    sm_params.fixed_track_confirm_hits =
        std::max(2, static_cast<int>(
                        this->get_parameter("fixed_track_confirm_hits").as_int()));
    sm_params.pending_seed_confirm_hits =
        std::max(2, static_cast<int>(
                        this->get_parameter("pending_seed_confirm_hits").as_int()));
    sm_params.semantic_vote_history_size =
        std::max(2, static_cast<int>(
                        this->get_parameter("semantic_vote_history_size").as_int()));
    sm_params.tentative_miss_limit =
        std::max(0, static_cast<int>(
                        this->get_parameter("tentative_miss_limit").as_int()));
    sm_params.lost_miss_limit =
        std::max(1, static_cast<int>(
                        this->get_parameter("lost_miss_limit").as_int()));
    sm_params.lost_prediction_publish_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("lost_prediction_publish_ms").as_int()));
    sm_params.camera_publish_hold_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("camera_publish_hold_ms").as_int()));
    sm_params.lidar_publish_hold_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("lidar_publish_hold_ms").as_int()));
    sm_params.recent_camera_constraint_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("recent_camera_constraint_ms").as_int()));
    sm_params.frame_sync_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("frame_sync_ms").as_int()));
    sm_params.box_iou_history_size =
        std::max(1, static_cast<int>(
                        this->get_parameter("box_iou_history_size").as_int()));
    sm_params.visual_bootstrap_on_first_frame =
        this->get_parameter("visual_bootstrap_on_first_frame").as_bool();
    sm_params.tentative_accept_same_id_camera =
        this->get_parameter("tentative_accept_same_id_camera").as_bool();
    sm_params.car_lidar_update_enabled =
        this->get_parameter("car_lidar_update_enabled").as_bool();
    sm_params.lidar_match_gate_m = static_cast<float>(
        std::max(0.1, this->get_parameter("lidar_match_gate_m").as_double()));
    sm_params.lost_lidar_match_gate_m = static_cast<float>(
        std::max(static_cast<double>(sm_params.lidar_match_gate_m),
                 this->get_parameter("lost_lidar_match_gate_m").as_double()));
    sm_params.camera_match_gate_m = static_cast<float>(
        std::max(0.1, this->get_parameter("camera_match_gate_m").as_double()));
    sm_params.camera_dynamic_speed_gain = static_cast<float>(
        std::max(0.0, this->get_parameter("camera_dynamic_speed_gain").as_double()));
    sm_params.vision_noise_margin_m = static_cast<float>(
        std::max(0.0, this->get_parameter("vision_noise_margin_m").as_double()));
    sm_params.box_consistency_gate_relax_m = static_cast<float>(
        std::max(0.0, this->get_parameter("box_consistency_gate_relax_m").as_double()));
    sm_params.pending_seed_gate_m = static_cast<float>(
        std::max(0.1, this->get_parameter("pending_seed_gate_m").as_double()));
    sm_params.box_iou_confirm_threshold = static_cast<float>(
        std::clamp(this->get_parameter("box_iou_confirm_threshold").as_double(),
                   0.0, 1.0));
    sm_params.box_center_diag_ratio = static_cast<float>(
        std::max(0.1, this->get_parameter("box_center_diag_ratio").as_double()));
    sm_params.publish_lost_prediction_to_decision =
        this->get_parameter("publish_lost_prediction_to_decision").as_bool();
    sm_params.blind_guess_enabled =
        this->get_parameter("blind_guess_enabled").as_bool();
    sm_params.blind_guess_after_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("blind_guess_after_ms").as_int()));
    sm_params.blind_guess_hold_ms =
        std::max(sm_params.blind_guess_after_ms,
                 static_cast<int>(
                     this->get_parameter("blind_guess_hold_ms").as_int()));
    sm_params.kf_predict_hold_ms =
        std::max(0, static_cast<int>(
                        this->get_parameter("kf_predict_hold_ms").as_int()));
    sm_params.max_predict_step_m = static_cast<float>(
        std::max(0.0, this->get_parameter("max_predict_step_m").as_double()));
    sm_params.max_predict_speed_mps = static_cast<float>(
        std::max(0.0, this->get_parameter("max_predict_speed_mps").as_double()));
    sm_params.blind_guess_distance_weight =
        std::max(0.0, this->get_parameter("blind_guess_distance_weight").as_double());
    sm_params.blind_guess_direction_weight =
        std::max(0.0, this->get_parameter("blind_guess_direction_weight").as_double());
    sm_params.blind_guess_d_factor =
        std::max(0.0, this->get_parameter("blind_guess_d_factor").as_double());
    sm_params.blind_guess_min_speed_mps =
        std::max(0.0, this->get_parameter("blind_guess_min_speed_mps").as_double());
    sm_params.blind_guess_publish_to_detect =
        this->get_parameter("blind_guess_publish_to_detect").as_bool();
    sm_params.blind_guess_publish_to_sentry =
        this->get_parameter("blind_guess_publish_to_sentry").as_bool();
    const std::string blind_guess_config_path =
        this->get_parameter("blind_guess_config_path").as_string();
    if (sm_params.blind_guess_enabled &&
        !load_blind_guess_points(blind_guess_config_path)) {
        sm_params.blind_guess_enabled = false;
    }
    referee_identity_color_ = self_color_override_;
    match_info.self_color = self_color_override_;
    // 输出节点启动日志信息
    RCLCPP_INFO(
        this->get_logger(),
        "Kalman_filter_Node has been started. use_match_info_color=%s self_color_override=%d semantic_tracking=%s history=%d confirm=%d switch=%d fixed_confirm=%d pending_seed=%d car_lidar=%s lidar_gate=%.2f lost_gate=%.2f camera_gate=%.2f dynamic_speed_gain=%.2f vision_margin=%.2f box_iou=%.2f hold(camera=%dms,lidar=%dms) frame_sync_ms=%d blind_guess=%s after=%dms hold=%dms kf_predict=%dms points=%zu",
        use_match_info_color_ ? "true" : "false",
        self_color_override_,
        sm_params.semantic_tracking_enabled ? "true" : "false",
        sm_params.semantic_history_size,
        sm_params.semantic_confirm_min_hits,
        sm_params.semantic_switch_min_hits,
        sm_params.fixed_track_confirm_hits,
        sm_params.pending_seed_confirm_hits,
        sm_params.car_lidar_update_enabled ? "enabled" : "disabled",
        sm_params.lidar_match_gate_m,
        sm_params.lost_lidar_match_gate_m,
        sm_params.camera_match_gate_m,
        sm_params.camera_dynamic_speed_gain,
        sm_params.vision_noise_margin_m,
        sm_params.box_iou_confirm_threshold,
        sm_params.camera_publish_hold_ms,
        sm_params.lidar_publish_hold_ms,
        sm_params.frame_sync_ms,
        sm_params.blind_guess_enabled ? "true" : "false",
        sm_params.blind_guess_after_ms,
        sm_params.blind_guess_hold_ms,
        sm_params.kf_predict_hold_ms,
        guess_points_.size());
}

// 匹配信息回调函数
void KalmanFilter::match_callback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    this->match_info = *msg;  // 保存匹配信息
    if (use_match_info_color_) {
        this->match_info.self_color = normalized_self_color(msg->self_color);
    } else {
        this->match_info.self_color = self_color_override_;
    }
    // RCLCPP_INFO(this->get_logger(), "Match_info_callback");  // 日志输出
}

void KalmanFilter::referee_identity_callback(
    const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
{
    if (!msg || !msg->valid) {
        return;
    }
    has_referee_identity_color_ = true;
    referee_identity_color_ = normalized_self_color(msg->self_color);
}

// 检测结果回调函数
void KalmanFilter::detect_callback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    static int callback_count = 0;
    callback_count++;

    rclcpp::Time time = msg->header.stamp;
    if (time.nanoseconds() == 0) {
        time = this->now();
    }

    struct CameraObservation {
        int color_id{1};
        int number_id{-1};
        pcl::PointXY point{};
        CarBox box{};
        bool matched{false};
        bool rejected{false};
    };

    auto make_box = [](float x1, float y1, float x2, float y2) {
        CarBox box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.valid = true;
        box.valid = FixedTrackStateMachine::valid_car_box(box);
        return box;
    };

    std::vector<CameraObservation> observations;
    observations.reserve(14);
    for (int i = 0; i < 7; i++) {
        pcl::PointXY red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        if (FixedTrackStateMachine::valid_measurement_point(red_point)) {
            CameraObservation observation;
            observation.color_id = 2;
            observation.number_id = i;
            observation.point = red_point;
            observation.box = make_box(msg->red_car_box_x1[i],
                                       msg->red_car_box_y1[i],
                                       msg->red_car_box_x2[i],
                                       msg->red_car_box_y2[i]);
            observations.push_back(observation);
        }

        pcl::PointXY blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        if (FixedTrackStateMachine::valid_measurement_point(blue_point)) {
            CameraObservation observation;
            observation.color_id = 0;
            observation.number_id = i;
            observation.point = blue_point;
            observation.box = make_box(msg->blue_car_box_x1[i],
                                       msg->blue_car_box_y1[i],
                                       msg->blue_car_box_x2[i],
                                       msg->blue_car_box_y2[i]);
            observations.push_back(observation);
        }
    }

    track_sm_.for_each_track([&](FixedTrackSlot& track) {
        track.matched_this_frame = false;
        track.camera_matched_this_frame = false;
        track.lidar_matched_this_frame = false;
        if (track.state != TrackState::INACTIVE && track.kf) {
            track_sm_.predict_track(track);
        }
    });

    for (auto& observation : observations) {
        FixedTrackSlot& track =
            track_sm_.get_track(observation.color_id, observation.number_id);
        if (!track.kf || track.state == TrackState::INACTIVE) {
            continue;
        }

        if (track_sm_.camera_observation_conflicts_with_confirmed_track(
                track, observation.point)) {
            track_sm_.reject_camera_observation(track, observation.point, time);
            observation.rejected = true;
            continue;
        }

        if (!track_sm_.camera_observation_matches_track(
                track, observation.point, observation.box, time)) {
            track_sm_.reject_camera_observation(track, observation.point, time);
            observation.rejected = true;
            continue;
        }

        track_sm_.update_track_camera(
            track, observation.point, observation.box, time);
        observation.matched = true;
    }

    for (auto& observation : observations) {
        if (observation.matched || observation.rejected) {
            continue;
        }

        FixedTrackSlot& track =
            track_sm_.get_track(observation.color_id, observation.number_id);
        if (track.state != TrackState::INACTIVE) {
            continue;
        }

        if (track_sm_.camera_observation_conflicts_with_confirmed_track(
                track, observation.point)) {
            track_sm_.reject_camera_observation(track, observation.point, time);
            observation.rejected = true;
            continue;
        }

        track_sm_.update_pending_seed(
            track, observation.point, observation.box, time);
        observation.matched = true;
    }

    track_sm_.for_each_track([&](FixedTrackSlot& track) {
        if (track.state == TrackState::INACTIVE || track.matched_this_frame) {
            return;
        }
        if (track_sm_.has_sync_lidar_update(track, time) &&
            !track_sm_.has_sync_miss(track, time, false)) {
            track.matched_this_frame = true;
            track.lidar_matched_this_frame = true;
            return;
        }
        track_sm_.update_track_miss(track, time, true);
    });

    publish_tracks(time, msg->header.stamp);
}
// 激光雷达检测回调函数
void KalmanFilter::lidar_callback(
    const vision_interface::msg::RadarWarn::SharedPtr msg)
{
    this->lidar_detect = *msg;  // 保存激光雷达检测结果
}

bool KalmanFilter::is_kf_available_now(const rclcpp::Time& now_time) const
{
    if (!kf_available_) {
        return false;
    }

    if (last_lidar_cluster_stamp_.nanoseconds() == 0) {
        return false;
    }

    const double dt_ms =
        std::fabs((now_time - last_lidar_cluster_stamp_).seconds()) * 1000.0;

    return dt_ms <= static_cast<double>(kf_available_timeout_ms_);
}

void KalmanFilter::publish_track_status(const rclcpp::Time& time)
{
    if (!track_status_pub_) {
        return;
    }

    std::ostringstream stream;
    stream << "stamp," << std::fixed << std::setprecision(6)
           << time.seconds() << '\n';

    auto append_track = [&](const FixedTrackSlot& track) {
        const bool publishable = track_sm_.should_publish_track(track, time);
        const bool current_matched =
            track.state != TrackState::INACTIVE && track.matched_this_frame;
        float x = std::numeric_limits<float>::quiet_NaN();
        float y = std::numeric_limits<float>::quiet_NaN();
        float vx = 0.0f;
        float vy = 0.0f;
        float blind_x = std::numeric_limits<float>::quiet_NaN();
        float blind_y = std::numeric_limits<float>::quiet_NaN();

        if (track.kf && track.state != TrackState::INACTIVE) {
            x = track.predict_point.x;
            y = track.predict_point.y;
            vx = track.velocity.x;
            vy = track.velocity.y;
            if (effective_self_color() == 0) {
                x = 28.0f - x;
                y = 15.0f - y;
                vx = -vx;
                vy = -vy;
            }
        }
        if (track.has_blind_guess_point) {
            blind_x = track.blind_guess_point.x;
            blind_y = track.blind_guess_point.y;
            if (effective_self_color() == 0) {
                blind_x = 28.0f - blind_x;
                blind_y = 15.0f - blind_y;
            }
        }

        stream << (track.color_id == 0 ? "B" : "R") << ','
               << track.number_id << ','
               << FixedTrackStateMachine::track_state_name(track.state) << ','
               << std::setprecision(3) << track.confidence << ','
               << track.hit_count << ','
               << track.semantic_hit_count << ','
               << track.lidar_hit_count << ','
               << track.miss_count << ','
               << x << ','
               << y << ','
               << vx << ','
               << vy << ','
               << blind_x << ','
               << blind_y << ','
               << (current_matched ? 1 : 0) << ','
               << (publishable ? 1 : 0) << ','
               << track.output_mode << ','
               << (track.has_blind_guess_point ? 1 : 0) << ','
               << track.blind_guess_score << '\n';
    };

    for (const auto& track : track_sm_.blue_tracks()) {
        append_track(track);
    }
    for (const auto& track : track_sm_.red_tracks()) {
        append_track(track);
    }

    std_msgs::msg::String status_msg;
    status_msg.data = stream.str();
    track_status_pub_->publish(status_msg);
}

void KalmanFilter::publish_tracks(const rclcpp::Time& time,
                                  const builtin_interfaces::msg::Time& stamp)
{
    update_track_output_modes(time);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    auto select_publish_point = [&](const FixedTrackSlot& track,
                                    pcl::PointXY& out_point) -> bool {
        if (!track_sm_.should_publish_track(track, time)) {
            return false;
        }
        if (track.output_mode == "BLIND_GUESS") {
            if (!track.has_blind_guess_point ||
                !track_sm_.params().blind_guess_publish_to_detect) {
                return false;
            }
            out_point = track.blind_guess_point;
            return true;
        }

        if (track.output_mode == "KF_PREDICT") {
            if (!track.has_last_good_point) {
                return false;
            }
            const float speed =
                std::sqrt(track.velocity.x * track.velocity.x +
                          track.velocity.y * track.velocity.y);
            const float step =
                FixedTrackStateMachine::point_distance(
                    track.predict_point, track.last_good_point);
            if (speed <= track_sm_.params().max_predict_speed_mps &&
                step <= track_sm_.params().max_predict_step_m) {
                out_point = track.predict_point;
                return true;
            }
            return false;
        }

        if (!track.has_last_good_point) {
            return false;
        }
        out_point = track.last_good_point;
        return true;
    };

    auto add_debug_point = [&](const FixedTrackSlot& track) {
        pcl::PointXY publish_point;
        if (!select_publish_point(track, publish_point)) {
            return;
        }

        pcl::PointXYZRGB point;
        point.x = publish_point.x;
        point.y = publish_point.y;
        point.z = 1.5f;

        if (track.state == TrackState::LOST) {
            point.r = 255;
            point.g = 220;
            point.b = 0;
        } else if (track.color_id == 0) {
            point.b = 255;
        } else if (track.color_id == 2) {
            point.r = 255;
        }
        cloud_filtered->points.push_back(point);
    };

    vision_interface::msg::DetectResult detect_msg;
    detect_msg.header.frame_id = "rm_frame";
    detect_msg.header.stamp = stamp;
    if (detect_msg.header.stamp.sec == 0 &&
        detect_msg.header.stamp.nanosec == 0) {
        detect_msg.header.stamp = time;
    }

    std::size_t active_count = 0;
    std::size_t published_count = 0;
    auto fill_detect = [&](const FixedTrackSlot& track) {
        if (track.state != TrackState::INACTIVE && track.kf) {
            active_count++;
            add_debug_point(track);
        }

        if (!track_sm_.should_publish_track(track, time)) {
            return;
        }

        pcl::PointXY publish_point;
        if (!select_publish_point(track, publish_point)) {
            return;
        }

        if (track.color_id == 0) {
            detect_msg.blue_x[track.number_id] = publish_point.x;
            detect_msg.blue_y[track.number_id] = publish_point.y;
            published_count++;
        } else if (track.color_id == 2) {
            detect_msg.red_x[track.number_id] = publish_point.x;
            detect_msg.red_y[track.number_id] = publish_point.y;
            published_count++;
        }
    };

    for (const auto& track : track_sm_.blue_tracks()) {
        fill_detect(track);
    }
    for (const auto& track : track_sm_.red_tracks()) {
        fill_detect(track);
    }

    if (effective_self_color() == 0) {
        for (int i = 0; i < 7; i++) {
            if (detect_msg.blue_x[i] != 0 && detect_msg.blue_y[i] != 0) {
                detect_msg.blue_x[i] = 28 - detect_msg.blue_x[i];
                detect_msg.blue_y[i] = 15 - detect_msg.blue_y[i];
            }
            if (detect_msg.red_x[i] != 0 && detect_msg.red_y[i] != 0) {
                detect_msg.red_x[i] = 28 - detect_msg.red_x[i];
                detect_msg.red_y[i] = 15 - detect_msg.red_y[i];
            }
        }
    }

    cloud_filtered->header.frame_id = "rm_frame";
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud_filtered, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = detect_msg.header.stamp;
    pub_->publish(output);

    bool has_kalman_data = false;
    std::string kalman_log = "[KalmanFilter发布] /kalman_detect: ";
    for (int i = 0; i < 7; i++) {
        if (detect_msg.blue_x[i] != 0 || detect_msg.blue_y[i] != 0) {
            kalman_log += "B" + std::to_string(i + 1) + "(";
            kalman_log +=
                std::to_string(static_cast<int>(detect_msg.blue_x[i])) + ",";
            kalman_log +=
                std::to_string(static_cast<int>(detect_msg.blue_y[i])) + ") ";
            has_kalman_data = true;
        }
        if (detect_msg.red_x[i] != 0 || detect_msg.red_y[i] != 0) {
            kalman_log += "R" + std::to_string(i + 1) + "(";
            kalman_log +=
                std::to_string(static_cast<int>(detect_msg.red_x[i])) + ",";
            kalman_log +=
                std::to_string(static_cast<int>(detect_msg.red_y[i])) + ") ";
            has_kalman_data = true;
        }
    }

    if (has_kalman_data) {
        RCLCPP_WARN(this->get_logger(), "%s [active=%zu published=%zu]",
                    kalman_log.c_str(), active_count, published_count);
    }

    radar_detect_pub_->publish(detect_msg);
    publish_track_status(time);

    vision_interface::msg::Radar2Sentry radar_msg;
    if (effective_self_color() == 0) {
        for (int i = 0; i < 6; i++) {
            const auto& track = track_sm_.red_tracks()[i];
            if (track.output_mode == "BLIND_GUESS" &&
                !track_sm_.params().blind_guess_publish_to_sentry) {
                continue;
            }
            radar_msg.radar_enemy_x[i] = detect_msg.red_x[i];
            radar_msg.radar_enemy_y[i] = detect_msg.red_y[i];
        }
        radar_pub_->publish(radar_msg);
    }
}

// 主回调函数：处理点云数据
void KalmanFilter::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rclcpp::Time time = msg->header.stamp;  // 获取消息时间戳
    if (time.nanoseconds() == 0) {
        time = this->now();
    }

    last_lidar_cluster_stamp_ = time;
    kf_available_ = true;

    std::vector<pcl::PointXY> lidar_points;
    if (!msg->data.empty() && msg->width != 0 && msg->height != 0) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        lidar_points.reserve(cloud->points.size());
        for (const auto& point : cloud->points) {
            pcl::PointXY point_xy;
            point_xy.x = point.x;
            point_xy.y = point.y;
            if (FixedTrackStateMachine::valid_measurement_point(point_xy)) {
                lidar_points.push_back(point_xy);
            }
        }
    }

    track_sm_.for_each_track([&](FixedTrackSlot& track) {
        if (track.state != TrackState::INACTIVE && track.kf) {
            track_sm_.predict_track(track);
            track.matched_this_frame = false;
            track.camera_matched_this_frame = false;
            track.lidar_matched_this_frame = false;
        }
    });

    struct LidarCandidate {
        int color_id;
        int number_id;
        int point_index;
        float cost;
    };

    std::vector<LidarCandidate> candidates;
    const bool allow_car_lidar_updates =
        track_sm_.params().car_lidar_update_enabled;
    for (int point_index = 0;
         point_index < static_cast<int>(lidar_points.size());
         ++point_index) {
        for (int i = 0; i < 7; ++i) {
            if (!allow_car_lidar_updates && i < 6) {
                continue;
            }

            const float blue_cost =
                track_sm_.lidar_match_cost(
                    track_sm_.blue_tracks()[i], lidar_points[point_index], time);
            if (std::isfinite(blue_cost)) {
                candidates.push_back({0, i, point_index, blue_cost});
            }

            const float red_cost =
                track_sm_.lidar_match_cost(
                    track_sm_.red_tracks()[i], lidar_points[point_index], time);
            if (std::isfinite(red_cost)) {
                candidates.push_back({2, i, point_index, red_cost});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const LidarCandidate& lhs, const LidarCandidate& rhs) {
                  return lhs.cost < rhs.cost;
              });

    std::vector<bool> point_used(lidar_points.size(), false);
    std::array<bool, 7> blue_used{};
    std::array<bool, 7> red_used{};

    auto track_used = [&](int color_id, int number_id) -> bool& {
        return color_id == 0 ? blue_used[number_id] : red_used[number_id];
    };

    for (const auto& candidate : candidates) {
        if (point_used[candidate.point_index] ||
            track_used(candidate.color_id, candidate.number_id)) {
            continue;
        }

        FixedTrackSlot& track =
            track_sm_.get_track(candidate.color_id, candidate.number_id);
        track_sm_.update_track_lidar(
            track, lidar_points[candidate.point_index], time);
        point_used[candidate.point_index] = true;
        track_used(candidate.color_id, candidate.number_id) = true;
    }

    track_sm_.for_each_track([&](FixedTrackSlot& track) {
        if (track.state == TrackState::INACTIVE || track.matched_this_frame) {
            return;
        }
        if (track_sm_.has_sync_camera_update(track, time) &&
            !track_sm_.has_sync_miss(track, time, true)) {
            track.matched_this_frame = true;
            track.camera_matched_this_frame = true;
            return;
        }
        track_sm_.update_track_miss(track, time, false);
    });

    publish_tracks(time, msg->header.stamp);
}
}  // namespace tdt_radar

// ROS2组件注册宏
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::KalmanFilter)
