#pragma once

#include <array>
#include <deque>
#include <memory>
#include <string>

#include <pcl/point_types.h>
#include <rclcpp/time.hpp>

#include "filter_plus.h"

namespace tdt_radar {

enum class TrackState {
    INACTIVE,
    TENTATIVE,
    CONFIRMED,
    LOST,
};

struct SemanticVote {
    int color_id{1};
    int number_id{-1};
    pcl::PointXY point{};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool accepted{false};
};

struct CarBox {
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
    bool valid{false};
};

struct FixedTrackSlot {
    int color_id{1};
    int number_id{0};
    TrackState state{TrackState::INACTIVE};
    std::unique_ptr<Kalman_filter_plus> kf;
    int hit_count{0};
    int semantic_hit_count{0};
    int lidar_hit_count{0};
    int miss_count{0};
    float confidence{0.0f};
    bool semantic_locked{false};
    int pending_seed_count{0};
    pcl::PointXY pending_seed_point{};
    int semantic_outlier_count{0};
    std::deque<SemanticVote> semantic_votes;
    rclcpp::Time last_camera_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_lidar_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_match_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_miss_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_state_change{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_good_update{0, 0, RCL_ROS_TIME};
    rclcpp::Time blind_guess_update{0, 0, RCL_ROS_TIME};
    pcl::PointXY predict_point{};
    pcl::PointXY velocity{};
    pcl::PointXY last_good_point{};
    pcl::PointXY blind_guess_point{};
    CarBox last_car_box{};
    CarBox pending_car_box{};
    std::deque<float> box_iou_history;
    int visual_track_hit_count{0};
    std::string output_mode{"NO_OUTPUT"};
    double blind_guess_score{0.0};
    bool has_last_good_point{false};
    bool has_blind_guess_point{false};
    bool matched_this_frame{false};
    bool camera_matched_this_frame{false};
    bool lidar_matched_this_frame{false};
    bool last_miss_from_camera{false};
};

struct TrackStateMachineParams {
    bool semantic_tracking_enabled{true};
    int semantic_history_size{20};
    int semantic_confirm_min_hits{2};
    int semantic_switch_min_hits{3};
    int fixed_track_confirm_hits{2};
    int pending_seed_confirm_hits{2};
    int semantic_vote_history_size{6};
    int tentative_miss_limit{2};
    int lost_miss_limit{100};
    int lost_prediction_publish_ms{0};
    int camera_publish_hold_ms{120};
    int lidar_publish_hold_ms{100};
    int recent_camera_constraint_ms{500};
    int frame_sync_ms{50};
    int box_iou_history_size{6};
    bool visual_bootstrap_on_first_frame{true};
    bool tentative_accept_same_id_camera{true};
    bool car_lidar_update_enabled{false};
    float lidar_match_gate_m{1.2f};
    float lost_lidar_match_gate_m{2.0f};
    float camera_match_gate_m{1.2f};
    float camera_dynamic_speed_gain{1.0f};
    float vision_noise_margin_m{0.8f};
    float box_consistency_gate_relax_m{2.0f};
    float pending_seed_gate_m{1.2f};
    float box_iou_confirm_threshold{0.25f};
    float box_center_diag_ratio{0.75f};
    bool publish_lost_prediction_to_decision{false};
    bool blind_guess_enabled{true};
    int blind_guess_after_ms{250};
    int blind_guess_hold_ms{1200};
    int kf_predict_hold_ms{1500};
    float max_predict_step_m{7.5f};
    float max_predict_speed_mps{5.0f};
    double blind_guess_distance_weight{0.6};
    double blind_guess_direction_weight{0.4};
    double blind_guess_d_factor{0.15};
    double blind_guess_min_speed_mps{0.2};
    bool blind_guess_publish_to_detect{false};
    bool blind_guess_publish_to_sentry{false};
};

class FixedTrackStateMachine {
public:
    FixedTrackStateMachine();

    TrackStateMachineParams& params() { return params_; }
    const TrackStateMachineParams& params() const { return params_; }

    std::array<FixedTrackSlot, 7>& red_tracks() { return red_tracks_; }
    std::array<FixedTrackSlot, 7>& blue_tracks() { return blue_tracks_; }
    const std::array<FixedTrackSlot, 7>& red_tracks() const { return red_tracks_; }
    const std::array<FixedTrackSlot, 7>& blue_tracks() const { return blue_tracks_; }

    FixedTrackSlot& get_track(int color_id, int number_id);
    const FixedTrackSlot& get_track(int color_id, int number_id) const;

    template <typename Fn>
    void for_each_track(Fn&& fn)
    {
        for (auto& track : blue_tracks_) {
            fn(track);
        }
        for (auto& track : red_tracks_) {
            fn(track);
        }
    }

    template <typename Fn>
    void for_each_track(Fn&& fn) const
    {
        for (const auto& track : blue_tracks_) {
            fn(track);
        }
        for (const auto& track : red_tracks_) {
            fn(track);
        }
    }

    void reset_track(FixedTrackSlot& track, const rclcpp::Time& time);
    void seed_track(FixedTrackSlot& track,
                    const pcl::PointXY& point,
                    const rclcpp::Time& time);
    void predict_track(FixedTrackSlot& track);
    void sync_track_from_kf(FixedTrackSlot& track);
    void update_track_hit(FixedTrackSlot& track,
                          const rclcpp::Time& time,
                          bool from_camera);
    void update_track_miss(FixedTrackSlot& track,
                           const rclcpp::Time& time,
                           bool from_camera_frame);
    void update_track_camera(FixedTrackSlot& track,
                             const pcl::PointXY& point,
                             const CarBox& box,
                             const rclcpp::Time& time);
    void update_track_lidar(FixedTrackSlot& track,
                            const pcl::PointXY& point,
                            const rclcpp::Time& time);
    void update_pending_seed(FixedTrackSlot& track,
                             const pcl::PointXY& point,
                             const CarBox& box,
                             const rclcpp::Time& time);
    void reject_camera_observation(FixedTrackSlot& track,
                                   const pcl::PointXY& point,
                                   const rclcpp::Time& time);
    void record_semantic_vote(FixedTrackSlot& track,
                              const pcl::PointXY& point,
                              const rclcpp::Time& time,
                              bool accepted);
    int semantic_vote_hits(const FixedTrackSlot& track) const;
    int semantic_confirm_threshold() const;
    float lidar_match_cost(const FixedTrackSlot& track,
                           const pcl::PointXY& point,
                           const rclcpp::Time& time) const;
    bool camera_observation_conflicts_with_confirmed_track(
        const FixedTrackSlot& target,
        const pcl::PointXY& point) const;
    bool has_recent_camera_update(const FixedTrackSlot& track,
                                  const rclcpp::Time& time) const;
    bool has_sync_camera_update(const FixedTrackSlot& track,
                                const rclcpp::Time& time) const;
    bool has_sync_lidar_update(const FixedTrackSlot& track,
                               const rclcpp::Time& time) const;
    bool has_sync_miss(const FixedTrackSlot& track,
                       const rclcpp::Time& time,
                       bool from_camera_frame) const;
    bool car_box_consistent(const CarBox& previous,
                            const CarBox& current) const;
    bool camera_observation_matches_track(const FixedTrackSlot& track,
                                          const pcl::PointXY& point,
                                          const CarBox& box,
                                          const rclcpp::Time& time) const;
    bool should_publish_track(const FixedTrackSlot& track,
                              const rclcpp::Time& time) const;

    static bool valid_measurement_point(const pcl::PointXY& point);
    static float point_distance(const pcl::PointXY& lhs,
                                const pcl::PointXY& rhs);
    static bool valid_car_box(const CarBox& box);
    static float car_box_iou(const CarBox& lhs, const CarBox& rhs);
    static float car_box_center_distance(const CarBox& lhs,
                                         const CarBox& rhs);
    static float car_box_diag(const CarBox& box);
    static const char* track_state_name(TrackState state);

private:
    void configure_semantic_tracking(Kalman_filter_plus& kf) const;
    void initialize_fixed_tracks();

    TrackStateMachineParams params_;
    std::array<FixedTrackSlot, 7> red_tracks_;
    std::array<FixedTrackSlot, 7> blue_tracks_;
};

}  // namespace tdt_radar
