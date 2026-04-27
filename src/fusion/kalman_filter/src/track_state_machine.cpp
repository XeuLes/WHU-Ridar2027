#include "track_state_machine.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tdt_radar {

FixedTrackStateMachine::FixedTrackStateMachine()
{
    initialize_fixed_tracks();
}

void FixedTrackStateMachine::initialize_fixed_tracks()
{
    for (int i = 0; i < 7; ++i) {
        red_tracks_[i].color_id = 2;
        red_tracks_[i].number_id = i;
        blue_tracks_[i].color_id = 0;
        blue_tracks_[i].number_id = i;
    }
}

FixedTrackSlot& FixedTrackStateMachine::get_track(int color_id, int number_id)
{
    return color_id == 0 ? blue_tracks_[number_id] : red_tracks_[number_id];
}

const FixedTrackSlot& FixedTrackStateMachine::get_track(int color_id,
                                                        int number_id) const
{
    return color_id == 0 ? blue_tracks_[number_id] : red_tracks_[number_id];
}

void FixedTrackStateMachine::configure_semantic_tracking(
    Kalman_filter_plus& kf) const
{
    kf.semantic_tracking_enabled = params_.semantic_tracking_enabled;
    kf.semantic_history_size = params_.semantic_history_size;
    kf.semantic_confirm_min_hits = params_.semantic_confirm_min_hits;
    kf.semantic_switch_min_hits = params_.semantic_switch_min_hits;
}

const char* FixedTrackStateMachine::track_state_name(TrackState state)
{
    switch (state) {
    case TrackState::INACTIVE:
        return "INACTIVE";
    case TrackState::TENTATIVE:
        return "TENTATIVE";
    case TrackState::CONFIRMED:
        return "CONFIRMED";
    case TrackState::LOST:
        return "LOST";
    }
    return "UNKNOWN";
}

bool FixedTrackStateMachine::valid_measurement_point(const pcl::PointXY& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           (point.x != 0.0f || point.y != 0.0f);
}

float FixedTrackStateMachine::point_distance(const pcl::PointXY& lhs,
                                             const pcl::PointXY& rhs)
{
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool FixedTrackStateMachine::valid_car_box(const CarBox& box)
{
    return box.valid &&
           std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) &&
           box.x2 > box.x1 + 1.0f && box.y2 > box.y1 + 1.0f;
}

float FixedTrackStateMachine::car_box_iou(const CarBox& lhs,
                                          const CarBox& rhs)
{
    if (!valid_car_box(lhs) || !valid_car_box(rhs)) {
        return 0.0f;
    }

    const float ix1 = std::max(lhs.x1, rhs.x1);
    const float iy1 = std::max(lhs.y1, rhs.y1);
    const float ix2 = std::min(lhs.x2, rhs.x2);
    const float iy2 = std::min(lhs.y2, rhs.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float intersection = iw * ih;
    const float lhs_area = (lhs.x2 - lhs.x1) * (lhs.y2 - lhs.y1);
    const float rhs_area = (rhs.x2 - rhs.x1) * (rhs.y2 - rhs.y1);
    const float union_area = lhs_area + rhs_area - intersection;
    if (union_area <= 1.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

float FixedTrackStateMachine::car_box_center_distance(const CarBox& lhs,
                                                      const CarBox& rhs)
{
    if (!valid_car_box(lhs) || !valid_car_box(rhs)) {
        return std::numeric_limits<float>::infinity();
    }
    const float lcx = (lhs.x1 + lhs.x2) * 0.5f;
    const float lcy = (lhs.y1 + lhs.y2) * 0.5f;
    const float rcx = (rhs.x1 + rhs.x2) * 0.5f;
    const float rcy = (rhs.y1 + rhs.y2) * 0.5f;
    const float dx = lcx - rcx;
    const float dy = lcy - rcy;
    return std::sqrt(dx * dx + dy * dy);
}

float FixedTrackStateMachine::car_box_diag(const CarBox& box)
{
    if (!valid_car_box(box)) {
        return 0.0f;
    }
    const float w = box.x2 - box.x1;
    const float h = box.y2 - box.y1;
    return std::sqrt(w * w + h * h);
}

bool FixedTrackStateMachine::car_box_consistent(const CarBox& previous,
                                                const CarBox& current) const
{
    if (!valid_car_box(previous) || !valid_car_box(current)) {
        return false;
    }

    const float iou = car_box_iou(previous, current);
    if (iou >= params_.box_iou_confirm_threshold) {
        return true;
    }

    const float center_distance =
        car_box_center_distance(previous, current);
    const float diag =
        std::max(car_box_diag(previous), car_box_diag(current));
    return diag > 1.0f &&
           center_distance <= params_.box_center_diag_ratio * diag;
}

int FixedTrackStateMachine::semantic_confirm_threshold() const
{
    return std::max(2, std::min(params_.semantic_confirm_min_hits,
                                params_.fixed_track_confirm_hits));
}

void FixedTrackStateMachine::record_semantic_vote(FixedTrackSlot& track,
                                                  const pcl::PointXY& point,
                                                  const rclcpp::Time& time,
                                                  bool accepted)
{
    SemanticVote vote;
    vote.color_id = track.color_id;
    vote.number_id = track.number_id;
    vote.point = point;
    vote.stamp = time;
    vote.accepted = accepted;
    track.semantic_votes.push_back(vote);
    while (static_cast<int>(track.semantic_votes.size()) >
           params_.semantic_vote_history_size) {
        track.semantic_votes.pop_front();
    }
}

int FixedTrackStateMachine::semantic_vote_hits(
    const FixedTrackSlot& track) const
{
    int hits = 0;
    for (const auto& vote : track.semantic_votes) {
        if (vote.accepted &&
            vote.color_id == track.color_id &&
            vote.number_id == track.number_id) {
            hits++;
        }
    }
    return hits;
}

void FixedTrackStateMachine::reset_track(FixedTrackSlot& track,
                                         const rclcpp::Time& time)
{
    track.kf.reset();
    track.state = TrackState::INACTIVE;
    track.hit_count = 0;
    track.semantic_hit_count = 0;
    track.lidar_hit_count = 0;
    track.miss_count = 0;
    track.confidence = 0.0f;
    track.semantic_locked = false;
    track.pending_seed_count = 0;
    track.pending_seed_point.x = 0.0f;
    track.pending_seed_point.y = 0.0f;
    track.semantic_outlier_count = 0;
    track.semantic_votes.clear();
    track.last_camera_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.last_lidar_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.last_match_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.last_miss_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.last_state_change = time;
    track.last_good_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.blind_guess_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    track.predict_point.x = 0.0f;
    track.predict_point.y = 0.0f;
    track.velocity.x = 0.0f;
    track.velocity.y = 0.0f;
    track.last_good_point.x = 0.0f;
    track.last_good_point.y = 0.0f;
    track.blind_guess_point.x = 0.0f;
    track.blind_guess_point.y = 0.0f;
    track.last_car_box = CarBox{};
    track.pending_car_box = CarBox{};
    track.box_iou_history.clear();
    track.visual_track_hit_count = 0;
    track.output_mode = "NO_OUTPUT";
    track.blind_guess_score = 0.0;
    track.has_last_good_point = false;
    track.has_blind_guess_point = false;
    track.matched_this_frame = false;
    track.camera_matched_this_frame = false;
    track.lidar_matched_this_frame = false;
    track.last_miss_from_camera = false;
}

void FixedTrackStateMachine::seed_track(FixedTrackSlot& track,
                                        const pcl::PointXY& point,
                                        const rclcpp::Time& time)
{
    pcl::PointXY seed = point;
    track.kf = std::make_unique<Kalman_filter_plus>(seed, time);
    configure_semantic_tracking(*track.kf);
    track.kf->update_semantic(track.color_id, track.number_id);
    track.state = TrackState::TENTATIVE;
    track.output_mode = "NO_OUTPUT";
    track.has_blind_guess_point = false;
    track.blind_guess_score = 0.0;
    track.miss_count = 0;
    track.confidence = 0.0f;
    track.last_state_change = time;
    sync_track_from_kf(track);
    track.last_good_point = point;
    track.has_last_good_point = true;
    track.last_good_update = time;
}

void FixedTrackStateMachine::sync_track_from_kf(FixedTrackSlot& track)
{
    if (!track.kf) {
        return;
    }
    track.predict_point = track.kf->predict_point;
    track.velocity = track.kf->velocity();
}

void FixedTrackStateMachine::predict_track(FixedTrackSlot& track)
{
    if (!track.kf || track.state == TrackState::INACTIVE) {
        return;
    }
    track.kf->update_predict_point();
    sync_track_from_kf(track);
}

void FixedTrackStateMachine::update_track_hit(FixedTrackSlot& track,
                                              const rclcpp::Time& time,
                                              bool from_camera)
{
    track.matched_this_frame = true;
    if (from_camera) {
        track.camera_matched_this_frame = true;
    } else {
        track.lidar_matched_this_frame = true;
    }
    track.miss_count = 0;
    track.last_match_update = time;
    track.last_miss_update = rclcpp::Time(0, 0, RCL_ROS_TIME);
    const TrackState previous_state = track.state;

    if (from_camera) {
        track.semantic_hit_count++;
        track.hit_count = track.semantic_hit_count;
        track.last_camera_update = time;
        track.semantic_outlier_count = 0;
        track.output_mode = "REAL";
        if (track.semantic_hit_count >= semantic_confirm_threshold() ||
            semantic_vote_hits(track) >= semantic_confirm_threshold()) {
            track.semantic_locked = true;
        }
    } else {
        track.lidar_hit_count++;
        track.last_lidar_update = time;
        track.output_mode = "REAL";
    }
    track.has_blind_guess_point = false;
    track.blind_guess_score = 0.0;

    if (from_camera) {
        if (track.state == TrackState::INACTIVE) {
            track.state = TrackState::TENTATIVE;
        }
        if ((track.state == TrackState::TENTATIVE ||
             track.state == TrackState::LOST) &&
            track.semantic_locked) {
            track.state = TrackState::CONFIRMED;
        }
    } else {
        if (track.state == TrackState::INACTIVE) {
            track.state = TrackState::TENTATIVE;
        } else if (track.state == TrackState::LOST &&
                   track.semantic_locked) {
            track.state = TrackState::CONFIRMED;
        }
    }

    if (track.state == TrackState::CONFIRMED) {
        track.confidence = 1.0f;
    } else {
        track.confidence =
            std::min(0.9f,
                     static_cast<float>(track.semantic_hit_count) /
                         static_cast<float>(semantic_confirm_threshold()));
    }

    if (track.state != previous_state) {
        track.last_state_change = time;
    }
}

void FixedTrackStateMachine::update_track_miss(FixedTrackSlot& track,
                                               const rclcpp::Time& time,
                                               bool from_camera_frame)
{
    if (track.state == TrackState::INACTIVE) {
        return;
    }

    track.matched_this_frame = false;
    track.camera_matched_this_frame = false;
    track.lidar_matched_this_frame = false;

    if (track.last_miss_update.nanoseconds() != 0 &&
        track.last_miss_from_camera != from_camera_frame) {
        const double dt_ms =
            std::fabs((time - track.last_miss_update).seconds()) * 1000.0;
        if (dt_ms <= static_cast<double>(params_.frame_sync_ms)) {
            return;
        }
    }

    track.last_miss_update = time;
    track.last_miss_from_camera = from_camera_frame;
    track.miss_count++;

    const TrackState previous_state = track.state;
    if (track.state == TrackState::TENTATIVE) {
        track.hit_count = 0;
        track.semantic_hit_count = 0;
        track.confidence = 0.0f;
        if (track.miss_count > params_.tentative_miss_limit) {
            reset_track(track, time);
            return;
        }
    } else if (track.state == TrackState::CONFIRMED) {
        track.state = TrackState::LOST;
        track.output_mode = "KF_PREDICT";
        track.confidence = 0.5f;
    } else if (track.state == TrackState::LOST) {
        track.confidence = std::max(0.1f, 0.5f - 0.05f * track.miss_count);
        if (track.miss_count > params_.lost_miss_limit) {
            reset_track(track, time);
            return;
        }
    }

    if (track.state != previous_state) {
        track.last_state_change = time;
    }
}

void FixedTrackStateMachine::update_track_camera(FixedTrackSlot& track,
                                                 const pcl::PointXY& point,
                                                 const CarBox& box,
                                                 const rclcpp::Time& time)
{
    if (!valid_measurement_point(point)) {
        return;
    }

    if (!track.kf || track.state == TrackState::INACTIVE) {
        return;
    }

    track.kf->update_semantic(track.color_id, track.number_id);
    record_semantic_vote(track, point, time, true);
    if (valid_car_box(box)) {
        if (valid_car_box(track.last_car_box)) {
            track.box_iou_history.push_back(car_box_iou(track.last_car_box, box));
            while (static_cast<int>(track.box_iou_history.size()) >
                   params_.box_iou_history_size) {
                track.box_iou_history.pop_front();
            }
            if (car_box_consistent(track.last_car_box, box)) {
                track.visual_track_hit_count++;
            } else {
                track.visual_track_hit_count = 1;
            }
        } else {
            track.visual_track_hit_count = std::max(1, track.visual_track_hit_count);
        }
        track.last_car_box = box;
    }
    track.kf->update_camera(point, time);
    sync_track_from_kf(track);
    track.last_good_point = point;
    track.has_last_good_point = true;
    track.last_good_update = time;
    update_track_hit(track, time, true);
}

void FixedTrackStateMachine::update_track_lidar(FixedTrackSlot& track,
                                                const pcl::PointXY& point,
                                                const rclcpp::Time& time)
{
    if (!valid_measurement_point(point)) {
        return;
    }

    if (!track.kf) {
        return;
    }

    track.kf->update_lidar(point, time);
    sync_track_from_kf(track);
    track.last_good_point = point;
    track.has_last_good_point = true;
    track.last_good_update = time;
    update_track_hit(track, time, false);
}

void FixedTrackStateMachine::reject_camera_observation(
    FixedTrackSlot& track,
    const pcl::PointXY& point,
    const rclcpp::Time& time)
{
    track.semantic_outlier_count++;
    if (!track.semantic_locked) {
        track.semantic_hit_count = 0;
        track.hit_count = 0;
    }
    if (track.state == TrackState::INACTIVE) {
        track.pending_seed_count = 0;
        track.pending_seed_point.x = 0.0f;
        track.pending_seed_point.y = 0.0f;
    }
    record_semantic_vote(track, point, time, false);
}

void FixedTrackStateMachine::update_pending_seed(FixedTrackSlot& track,
                                                 const pcl::PointXY& point,
                                                 const CarBox& box,
                                                 const rclcpp::Time& time)
{
    if (!valid_measurement_point(point) ||
        track.state != TrackState::INACTIVE) {
        return;
    }

    bool keep_pending = track.pending_seed_count > 0;
    if (keep_pending && track.last_camera_update.nanoseconds() != 0) {
        const double dt_ms =
            std::fabs((time - track.last_camera_update).seconds()) * 1000.0;
        if (dt_ms > static_cast<double>(params_.recent_camera_constraint_ms)) {
            keep_pending = false;
        }
    }

    const bool box_matches_pending =
        car_box_consistent(track.pending_car_box, box);
    const bool point_matches_pending =
        keep_pending &&
        point_distance(track.pending_seed_point, point) <=
            params_.pending_seed_gate_m;

    if (keep_pending && (box_matches_pending || point_matches_pending)) {
        const float old_count =
            static_cast<float>(std::max(1, track.pending_seed_count));
        track.pending_seed_point.x =
            (track.pending_seed_point.x * old_count + point.x) /
            (old_count + 1.0f);
        track.pending_seed_point.y =
            (track.pending_seed_point.y * old_count + point.y) /
            (old_count + 1.0f);
        track.pending_seed_count++;
    } else {
        track.pending_seed_point = point;
        track.pending_seed_count = 1;
    }
    if (valid_car_box(box)) {
        if (valid_car_box(track.pending_car_box)) {
            track.box_iou_history.push_back(car_box_iou(track.pending_car_box, box));
            while (static_cast<int>(track.box_iou_history.size()) >
                   params_.box_iou_history_size) {
                track.box_iou_history.pop_front();
            }
        }
        track.pending_car_box = box;
        track.last_car_box = box;
    }

    track.last_camera_update = time;
    track.matched_this_frame = true;
    track.camera_matched_this_frame = true;
    track.visual_track_hit_count = track.pending_seed_count;
    track.semantic_hit_count = track.pending_seed_count;
    track.hit_count = track.semantic_hit_count;
    record_semantic_vote(track, point, time, true);

    if (!track.kf && params_.visual_bootstrap_on_first_frame) {
        seed_track(track, point, time);
        track.pending_seed_count = std::max(1, track.pending_seed_count);
        track.pending_seed_point = point;
        track.semantic_hit_count = track.pending_seed_count;
        track.hit_count = track.semantic_hit_count;
        track.visual_track_hit_count = track.pending_seed_count;
        track.last_camera_update = time;
        track.last_match_update = time;
        track.matched_this_frame = true;
        track.camera_matched_this_frame = true;
    }

    if (track.pending_seed_count < params_.pending_seed_confirm_hits) {
        track.confidence =
            static_cast<float>(track.pending_seed_count) /
            static_cast<float>(std::max(1, params_.pending_seed_confirm_hits));
        return;
    }

    const int confirmed_pending_count = track.pending_seed_count;
    seed_track(track, point, time);
    track.pending_seed_count = confirmed_pending_count;
    track.pending_seed_point = point;
    track.semantic_hit_count = confirmed_pending_count;
    track.hit_count = track.semantic_hit_count;
    track.visual_track_hit_count = confirmed_pending_count;
    track.last_camera_update = time;
    track.last_match_update = time;
    track.semantic_locked =
        track.semantic_hit_count >= semantic_confirm_threshold() ||
        semantic_vote_hits(track) >= semantic_confirm_threshold();

    if (track.semantic_locked) {
        track.state = TrackState::CONFIRMED;
        track.confidence = 1.0f;
    } else {
        track.state = TrackState::TENTATIVE;
        track.confidence =
            std::min(0.9f,
                     static_cast<float>(track.semantic_hit_count) /
                         static_cast<float>(semantic_confirm_threshold()));
    }
}

bool FixedTrackStateMachine::camera_observation_matches_track(
    const FixedTrackSlot& track,
    const pcl::PointXY& point,
    const CarBox& box,
    const rclcpp::Time& time) const
{
    if (!track.kf || track.state == TrackState::INACTIVE) {
        return false;
    }

    const pcl::PointXY& reference = track.predict_point;

    double dt_s = 0.0;
    if (track.last_match_update.nanoseconds() != 0) {
        dt_s = std::fabs((time - track.last_match_update).seconds());
    } else if (track.last_camera_update.nanoseconds() != 0) {
        dt_s = std::fabs((time - track.last_camera_update).seconds());
    }
    dt_s = std::min(dt_s, 1.0);

    const float speed =
        std::sqrt(track.velocity.x * track.velocity.x +
                  track.velocity.y * track.velocity.y);
    const float dynamic_gate =
        params_.camera_match_gate_m +
        params_.camera_dynamic_speed_gain * speed * static_cast<float>(dt_s) +
        params_.vision_noise_margin_m;

    const bool box_consistent =
        car_box_consistent(track.last_car_box, box) ||
        car_box_consistent(track.pending_car_box, box);

    if (params_.tentative_accept_same_id_camera &&
        track.state == TrackState::TENTATIVE &&
        !track.semantic_locked) {
        return true;
    }

    if (box_consistent &&
        (track.state == TrackState::TENTATIVE || !track.semantic_locked)) {
        return true;
    }

    const float allowed_gate =
        box_consistent ? dynamic_gate + params_.box_consistency_gate_relax_m
                       : dynamic_gate;

    return point_distance(reference, point) <= allowed_gate;
}

bool FixedTrackStateMachine::has_recent_camera_update(
    const FixedTrackSlot& track,
    const rclcpp::Time& time) const
{
    if (track.last_camera_update.nanoseconds() == 0) {
        return false;
    }
    const double dt_ms =
        std::fabs((time - track.last_camera_update).seconds()) * 1000.0;
    return dt_ms <= static_cast<double>(params_.recent_camera_constraint_ms);
}

bool FixedTrackStateMachine::has_sync_camera_update(
    const FixedTrackSlot& track,
    const rclcpp::Time& time) const
{
    if (track.last_camera_update.nanoseconds() == 0) {
        return false;
    }
    const double dt_ms =
        std::fabs((time - track.last_camera_update).seconds()) * 1000.0;
    return dt_ms <= static_cast<double>(params_.frame_sync_ms);
}

bool FixedTrackStateMachine::has_sync_lidar_update(
    const FixedTrackSlot& track,
    const rclcpp::Time& time) const
{
    if (track.last_lidar_update.nanoseconds() == 0) {
        return false;
    }
    const double dt_ms =
        std::fabs((time - track.last_lidar_update).seconds()) * 1000.0;
    return dt_ms <= static_cast<double>(params_.frame_sync_ms);
}

bool FixedTrackStateMachine::has_sync_miss(
    const FixedTrackSlot& track,
    const rclcpp::Time& time,
    bool from_camera_frame) const
{
    if (track.last_miss_update.nanoseconds() == 0 ||
        track.last_miss_from_camera != from_camera_frame) {
        return false;
    }
    const double dt_ms =
        std::fabs((time - track.last_miss_update).seconds()) * 1000.0;
    return dt_ms <= static_cast<double>(params_.frame_sync_ms);
}

float FixedTrackStateMachine::lidar_match_cost(
    const FixedTrackSlot& track,
    const pcl::PointXY& point,
    const rclcpp::Time& time) const
{
    if (!track.kf || track.state == TrackState::INACTIVE) {
        return std::numeric_limits<float>::infinity();
    }

    pcl::PointXY reference = track.predict_point;
    float gate = params_.lidar_match_gate_m;
    if (track.state == TrackState::LOST) {
        gate = params_.lost_lidar_match_gate_m;
        if (track.has_last_good_point) {
            reference = track.last_good_point;
        }
    }

    const float dx = point.x - reference.x;
    const float dy = point.y - reference.y;
    const float distance = std::sqrt(dx * dx + dy * dy);
    if (distance > gate) {
        return std::numeric_limits<float>::infinity();
    }

    float cost = distance;
    const float speed =
        std::sqrt(track.velocity.x * track.velocity.x +
                  track.velocity.y * track.velocity.y);
    if (track.state != TrackState::LOST && speed > 0.05f &&
        distance > 0.05f) {
        const float direction_consistency =
            (track.velocity.x * dx + track.velocity.y * dy) /
            (speed * distance);
        cost += 0.2f * (1.0f - direction_consistency);
    }

    if (track.state == TrackState::TENTATIVE) {
        cost += 0.15f;
    } else if (track.state == TrackState::LOST) {
        cost += 0.08f * static_cast<float>(track.miss_count);
    }

    if (has_recent_camera_update(track, time)) {
        cost -= 0.1f;
    }
    return cost;
}

bool FixedTrackStateMachine::camera_observation_conflicts_with_confirmed_track(
    const FixedTrackSlot& target,
    const pcl::PointXY& point) const
{
    const auto& tracks = target.color_id == 0 ? blue_tracks_ : red_tracks_;
    const float target_distance =
        (target.kf && target.state != TrackState::INACTIVE)
            ? point_distance(target.predict_point, point)
            : std::numeric_limits<float>::infinity();

    for (const auto& other : tracks) {
        if (other.number_id == target.number_id ||
            !other.kf ||
            !other.semantic_locked ||
            !(other.state == TrackState::CONFIRMED ||
              other.state == TrackState::LOST)) {
            continue;
        }

        const float other_distance = point_distance(other.predict_point, point);
        if (other_distance < params_.camera_match_gate_m &&
            other_distance + 0.15f < target_distance) {
            return true;
        }
    }
    return false;
}

bool FixedTrackStateMachine::should_publish_track(
    const FixedTrackSlot& track,
    const rclcpp::Time& time) const
{
    if (!track.kf) {
        return false;
    }
    if (track.state == TrackState::CONFIRMED) {
        return track.semantic_locked;
    }
    if (track.state != TrackState::LOST || !track.semantic_locked) {
        return false;
    }
    const rclcpp::Time base_time =
        track.last_match_update.nanoseconds() != 0 ? track.last_match_update
                                                   : track.last_state_change;
    if (base_time.nanoseconds() == 0) {
        return false;
    }
    const double lost_ms =
        std::fabs((time - base_time).seconds()) * 1000.0;
    if (track.output_mode == "KF_PREDICT") {
        return lost_ms <= static_cast<double>(params_.kf_predict_hold_ms);
    }
    if (track.output_mode == "BLIND_GUESS") {
        return params_.blind_guess_publish_to_detect &&
               lost_ms <= static_cast<double>(params_.blind_guess_hold_ms);
    }
    return false;
}

}  // namespace tdt_radar
