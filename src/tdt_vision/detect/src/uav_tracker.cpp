#include "uav_tracker.h"
#include <algorithm>
#include <limits>
#include <iostream>

namespace tdt_radar {

// 静态成员初始化
int UAVTrack::next_id_ = 0;

// ==================== UAVTrack 实现 ====================

UAVTrack::UAVTrack(const cv::Rect& bbox, float confidence, const cv::Mat& feature)
    : track_id_(next_id_++), 
      current_bbox_(bbox),
      predicted_bbox_(bbox),
      state_(TrackState::Tentative),
      age_(1),
      hit_streak_(1),
      time_since_update_(0) {
    
    // 初始化卡尔曼滤波器
    initKalmanFilter(bbox);
    
    // 初始化外观特征
    if (!feature.empty()) {
        updateFeature(feature);
    }
}

void UAVTrack::initKalmanFilter(const cv::Rect& bbox) {
    // 状态向量：[x, y, w, h, vx, vy, vw, vh] (8维)
    // 测量向量：[x, y, w, h] (4维)
    kf_.init(8, 4, 0);
    
    // 状态转移矩阵 A (运动模型：匀速运动)
    kf_.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,  // x = x + vx
        0, 1, 0, 0, 0, 1, 0, 0,  // y = y + vy
        0, 0, 1, 0, 0, 0, 1, 0,  // w = w + vw
        0, 0, 0, 1, 0, 0, 0, 1,  // h = h + vh
        0, 0, 0, 0, 1, 0, 0, 0,  // vx = vx
        0, 0, 0, 0, 0, 1, 0, 0,  // vy = vy
        0, 0, 0, 0, 0, 0, 1, 0,  // vw = vw
        0, 0, 0, 0, 0, 0, 0, 1   // vh = vh
    );
    
    // 测量矩阵 H
    kf_.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;  // 测量x
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;  // 测量y
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;  // 测量w
    kf_.measurementMatrix.at<float>(3, 3) = 1.0f;  // 测量h
    
    // 过程噪声协方差矩阵 Q（位置噪声小，速度噪声大）
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    kf_.processNoiseCov.at<float>(4, 4) = 1e-1f;  // vx噪声
    kf_.processNoiseCov.at<float>(5, 5) = 1e-1f;  // vy噪声
    kf_.processNoiseCov.at<float>(6, 6) = 1e-1f;  // vw噪声
    kf_.processNoiseCov.at<float>(7, 7) = 1e-1f;  // vh噪声
    
    // 测量噪声协方差矩阵 R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // 后验误差协方差矩阵 P
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));
    
    // 初始化状态
    kf_.statePost.at<float>(0) = bbox.x + bbox.width / 2.0f;   // x
    kf_.statePost.at<float>(1) = bbox.y + bbox.height / 2.0f;  // y
    kf_.statePost.at<float>(2) = bbox.width;                    // w
    kf_.statePost.at<float>(3) = bbox.height;                   // h
    kf_.statePost.at<float>(4) = 0;  // vx
    kf_.statePost.at<float>(5) = 0;  // vy
    kf_.statePost.at<float>(6) = 0;  // vw
    kf_.statePost.at<float>(7) = 0;  // vh
}

cv::Rect UAVTrack::predict() {
    // 卡尔曼预测
    cv::Mat prediction = kf_.predict();
    predicted_bbox_ = stateToBox(prediction);
    
    age_++;
    time_since_update_++;
    
    // 如果丢失时间过长，标记为删除
    if (time_since_update_ > MAX_AGE) {
        state_ = TrackState::Deleted;
    }
    
    return predicted_bbox_;
}

void UAVTrack::update(const cv::Rect& bbox, float confidence, const cv::Mat& feature) {
    // 卡尔曼更新
    cv::Mat measurement = boxToMeasurement(bbox);
    kf_.correct(measurement);
    
    // 更新状态
    current_bbox_ = bbox;
    hit_streak_++;
    time_since_update_ = 0;
    
    // 更新外观特征
    if (!feature.empty()) {
        updateFeature(feature);
    }
    
    // 状态转换：Tentative -> Confirmed
    if (state_ == TrackState::Tentative && hit_streak_ >= N_INIT) {
        state_ = TrackState::Confirmed;
    }
}

void UAVTrack::markMissed() {
    // 如果是临时状态且未匹配，直接删除
    if (state_ == TrackState::Tentative) {
        state_ = TrackState::Deleted;
    }
    
    hit_streak_ = 0;
}

float UAVTrack::calculateIOU(const cv::Rect& bbox) const {
    int x1 = std::max(current_bbox_.x, bbox.x);
    int y1 = std::max(current_bbox_.y, bbox.y);
    int x2 = std::min(current_bbox_.x + current_bbox_.width, bbox.x + bbox.width);
    int y2 = std::min(current_bbox_.y + current_bbox_.height, bbox.y + bbox.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    int inter_area = (x2 - x1) * (y2 - y1);
    int union_area = current_bbox_.area() + bbox.area() - inter_area;
    
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

float UAVTrack::calculateFeatureDistance(const cv::Mat& feature) const {
    if (smooth_feature_.empty() || feature.empty()) {
        return 1.0f;  // 最大距离（无特征时）
    }
    
    // 余弦距离 = 1 - 余弦相似度
    double dot_product = smooth_feature_.dot(feature);
    double norm1 = cv::norm(smooth_feature_);
    double norm2 = cv::norm(feature);
    
    if (norm1 < 1e-6 || norm2 < 1e-6) return 1.0f;
    
    float cosine_similarity = static_cast<float>(dot_product / (norm1 * norm2));
    return 1.0f - cosine_similarity;
}

cv::Rect UAVTrack::stateToBox(const cv::Mat& state) const {
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float w = state.at<float>(2);
    float h = state.at<float>(3);
    
    int x = static_cast<int>(cx - w / 2.0f);
    int y = static_cast<int>(cy - h / 2.0f);
    
    return cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h));
}

cv::Mat UAVTrack::boxToMeasurement(const cv::Rect& bbox) const {
    cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
    measurement.at<float>(0) = bbox.x + bbox.width / 2.0f;   // cx
    measurement.at<float>(1) = bbox.y + bbox.height / 2.0f;  // cy
    measurement.at<float>(2) = bbox.width;
    measurement.at<float>(3) = bbox.height;
    return measurement;
}

void UAVTrack::updateFeature(const cv::Mat& feature) {
    // 归一化特征
    cv::Mat normalized_feature;
    cv::normalize(feature, normalized_feature);
    
    // 添加到历史
    feature_history_.push_back(normalized_feature.clone());
    if (feature_history_.size() > MAX_FEATURE_HISTORY) {
        feature_history_.pop_front();
    }
    
    // 计算平滑特征（简单平均）
    if (!feature_history_.empty()) {
        smooth_feature_ = cv::Mat::zeros(feature_history_[0].size(), feature_history_[0].type());
        for (const auto& f : feature_history_) {
            smooth_feature_ += f;
        }
        smooth_feature_ /= static_cast<float>(feature_history_.size());
        cv::normalize(smooth_feature_, smooth_feature_);
    }
}

// ==================== UAVDeepSORTTracker 实现 ====================

UAVDeepSORTTracker::UAVDeepSORTTracker(float max_iou_distance, 
                                       float max_feature_distance,
                                       float feature_weight)
    : max_iou_distance_(max_iou_distance),
      max_feature_distance_(max_feature_distance),
      feature_weight_(feature_weight) {}

std::vector<std::shared_ptr<UAVTrack>> UAVDeepSORTTracker::update(
    const std::vector<cv::Rect>& bboxes,
    const std::vector<float>& confidences,
    const std::vector<cv::Mat>& features) {
    
    // 1. 预测所有跟踪目标的位置
    for (auto& track : tracks_) {
        track->predict();
    }
    
    // 2. 数据关联（匹配检测框与跟踪目标）
    auto [matches, unmatched_tracks, unmatched_detections] = associate(bboxes, features);
    
    // 3. 更新匹配的跟踪目标
    for (const auto& [track_idx, det_idx] : matches) {
        cv::Mat feature = features.empty() ? cv::Mat() : features[det_idx];
        tracks_[track_idx]->update(bboxes[det_idx], confidences[det_idx], feature);
    }
    
    // 4. 标记未匹配的跟踪目标
    for (int track_idx : unmatched_tracks) {
        tracks_[track_idx]->markMissed();
    }
    
    // 5. 为未匹配的检测创建新跟踪目标
    for (int det_idx : unmatched_detections) {
        cv::Mat feature = features.empty() ? cv::Mat() : features[det_idx];
        auto new_track = std::make_shared<UAVTrack>(bboxes[det_idx], confidences[det_idx], feature);
        tracks_.push_back(new_track);
    }
    
    // 6. 删除已标记为删除的跟踪目标
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [](const std::shared_ptr<UAVTrack>& track) {
                return track->isDeleted();
            }),
        tracks_.end()
    );
    
    return getConfirmedTracks();
}

std::vector<std::shared_ptr<UAVTrack>> UAVDeepSORTTracker::getActiveTracks() const {
    return tracks_;
}

std::vector<std::shared_ptr<UAVTrack>> UAVDeepSORTTracker::getConfirmedTracks() const {
    std::vector<std::shared_ptr<UAVTrack>> confirmed;
    for (const auto& track : tracks_) {
        if (track->isConfirmed()) {
            confirmed.push_back(track);
        }
    }
    return confirmed;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
UAVDeepSORTTracker::associate(const std::vector<cv::Rect>& detections,
                               const std::vector<cv::Mat>& features) {
    
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
    
    if (tracks_.empty()) {
        // 没有跟踪目标，所有检测都是未匹配
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_detections.push_back(i);
        }
        return {matches, unmatched_tracks, unmatched_detections};
    }
    
    if (detections.empty()) {
        // 没有检测，所有跟踪都是未匹配
        for (size_t i = 0; i < tracks_.size(); i++) {
            unmatched_tracks.push_back(i);
        }
        return {matches, unmatched_tracks, unmatched_detections};
    }
    
    // 计算代价矩阵
    cv::Mat cost_matrix = calculateCostMatrix(detections, features);
    
    // 匈牙利算法求解
    matches = hungarianAlgorithm(cost_matrix);
    
    // 找出未匹配的跟踪目标
    std::vector<bool> track_matched(tracks_.size(), false);
    for (const auto& [track_idx, det_idx] : matches) {
        track_matched[track_idx] = true;
    }
    for (size_t i = 0; i < tracks_.size(); i++) {
        if (!track_matched[i]) {
            unmatched_tracks.push_back(i);
        }
    }
    
    // 找出未匹配的检测
    std::vector<bool> det_matched(detections.size(), false);
    for (const auto& [track_idx, det_idx] : matches) {
        det_matched[det_idx] = true;
    }
    for (size_t i = 0; i < detections.size(); i++) {
        if (!det_matched[i]) {
            unmatched_detections.push_back(i);
        }
    }
    
    return {matches, unmatched_tracks, unmatched_detections};
}

cv::Mat UAVDeepSORTTracker::calculateCostMatrix(const std::vector<cv::Rect>& detections,
                                                 const std::vector<cv::Mat>& features) {
    int num_tracks = tracks_.size();
    int num_dets = detections.size();
    
    cv::Mat cost_matrix = cv::Mat::zeros(num_tracks, num_dets, CV_32F);
    
    for (int i = 0; i < num_tracks; i++) {
        for (int j = 0; j < num_dets; j++) {
            // IOU距离（1 - IOU，越小越好）
            float iou = tracks_[i]->calculateIOU(detections[j]);
            float iou_distance = 1.0f - iou;
            
            // 特征距离（余弦距离）
            float feature_distance = 0.0f;
            if (!features.empty() && !features[j].empty()) {
                feature_distance = tracks_[i]->calculateFeatureDistance(features[j]);
            }
            
            // 组合距离（加权平均）
            float combined_distance;
            if (features.empty()) {
                // 无特征时仅用IOU
                combined_distance = iou_distance;
            } else {
                // 结合IOU和特征
                combined_distance = (1.0f - feature_weight_) * iou_distance + 
                                   feature_weight_ * feature_distance;
            }
            
            // 如果距离过大，设为无穷大（表示不可匹配）
            if (iou_distance > (1.0f - max_iou_distance_) || 
                feature_distance > max_feature_distance_) {
                combined_distance = 1e6f;  // 无穷大
            }
            
            cost_matrix.at<float>(i, j) = combined_distance;
        }
    }
    
    return cost_matrix;
}

std::vector<std::pair<int, int>> UAVDeepSORTTracker::hungarianAlgorithm(const cv::Mat& cost_matrix) {
    // 简化版匈牙利算法（贪心近似）
    // 对于UAV跟踪（数量少），贪心算法足够高效
    std::vector<std::pair<int, int>> matches;
    
    int num_tracks = cost_matrix.rows;
    int num_dets = cost_matrix.cols;
    
    std::vector<bool> track_used(num_tracks, false);
    std::vector<bool> det_used(num_dets, false);
    
    // 贪心匹配：每次选择代价最小的配对
    for (int iter = 0; iter < std::min(num_tracks, num_dets); iter++) {
        float min_cost = 1e6f;
        int best_track = -1;
        int best_det = -1;
        
        for (int i = 0; i < num_tracks; i++) {
            if (track_used[i]) continue;
            for (int j = 0; j < num_dets; j++) {
                if (det_used[j]) continue;
                float cost = cost_matrix.at<float>(i, j);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_track = i;
                    best_det = j;
                }
            }
        }
        
        // 如果找到有效匹配（代价不是无穷大）
        if (best_track >= 0 && min_cost < 1e5f) {
            matches.push_back({best_track, best_det});
            track_used[best_track] = true;
            det_used[best_det] = true;
        } else {
            break;  // 没有更多有效匹配
        }
    }
    
    return matches;
}

} // namespace tdt_radar
