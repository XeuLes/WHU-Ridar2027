#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <memory>
#include <deque>

namespace tdt_radar {

/**
 * @brief UAV跟踪状态
 */
enum class TrackState {
    Tentative,   // 临时状态（未确认）
    Confirmed,   // 确认状态
    Deleted      // 已删除
};

/**
 * @brief UAV跟踪目标类
 * @details 使用卡尔曼滤波预测运动，结合外观特征匹配（DeepSORT）
 */
class UAVTrack {
public:
    /**
     * @brief 构造函数
     * @param bbox 初始边界框 [x, y, w, h]
     * @param confidence 检测置信度
     * @param feature 外观特征向量（可选，用于DeepSORT）
     */
    UAVTrack(const cv::Rect& bbox, float confidence, const cv::Mat& feature = cv::Mat());
    
    /**
     * @brief 预测下一帧位置（卡尔曼滤波）
     * @return 预测的边界框
     */
    cv::Rect predict();
    
    /**
     * @brief 更新跟踪状态（卡尔曼滤波校正）
     * @param bbox 观测到的边界框
     * @param confidence 检测置信度
     * @param feature 外观特征向量
     */
    void update(const cv::Rect& bbox, float confidence, const cv::Mat& feature = cv::Mat());
    
    /**
     * @brief 标记为未匹配（连续丢失）
     */
    void markMissed();
    
    /**
     * @brief 获取当前边界框
     */
    cv::Rect getBox() const { return current_bbox_; }
    
    /**
     * @brief 获取预测边界框
     */
    cv::Rect getPredictedBox() const { return predicted_bbox_; }
    
    /**
     * @brief 获取跟踪ID
     */
    int getID() const { return track_id_; }
    
    /**
     * @brief 获取跟踪状态
     */
    TrackState getState() const { return state_; }
    
    /**
     * @brief 获取存活帧数
     */
    int getAge() const { return age_; }
    
    /**
     * @brief 获取连续匹配次数
     */
    int getHitStreak() const { return hit_streak_; }
    
    /**
     * @brief 获取自上次更新后的丢失帧数
     */
    int getTimeSinceUpdate() const { return time_since_update_; }
    
    /**
     * @brief 计算与检测框的IOU
     */
    float calculateIOU(const cv::Rect& bbox) const;
    
    /**
     * @brief 计算与特征向量的余弦相似度（外观距离）
     */
    float calculateFeatureDistance(const cv::Mat& feature) const;
    
    /**
     * @brief 是否为确认状态
     */
    bool isConfirmed() const { return state_ == TrackState::Confirmed; }
    
    /**
     * @brief 是否应该删除
     */
    bool isDeleted() const { return state_ == TrackState::Deleted; }

private:
    // 卡尔曼滤波器（状态：[x, y, w, h, vx, vy, vw, vh]）
    cv::KalmanFilter kf_;
    
    // 跟踪信息
    int track_id_;                    // 全局唯一ID
    static int next_id_;              // 下一个可用ID
    
    // 边界框
    cv::Rect current_bbox_;           // 当前边界框
    cv::Rect predicted_bbox_;         // 预测边界框
    
    // 外观特征（DeepSORT）
    std::deque<cv::Mat> feature_history_;  // 特征历史（用于平滑）
    cv::Mat smooth_feature_;               // 平滑后的特征
    static const int MAX_FEATURE_HISTORY = 100;  // 最大特征历史长度
    
    // 状态跟踪
    TrackState state_;                // 跟踪状态
    int age_;                         // 总存活帧数
    int hit_streak_;                  // 连续匹配次数
    int time_since_update_;           // 自上次更新的帧数
    
    // 参数
    static const int N_INIT = 3;      // 确认所需的连续匹配帧数
    static const int MAX_AGE = 30;    // 最大允许丢失帧数（卡尔曼滤波有预测能力，可设大一些）
    
    /**
     * @brief 初始化卡尔曼滤波器
     */
    void initKalmanFilter(const cv::Rect& bbox);
    
    /**
     * @brief 更新外观特征（平滑处理）
     */
    void updateFeature(const cv::Mat& feature);
    
    /**
     * @brief 从卡尔曼状态向量转换为边界框
     */
    cv::Rect stateToBox(const cv::Mat& state) const;
    
    /**
     * @brief 从边界框转换为卡尔曼测量向量
     */
    cv::Mat boxToMeasurement(const cv::Rect& bbox) const;
};

/**
 * @brief UAV DeepSORT跟踪器
 * @details 结合运动模型（卡尔曼滤波）和外观模型（特征匹配）
 */
class UAVDeepSORTTracker {
public:
    /**
     * @brief 构造函数
     * @param max_iou_distance IOU匹配阈值（默认0.7）
     * @param max_feature_distance 特征距离阈值（默认0.3）
     * @param feature_weight 特征权重（默认0.5，0表示仅用IOU，1表示仅用特征）
     */
    UAVDeepSORTTracker(float max_iou_distance = 0.7f, 
                       float max_feature_distance = 0.3f,
                       float feature_weight = 0.5f);
    
    /**
     * @brief 更新跟踪器
     * @param detections 检测结果列表 [bbox, confidence, feature]
     * @return 确认状态的跟踪目标列表
     */
    std::vector<std::shared_ptr<UAVTrack>> update(
        const std::vector<cv::Rect>& bboxes,
        const std::vector<float>& confidences,
        const std::vector<cv::Mat>& features = std::vector<cv::Mat>());
    
    /**
     * @brief 获取所有活跃跟踪目标
     */
    std::vector<std::shared_ptr<UAVTrack>> getActiveTracks() const;
    
    /**
     * @brief 获取确认状态的跟踪目标
     */
    std::vector<std::shared_ptr<UAVTrack>> getConfirmedTracks() const;

private:
    std::vector<std::shared_ptr<UAVTrack>> tracks_;  // 所有跟踪目标
    
    // 参数
    float max_iou_distance_;      // IOU匹配阈值
    float max_feature_distance_;  // 特征距离阈值
    float feature_weight_;        // 特征权重（0-1）
    
    /**
     * @brief 匈牙利算法匹配
     * @return [matched_tracks, matched_detections, unmatched_tracks, unmatched_detections]
     */
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    associate(const std::vector<cv::Rect>& detections,
              const std::vector<cv::Mat>& features);
    
    /**
     * @brief 计算代价矩阵（结合IOU和特征距离）
     */
    cv::Mat calculateCostMatrix(const std::vector<cv::Rect>& detections,
                                 const std::vector<cv::Mat>& features);
    
    /**
     * @brief 匈牙利算法求解最优匹配
     */
    std::vector<std::pair<int, int>> hungarianAlgorithm(const cv::Mat& cost_matrix);
};

} // namespace tdt_radar
