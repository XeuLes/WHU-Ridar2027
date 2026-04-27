#include <opencv2/core/hal/interface.h>
#include <vector>
#include <opencv2/core/types.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <algorithm>
#include <map>
#include "opencv2/opencv.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <cmath>
#pragma once

class Kalman_filter_plus {
private:
    cv::KalmanFilter KF;

public:
    float Distance(const pcl::PointXY& a, const pcl::PointXY& b) const
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y));
}
    float get_time()
    {
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - timer);
        return duration.count() / 1000.0;
    }
    float                                        last_time = 0;
    std::chrono::steady_clock::time_point        timer;
    float                                        delete_time = 2.0;
    std::vector<std::pair<double, pcl::PointXY>> history;
    std::vector<std::pair<int, int>>             detect_history;
    int                                          max_history = 20;
    int                                          semantic_history_size = 20;
    int                                          semantic_confirm_min_hits = 1;
    int                                          semantic_switch_min_hits = 3;
    bool                                         semantic_tracking_enabled = true;
    bool                                         semantic_confirmed = false;
    int                                          semantic_color = 1;
    int                                          semantic_number = -1;

    pcl::PointXY predict_point;
    pcl::PointXY velocity_point;
    float        detect_r = 1;
    float        car_speed = 2;
    float        car_max_speed = 2.5;
    cv::Scalar   color;
    bool         has_updated = false;
cv::Mat      Q = cv::Mat::zeros(4, 4, CV_32F);

// 分开给雷达和相机的测量噪声
cv::Mat      R_lidar = cv::Mat::zeros(2, 2, CV_32F);
cv::Mat      R_camera = cv::Mat::zeros(2, 2, CV_32F);

float        dt_ = 0.1f;
float        sigma_q_x = 50.0f;
float        sigma_q_y = 50.0f;

// 雷达测量噪声
float        sigma_r_lidar_x = 0.1f;
float        sigma_r_lidar_y = 0.1f;

// 相机测量噪声：先设得比雷达更可信一点
float        sigma_r_camera_x = 0.05f;
float        sigma_r_camera_y = 0.05f;

// 相机匹配门限：建议比 detect_r 更紧一点
float        camera_match_r = 0.8f;

    Kalman_filter_plus(pcl::PointXY& input, rclcpp::Time time)
    {
        predict_point = input;
        history.push_back(std::make_pair(GetTimeByRosTime(time), input));
        timer = std::chrono::steady_clock::now();
        color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        int          stateSize = 4;
        int          measSize = 2;
        int          contrSize = 0;
        unsigned int type = CV_32F;
        KF.init(stateSize, measSize, contrSize, type);

        cv::Mat state = cv::Mat::zeros(stateSize, 1, type);
        cv::Mat meas(measSize, 1, type);
        meas.at<float>(0) = input.x;
        meas.at<float>(1) = input.y;

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(2) = meas.at<float>(1);
        KF.statePost = state;
        velocity_point.x = 0.0f;
        velocity_point.y = 0.0f;
        KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1,
                               0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);

        KF.measurementMatrix =
            (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);

        KF.processNoiseCov =
            (cv::Mat_<float>(4, 4) << sigma_q_x * pow(dt_, 3) / 3,
             sigma_q_x * pow(dt_, 2) / 2, 0, 0, sigma_q_x * pow(dt_, 2) / 2,
             sigma_q_x * pow(dt_, 1), 0, 0, 0, 0,
             sigma_q_y * pow(dt_, 3) / 3, sigma_q_y * pow(dt_, 2) / 2, 0, 0,
             sigma_q_y * pow(dt_, 2) / 2, sigma_q_y * pow(dt_, 1));

        R_lidar =
            (cv::Mat_<float>(2, 2) << sigma_r_lidar_x, 0,
                                    0, sigma_r_lidar_y);

        R_camera =
            (cv::Mat_<float>(2, 2) << sigma_r_camera_x, 0,
                                    0, sigma_r_camera_y);

        // 默认先用雷达测量噪声
        KF.measurementNoiseCov = R_lidar.clone();

        setIdentity(KF.errorCovPost, cv::Scalar::all(1));
        has_updated = true;
    }

    ~Kalman_filter_plus() {}

    pcl::PointXY velocity() const
    {
        return velocity_point;
    }

    static bool valid_semantic(int color_id, int number_id)
    {
        return (color_id == 0 || color_id == 2) && number_id >= 0 && number_id < 7;
    }

    int count_semantic(int color_id, int number_id) const
    {
        int count = 0;
        for (const auto& label : detect_history) {
            if (label.first == color_id && label.second == number_id) {
                count++;
            }
        }
        return count;
    }

    bool best_semantic(int& color_id, int& number_id, int& count) const
    {
        std::map<std::pair<int, int>, int> label_counts;
        for (const auto& label : detect_history) {
            if (valid_semantic(label.first, label.second)) {
                label_counts[label]++;
            }
        }
        if (label_counts.empty()) {
            return false;
        }

        count = 0;
        color_id = 1;
        number_id = -1;
        for (const auto& entry : label_counts) {
            if (entry.second > count) {
                count = entry.second;
                color_id = entry.first.first;
                number_id = entry.first.second;
            }
        }
        return true;
    }

    void update_semantic(int color_id, int number_id)
    {
        if (!valid_semantic(color_id, number_id)) {
            return;
        }

        detect_history.push_back(std::make_pair(color_id, number_id));
        const int max_semantic_history = std::max(1, semantic_history_size);
        while (static_cast<int>(detect_history.size()) > max_semantic_history) {
            detect_history.erase(detect_history.begin());
        }

        if (!semantic_tracking_enabled) {
            int best_color = 1;
            int best_number = -1;
            int best_count = 0;
            if (best_semantic(best_color, best_number, best_count)) {
                semantic_color = best_color;
                semantic_number = best_number;
                semantic_confirmed = true;
            }
            return;
        }

        const int observed_count = count_semantic(color_id, number_id);
        if (!semantic_confirmed) {
            if (observed_count >= std::max(1, semantic_confirm_min_hits)) {
                semantic_color = color_id;
                semantic_number = number_id;
                semantic_confirmed = true;
            }
            return;
        }

        if (semantic_color == color_id && semantic_number == number_id) {
            return;
        }

        if (observed_count >= std::max(1, semantic_switch_min_hits)) {
            semantic_color = color_id;
            semantic_number = number_id;
        }
    }

    int semantic_confidence_hits() const
    {
        if (!semantic_confirmed) {
            return 0;
        }
        return count_semantic(semantic_color, semantic_number);
    }

    int get_color() const
    {
        if (semantic_tracking_enabled) {
            return semantic_confirmed ? semantic_color : 1;
        }
        if (detect_history.size() == 0) {
            return 1;
        }
        int red = 0;
        int blue = 0;
        for (auto& color : detect_history) {
            if (color.first == 0) {
                blue++;
            } else {
                red++;
            }
        }
        if (red > blue) {
            return 2;
        } else {
            return 0;
        }
    }

    int get_number() const
    {
        if (semantic_tracking_enabled) {
            return semantic_confirmed ? semantic_number : 0;
        }
        int                color = get_color();
        std::map<int, int> number_map;
        for (auto& number : detect_history) {
            if (number.first == color) {
                number_map[number.second]++;
            }
        }
        int max_count = 0;
        int max_number = 0;
        for (auto& entry : number_map) {
            if (entry.second > max_count) {
                max_count = entry.second;
                max_number = entry.first;
            }
        }
        return max_number;
    }

    void update_lidar(const pcl::PointXY& input, rclcpp::Time time)
{
    cv::Mat meas = cv::Mat::zeros(2, 1, CV_32F);
    meas.at<float>(0) = input.x;
    meas.at<float>(1) = input.y;

    // 雷达更新时使用雷达测量噪声
    KF.measurementNoiseCov = R_lidar;
    KF.correct(meas);

    predict_point.x = KF.statePost.at<float>(0);
    predict_point.y = KF.statePost.at<float>(2);
    velocity_point.x = KF.statePost.at<float>(1);
    velocity_point.y = KF.statePost.at<float>(3);

    has_updated = true;
    last_time = 0;

    pcl::PointXY temp_point = input;
    history.push_back(std::make_pair(GetTimeByRosTime(time), temp_point));
    if (history.size() > max_history) {
        history.erase(history.begin());
    }
}

void update_camera(const pcl::PointXY& input, rclcpp::Time time)
{
    cv::Mat meas = cv::Mat::zeros(2, 1, CV_32F);
    meas.at<float>(0) = input.x;
    meas.at<float>(1) = input.y;

    // 相机更新时使用相机测量噪声
    KF.measurementNoiseCov = R_camera;
    KF.correct(meas);

    predict_point.x = KF.statePost.at<float>(0);
    predict_point.y = KF.statePost.at<float>(2);
    velocity_point.x = KF.statePost.at<float>(1);
    velocity_point.y = KF.statePost.at<float>(3);

    has_updated = true;
    last_time = 0;

    pcl::PointXY temp_point = input;
    history.push_back(std::make_pair(GetTimeByRosTime(time), temp_point));
    if (history.size() > max_history) {
        history.erase(history.begin());
    }
}

    void update_predict_point()
    {
        dt_ = get_time();
        timer = std::chrono::steady_clock::now();

        KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, dt_, 0,   0,
            0, 1,   0,   0,
            0, 0,   1, dt_,
            0, 0,   0,   1);

        KF.processNoiseCov =
            (cv::Mat_<float>(4, 4) <<
                sigma_q_x * std::pow(dt_, 3) / 3, sigma_q_x * std::pow(dt_, 2) / 2, 0, 0,
                sigma_q_x * std::pow(dt_, 2) / 2, sigma_q_x * dt_,                   0, 0,
                0, 0, sigma_q_y * std::pow(dt_, 3) / 3, sigma_q_y * std::pow(dt_, 2) / 2,
                0, 0, sigma_q_y * std::pow(dt_, 2) / 2, sigma_q_y * dt_);

        auto result = KF.predict();
        last_time += dt_;

        predict_point.x = result.at<float>(0);
        predict_point.y = result.at<float>(2);
        velocity_point.x = result.at<float>(1);
        velocity_point.y = result.at<float>(3);
    }

    bool match(const pcl::PointXY& input)
    {
        return Distance(predict_point, input) <
            (car_max_speed * dt_ + detect_r);
    }
    bool match_camera(const pcl::PointXY& input) const
    {
        return Distance(predict_point, input) < camera_match_r;
    }

    bool camera_match_score(const rclcpp::Time& time,
                            const pcl::PointXY& input,
                            float& score) const
    {
        const double TIME_THRESHOLD = 1.0;
        double input_time = GetTimeByRosTime(time);
        double differ_time = 1000.0;
        pcl::PointXY match_point = predict_point;

        // 保留你原来的时间最近历史点逻辑
        for (auto& point : history) {
            double differ = std::abs(point.first - input_time);
            if (differ < differ_time) {
                differ_time = differ;
                match_point = point.second;
            }
        }

        // 1) 时间门控
        if (differ_time > TIME_THRESHOLD) {
            return false;
        }

        // 2) 历史点距离门控（保留你原逻辑）
        if (Distance(match_point, input) >= detect_r) {
            return false;
        }

        // 3) 当前预测点门控（新增）
        if (!match_camera(input)) {
            return false;
        }

        score = Distance(predict_point, input);
        return true;
    }

    bool camera_match_and_update(const rclcpp::Time& time,
                                 const pcl::PointXY& input,
                                 int color_id,
                                 int number_id)
    {
        float score = 0.0f;
        if (!camera_match_score(time, input, score)) {
            return false;
        }

        // 4) 记录稳定语义，未知/无效语义不会清空已确认标签
        update_semantic(color_id, number_id);

        // 5) 用相机坐标修正KF
        update_camera(input, time);
        return true;
    }
    static double GetTimeByRosTime(const rclcpp::Time& ros_time)
    {
        return ros_time.nanoseconds() / 1e9;
    }
};
