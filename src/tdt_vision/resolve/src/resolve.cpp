#include "resolve.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace tdt_radar {

static inline bool is_stamp_zero(const builtin_interfaces::msg::Time& t)
{
    return (t.sec == 0 && t.nanosec == 0);
}

Resolve::Resolve(const rclcpp::NodeOptions& node_options)
    : Node("radar_resolve_node", node_options)
{
    parser_ = new parser();
    minimap = cv::imread("config/RM2025.png");

    point_sub = this->create_subscription<geometry_msgs::msg::Vector3>(
        "camera_point2D", rclcpp::SensorDataQoS(),
        std::bind(&Resolve::callback, this, std::placeholders::_1));

    match_info_sub =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "match_info", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::MatchInfoCallback, this,
                      std::placeholders::_1));

    detect_sub =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "detect_result", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::DetectCallback, this,
                      std::placeholders::_1));

    pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera_point3D", rclcpp::SensorDataQoS());

    pub_radar = this->create_publisher<vision_interface::msg::DetectResult>(
        "resolve_result", rclcpp::SensorDataQoS());

    // UAV topics
    pub_uav_positions_ =
        this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/uav_positions", rclcpp::SensorDataQoS());

    sub_uav_other_cluster_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar_other_cluster", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::UavOtherClusterCallback, this,
                      std::placeholders::_1));

    // params
    this->declare_parameter<int>("uav_other_timeout_ms", 500);
    this->declare_parameter<double>("uav_default_z", 1.95);
    this->declare_parameter<double>("uav_huber_delta_xy", 0.8);
    this->declare_parameter<double>("uav_w_vision_xy", 1.0);
    this->declare_parameter<double>("uav_w_lidar_xy", 0.7);
    this->declare_parameter<double>("uav_assign_gate_m", 3.0);

    this->declare_parameter<double>("uav_z_ema_alpha", 0.90);
    this->declare_parameter<double>("uav_z_jump_max", 0.80);

    this->declare_parameter<int>("uav_lost_timeout_ms", 500);
    this->declare_parameter<double>("uav_q_pos", 2.0);
    this->declare_parameter<double>("uav_q_vel", 5.0);
    this->declare_parameter<double>("uav_r_meas_xy", 0.25);
    this->declare_parameter<double>("uav_r_meas_z", 0.16);
    this->declare_parameter<double>("uav_gate_maha", 16.27);

    uav_other_timeout_ms_ =
        this->get_parameter("uav_other_timeout_ms").as_int();
    uav_default_z_ = this->get_parameter("uav_default_z").as_double();
    uav_huber_delta_xy_ =
        this->get_parameter("uav_huber_delta_xy").as_double();
    uav_w_vision_xy_ = this->get_parameter("uav_w_vision_xy").as_double();
    uav_w_lidar_xy_ = this->get_parameter("uav_w_lidar_xy").as_double();
    uav_assign_gate_m_ =
        this->get_parameter("uav_assign_gate_m").as_double();

    uav_z_ema_alpha_ = this->get_parameter("uav_z_ema_alpha").as_double();
    uav_z_jump_max_ = this->get_parameter("uav_z_jump_max").as_double();
    if (uav_z_ema_alpha_ < 0.0)
        uav_z_ema_alpha_ = 0.0;
    if (uav_z_ema_alpha_ > 0.99)
        uav_z_ema_alpha_ = 0.99;
    if (uav_z_jump_max_ < 0.0)
        uav_z_jump_max_ = 0.0;

    uav_lost_timeout_ms_ =
        this->get_parameter("uav_lost_timeout_ms").as_int();
    uav_q_pos_ = this->get_parameter("uav_q_pos").as_double();
    uav_q_vel_ = this->get_parameter("uav_q_vel").as_double();
    uav_r_meas_xy_ = this->get_parameter("uav_r_meas_xy").as_double();
    uav_r_meas_z_ = this->get_parameter("uav_r_meas_z").as_double();
    uav_gate_maha_ = this->get_parameter("uav_gate_maha").as_double();
}

void Resolve::MatchInfoCallback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    for (int i = 0; i < 6; i++)
        markers[i] = msg->marks[i];
    for (int i = 0; i < 16; i++)
        robot_hp[i] = msg->robot_hp[i];
    match_time = msg->match_time;
}

void Resolve::callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    cv::Point2f point(msg->x, msg->y);
    auto        center_point = parser_->parse(point);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ p;
    p.x = center_point.x;
    p.y = 15 - center_point.y;
    p.z = 1.0;
    cloud->points.push_back(p);

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "rm_frame";
    pub->publish(output);
}

void Resolve::UavOtherClusterCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::lock_guard<std::mutex> lk(uav_other_mutex_);
    uav_other_count_ = 0;

    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z))
            continue;
        if (uav_other_count_ < 2) {
            uav_other_x_[uav_other_count_] = p.x;
            uav_other_y_[uav_other_count_] = p.y;
            uav_other_z_[uav_other_count_] = p.z;
            uav_other_count_++;
        } else {
            break;
        }
    }

    uav_other_stamp_ = is_stamp_zero(msg->header.stamp)
                           ? this->now()
                           : rclcpp::Time(msg->header.stamp);
}

// 3.4: z-plane filter (EMA + jump gate), used only for intersection plane
float Resolve::GetFilteredZPlane(int idx, double z_raw)
{
    if (!uav_z_filt_inited_[idx]) {
        uav_z_filt_inited_[idx] = true;
        uav_z_plane_filt_[idx] = z_raw;
        return static_cast<float>(z_raw);
    }

    const double z_prev = uav_z_plane_filt_[idx];
    const double dz = std::fabs(z_raw - z_prev);

    if (uav_z_jump_max_ > 0.0 && dz > uav_z_jump_max_) {
        // reject jump
        return static_cast<float>(z_prev);
    }

    const double a = uav_z_ema_alpha_;
    uav_z_plane_filt_[idx] = a * z_prev + (1.0 - a) * z_raw;
    return static_cast<float>(uav_z_plane_filt_[idx]);
}

// KF helpers
void Resolve::UavKalmanReset(int idx)
{
    uav_kf_inited_[idx] = false;
    uav_kf_stamp_[idx] = rclcpp::Time(0, 0, RCL_ROS_TIME);
    uav_kf_last_update_[idx] = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void Resolve::UavKalmanInit(int idx, double x, double y, double z,
                            const rclcpp::Time& stamp)
{
    uav_kf_inited_[idx] = true;
    uav_kf_stamp_[idx] = stamp;
    uav_kf_last_update_[idx] = stamp;

    uav_x_[idx] = x;
    uav_y_[idx] = y;
    uav_z_[idx] = z;
    uav_vx_[idx] = 0.0;
    uav_vy_[idx] = 0.0;
    uav_vz_[idx] = 0.0;

    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            uav_P_[idx][r][c] = 0.0;
    uav_P_[idx][0][0] = 1.0;
    uav_P_[idx][1][1] = 1.0;
    uav_P_[idx][2][2] = 4.0;
    uav_P_[idx][3][3] = 25.0;
    uav_P_[idx][4][4] = 25.0;
    uav_P_[idx][5][5] = 25.0;
}

void Resolve::UavKalmanPredict(int idx, const rclcpp::Time& stamp)
{
    if (!uav_kf_inited_[idx])
        return;
    double dt = (stamp - uav_kf_stamp_[idx]).seconds();
    if (dt < 0.0)
        dt = 0.0;
    if (dt > 0.2)
        dt = 0.2;
    uav_kf_stamp_[idx] = stamp;

    uav_x_[idx] += uav_vx_[idx] * dt;
    uav_y_[idx] += uav_vy_[idx] * dt;
    uav_z_[idx] += uav_vz_[idx] * dt;

    double F[6][6] = {0};
    for (int i = 0; i < 6; i++)
        F[i][i] = 1.0;
    F[0][3] = dt;
    F[1][4] = dt;
    F[2][5] = dt;

    const double dt2 = dt * dt;
    const double dt4 = dt2 * dt2;

    double Q[6][6] = {0};
    Q[0][0] = uav_q_pos_ * dt4;
    Q[1][1] = uav_q_pos_ * dt4;
    Q[2][2] = uav_q_pos_ * dt4;
    Q[3][3] = uav_q_vel_ * dt2;
    Q[4][4] = uav_q_vel_ * dt2;
    Q[5][5] = uav_q_vel_ * dt2;

    double FP[6][6] = {0};
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++)
                FP[i][j] += F[i][k] * uav_P_[idx][k][j];
    double FPFt[6][6] = {0};
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++)
                FPFt[i][j] += FP[i][k] * F[j][k];

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            uav_P_[idx][i][j] = FPFt[i][j] + Q[i][j];
}

void Resolve::UavKalmanUpdate(int idx, double mx, double my, double mz)
{
    if (!uav_kf_inited_[idx])
        return;

    const double y0 = mx - uav_x_[idx];
    const double y1 = my - uav_y_[idx];
    const double y2 = mz - uav_z_[idx];

    double S[3][3] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            S[i][j] = uav_P_[idx][i][j];
    S[0][0] += uav_r_meas_xy_;
    S[1][1] += uav_r_meas_xy_;
    S[2][2] += uav_r_meas_z_;

    const double a = S[0][0], b = S[0][1], c = S[0][2];
    const double d = S[1][0], e = S[1][1], f = S[1][2];
    const double g = S[2][0], h = S[2][1], i = S[2][2];
    const double det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::fabs(det) < 1e-12)
        return;
    const double invdet = 1.0 / det;

    double Sinv[3][3];
    Sinv[0][0] = (e * i - f * h) * invdet;
    Sinv[0][1] = -(b * i - c * h) * invdet;
    Sinv[0][2] = (b * f - c * e) * invdet;
    Sinv[1][0] = -(d * i - f * g) * invdet;
    Sinv[1][1] = (a * i - c * g) * invdet;
    Sinv[1][2] = -(a * f - c * d) * invdet;
    Sinv[2][0] = (d * h - e * g) * invdet;
    Sinv[2][1] = -(a * h - b * g) * invdet;
    Sinv[2][2] = (a * e - b * d) * invdet;

    const double maha =
        y0 * (Sinv[0][0] * y0 + Sinv[0][1] * y1 + Sinv[0][2] * y2) +
        y1 * (Sinv[1][0] * y0 + Sinv[1][1] * y1 + Sinv[1][2] * y2) +
        y2 * (Sinv[2][0] * y0 + Sinv[2][1] * y1 + Sinv[2][2] * y2);

    if (maha > uav_gate_maha_)
        return;

    double PHt[6][3] = {0};
    for (int r = 0; r < 6; r++)
        for (int c2 = 0; c2 < 3; c2++)
            PHt[r][c2] = uav_P_[idx][r][c2];
    double K[6][3] = {0};
    for (int r = 0; r < 6; r++)
        for (int c2 = 0; c2 < 3; c2++)
            for (int k = 0; k < 3; k++)
                K[r][c2] += PHt[r][k] * Sinv[k][c2];

    uav_x_[idx] += K[0][0] * y0 + K[0][1] * y1 + K[0][2] * y2;
    uav_y_[idx] += K[1][0] * y0 + K[1][1] * y1 + K[1][2] * y2;
    uav_z_[idx] += K[2][0] * y0 + K[2][1] * y1 + K[2][2] * y2;
    uav_vx_[idx] += K[3][0] * y0 + K[3][1] * y1 + K[3][2] * y2;
    uav_vy_[idx] += K[4][0] * y0 + K[4][1] * y1 + K[4][2] * y2;
    uav_vz_[idx] += K[5][0] * y0 + K[5][1] * y1 + K[5][2] * y2;

    double I_KH[6][6] = {0};
    for (int r = 0; r < 6; r++)
        I_KH[r][r] = 1.0;
    for (int r = 0; r < 6; r++) {
        I_KH[r][0] -= K[r][0];
        I_KH[r][1] -= K[r][1];
        I_KH[r][2] -= K[r][2];
    }

    double newP[6][6] = {0};
    for (int r = 0; r < 6; r++)
        for (int c2 = 0; c2 < 6; c2++)
            for (int k = 0; k < 6; k++)
                newP[r][c2] += I_KH[r][k] * uav_P_[idx][k][c2];
    for (int r = 0; r < 6; r++)
        for (int c2 = 0; c2 < 6; c2++)
            uav_P_[idx][r][c2] = newP[r][c2];
}

void Resolve::DetectCallback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    vision_interface::msg::DetectResult send_data;
    send_data.header = msg->header;

    // Prepare uav output (always publish, NaN means invalid)
    geometry_msgs::msg::PoseArray uav_out;
    uav_out.header = msg->header;
    uav_out.header.frame_id = "rm_frame";
    uav_out.poses.resize(2);

    auto set_nan_pose = [](geometry_msgs::msg::Pose& p) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        p.position.x = nan;
        p.position.y = nan;
        p.position.z = nan;
        p.orientation.x = nan;
        p.orientation.y = nan;
        p.orientation.z = nan;
        p.orientation.w = nan;
    };
    set_nan_pose(uav_out.poses[0]);
    set_nan_pose(uav_out.poses[1]);

    const bool has_blue_uav =
        (msg->blue_x[6] != 0.0f && msg->blue_y[6] != 0.0f);
    const bool has_red_uav =
        (msg->red_x[6] != 0.0f && msg->red_y[6] != 0.0f);

    // read other cache
    double cX[2]{0.0, 0.0}, cY[2]{0.0, 0.0}, cZ[2]{0.0, 0.0};
    int    cN = 0;
    bool   other_fresh = false;
    {
        std::lock_guard<std::mutex> lk(uav_other_mutex_);
        cN = uav_other_count_;
        if (cN > 0) {
            cX[0] = uav_other_x_[0];
            cY[0] = uav_other_y_[0];
            cZ[0] = uav_other_z_[0];
        }
        if (cN > 1) {
            cX[1] = uav_other_x_[1];
            cY[1] = uav_other_y_[1];
            cZ[1] = uav_other_z_[1];
        }
        const rclcpp::Time t_cam(msg->header.stamp);
        const auto         dt_ns = (t_cam - uav_other_stamp_).nanoseconds();
        const auto         dt_ms = std::llabs(dt_ns) / 1000000LL;
        other_fresh = (dt_ms <= uav_other_timeout_ms_);
    }

    auto solve_xy_on_z = [&](float px, float py,
                             float z_plane) -> cv::Point2f {
        cv::Point2f p_in(px, py);
        auto        p = parser_->parse_with_height(p_in, z_plane);
        p.y = 15 + p.y;
        p.x = std::clamp(p.x, 0.0f, 28.0f);
        p.y = std::clamp(p.y, 0.0f, 15.0f);
        return p;
    };

    // assign
    int    assign_blue = -1, assign_red = -1;
    double dist_blue = 1e9, dist_red = 1e9;

    if (other_fresh && cN > 0 && (has_blue_uav || has_red_uav)) {
        const float z0 = static_cast<float>(uav_default_z_);
        bool        blue_ok = false, red_ok = false;
        cv::Point2f blue0, red0;
        if (has_blue_uav) {
            blue0 = solve_xy_on_z(msg->blue_x[6], msg->blue_y[6], z0);
            blue_ok = true;
        }
        if (has_red_uav) {
            red0 = solve_xy_on_z(msg->red_x[6], msg->red_y[6], z0);
            red_ok = true;
        }

        auto dist2 = [&](double x, double y,
                         const cv::Point2f& p) -> double {
            double dx = x - p.x, dy = y - p.y;
            return dx * dx + dy * dy;
        };

        if (cN == 1) {
            if (blue_ok && !red_ok) {
                assign_blue = 0;
                dist_blue = std::sqrt(dist2(cX[0], cY[0], blue0));
            } else if (!blue_ok && red_ok) {
                assign_red = 0;
                dist_red = std::sqrt(dist2(cX[0], cY[0], red0));
            } else if (blue_ok && red_ok) {
                double db = dist2(cX[0], cY[0], blue0),
                       dr = dist2(cX[0], cY[0], red0);
                if (db <= dr) {
                    assign_blue = 0;
                    dist_blue = std::sqrt(db);
                } else {
                    assign_red = 0;
                    dist_red = std::sqrt(dr);
                }
            }
        } else {  // >=2
            if (blue_ok && !red_ok) {
                double d0 = dist2(cX[0], cY[0], blue0),
                       d1 = dist2(cX[1], cY[1], blue0);
                if (d0 <= d1) {
                    assign_blue = 0;
                    dist_blue = std::sqrt(d0);
                } else {
                    assign_blue = 1;
                    dist_blue = std::sqrt(d1);
                }
            } else if (!blue_ok && red_ok) {
                double d0 = dist2(cX[0], cY[0], red0),
                       d1 = dist2(cX[1], cY[1], red0);
                if (d0 <= d1) {
                    assign_red = 0;
                    dist_red = std::sqrt(d0);
                } else {
                    assign_red = 1;
                    dist_red = std::sqrt(d1);
                }
            } else if (blue_ok && red_ok) {
                double db0 = dist2(cX[0], cY[0], blue0),
                       db1 = dist2(cX[1], cY[1], blue0);
                double dr0 = dist2(cX[0], cY[0], red0),
                       dr1 = dist2(cX[1], cY[1], red0);
                double sumA = db0 + dr1, sumB = db1 + dr0;
                if (sumA <= sumB) {
                    assign_blue = 0;
                    assign_red = 1;
                    dist_blue = std::sqrt(db0);
                    dist_red = std::sqrt(dr1);
                } else {
                    assign_blue = 1;
                    assign_red = 0;
                    dist_blue = std::sqrt(db1);
                    dist_red = std::sqrt(dr0);
                }
            }
        }

        const double gate = std::max(0.1, uav_assign_gate_m_);
        if (assign_blue >= 0 && dist_blue > gate)
            assign_blue = -1;
        if (assign_red >= 0 && dist_red > gate)
            assign_red = -1;
        if (has_blue_uav && has_red_uav) {
            if (assign_blue < 0 || assign_red < 0) {
                assign_blue = -1;
                assign_red = -1;
            }
        }
    }

    // compute UAV measurement per side
    bool   meas_ok[2]{false, false};
    double meas_x[2]{0, 0}, meas_y[2]{0, 0}, meas_z[2]{0, 0};

    auto compute_meas = [&](int idx, float px, float py, int assigned) {
        bool has_cluster = (other_fresh && assigned >= 0 && assigned < cN);
        double xL = 0, yL = 0, zL = uav_default_z_;
        float  z_plane = static_cast<float>(uav_default_z_);
        if (has_cluster) {
            xL = cX[assigned];
            yL = cY[assigned];
            zL = cZ[assigned];
            z_plane = GetFilteredZPlane(idx, zL);  // 3.4 only here
        }
        cv::Point2f pv = solve_xy_on_z(px, py, z_plane);
        double      xV = pv.x, yV = pv.y;
        double      x_out = xV, y_out = yV;
        double      z_out = has_cluster ? zL : uav_default_z_;

        if (has_cluster) {
            const double dx = xV - xL, dy = yV - yL;
            const double r_xy = std::sqrt(dx * dx + dy * dy);
            const double delta = std::max(1e-6, uav_huber_delta_xy_);
            const double huber =
                (r_xy <= delta) ? 1.0 : (delta / std::max(1e-6, r_xy));
            const double wV = std::max(1e-6, uav_w_vision_xy_);
            const double wL = std::max(1e-6, uav_w_lidar_xy_) * huber;
            x_out = (wV * xV + wL * xL) / (wV + wL);
            y_out = (wV * yV + wL * yL) / (wV + wL);
        }

        meas_ok[idx] = true;
        meas_x[idx] = x_out;
        meas_y[idx] = y_out;
        meas_z[idx] = z_out;
    };

    if (has_blue_uav)
        compute_meas(0, msg->blue_x[6], msg->blue_y[6], assign_blue);
    if (has_red_uav)
        compute_meas(1, msg->red_x[6], msg->red_y[6], assign_red);

    // KF update/predict + output pose
    const rclcpp::Time t_now(msg->header.stamp);
    for (int idx = 0; idx < 2; ++idx) {
        if (uav_kf_inited_[idx]) {
            UavKalmanPredict(idx, t_now);
            if (meas_ok[idx]) {
                UavKalmanUpdate(idx, meas_x[idx], meas_y[idx], meas_z[idx]);
                uav_kf_last_update_[idx] = t_now;
            } else {
                const double dt_ms =
                    std::fabs(
                        (t_now - uav_kf_last_update_[idx]).seconds()) *
                    1000.0;
                if (uav_kf_last_update_[idx].nanoseconds() != 0 &&
                    dt_ms > (double)uav_lost_timeout_ms_) {
                    UavKalmanReset(idx);
                }
            }
        } else {
            if (meas_ok[idx])
                UavKalmanInit(idx, meas_x[idx], meas_y[idx], meas_z[idx],
                              t_now);
        }

        if (uav_kf_inited_[idx]) {
            uav_out.poses[idx].position.x = uav_x_[idx];
            uav_out.poses[idx].position.y = uav_y_[idx];
            uav_out.poses[idx].position.z = uav_z_[idx];
        } else if (meas_ok[idx]) {
            uav_out.poses[idx].position.x = meas_x[idx];
            uav_out.poses[idx].position.y = meas_y[idx];
            uav_out.poses[idx].position.z = meas_z[idx];
        }
    }

    // Now keep original car outputs (i=0..5) and fill UAV slot from
    // KF/measurement
    for (int i = 0; i < 7; i++) {
        // blue
        if (msg->blue_x[i] != 0 && msg->blue_y[i] != 0) {
            cv::Point2f p(msg->blue_x[i], msg->blue_y[i]);
            float       h = parser_->get_height(p);
            auto        cp = parser_->parse_with_height(p, h);
            cp.y = 15 + cp.y;
            cp.x = std::clamp(cp.x, 0.0f, 28.0f);
            cp.y = std::clamp(cp.y, 0.0f, 15.0f);

            if (i == 6) {
                if (uav_kf_inited_[0]) {
                    cp.x = (float)uav_x_[0];
                    cp.y = (float)uav_y_[0];
                } else if (meas_ok[0]) {
                    cp.x = (float)meas_x[0];
                    cp.y = (float)meas_y[0];
                }
            }
            send_data.blue_x[i] = cp.x;
            send_data.blue_y[i] = cp.y;
        } else {
            send_data.blue_x[i] = 0;
            send_data.blue_y[i] = 0;
        }

        // red
        if (msg->red_x[i] != 0 && msg->red_y[i] != 0) {
            cv::Point2f p(msg->red_x[i], msg->red_y[i]);
            float       h = parser_->get_height(p);
            auto        cp = parser_->parse_with_height(p, h);
            cp.y = 15 + cp.y;
            cp.x = std::clamp(cp.x, 0.0f, 28.0f);
            cp.y = std::clamp(cp.y, 0.0f, 15.0f);

            if (i == 6) {
                if (uav_kf_inited_[1]) {
                    cp.x = (float)uav_x_[1];
                    cp.y = (float)uav_y_[1];
                } else if (meas_ok[1]) {
                    cp.x = (float)meas_x[1];
                    cp.y = (float)meas_y[1];
                }
            }
            send_data.red_x[i] = cp.x;
            send_data.red_y[i] = cp.y;
        } else {
            send_data.red_x[i] = 0;
            send_data.red_y[i] = 0;
        }
    }

    // publish resolve_result (debug_map depends on this)
    pub_radar->publish(send_data);

    // publish /uav_positions every frame (debug_map depends on timely
    // updates)
    pub_uav_positions_->publish(uav_out);
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Resolve)
