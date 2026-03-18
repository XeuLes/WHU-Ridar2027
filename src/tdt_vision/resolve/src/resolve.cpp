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
    // -------------------- parser config paths (ROS2 parameters) --------------------
    this->declare_parameter<std::string>("out_matrix_path", "config/out_matrix.yaml");
    this->declare_parameter<std::string>("camera_params_path", "config/camera_params.yaml");
    this->declare_parameter<std::string>("region_points_path", "config/RM2025_Points.yaml");

    this->declare_parameter<bool>("is_test", false);

    this->declare_parameter<double>("uav_default_out_x", 16.6);
    this->declare_parameter<double>("uav_default_out_y", 6.18);
    this->declare_parameter<double>("uav_default_out_z", 1.58);

    // minimap path (optional)
    this->declare_parameter<std::string>("minimap_path", "config/RM2025.png");

    const std::string out_matrix_path   = this->get_parameter("out_matrix_path").as_string();
    const std::string camera_params_path = this->get_parameter("camera_params_path").as_string();
    const std::string region_points_path = this->get_parameter("region_points_path").as_string();
    const std::string minimap_path       = this->get_parameter("minimap_path").as_string();

    is_test_ = this->get_parameter("is_test").as_bool();

    // -------------------- construct parser --------------------
    parser_ = nullptr;
    try {
        parser_ = new parser(out_matrix_path, camera_params_path, region_points_path);
        RCLCPP_INFO(this->get_logger(),
                    "[INIT] parser config:\n  out_matrix=%s\n  camera_params=%s\n  region_points=%s",
                    out_matrix_path.c_str(),
                    camera_params_path.c_str(),
                    region_points_path.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[FATAL] parser构造异常: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "[FATAL] parser构造未知异常");
    }

    // -------------------- load minimap --------------------
    minimap = cv::imread(minimap_path);
    if (minimap.empty()) {
        RCLCPP_WARN(this->get_logger(), "minimap图片加载失败: %s", minimap_path.c_str());
    }

    

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
    this->declare_parameter<double>("uav_default_z", 2.1);
    this->declare_parameter<double>("uav_huber_delta_xy", 0.8);
    this->declare_parameter<double>("uav_w_vision_xy", 0.5);  // 降低相机权重(透视变换外推不准)
    this->declare_parameter<double>("uav_w_lidar_xy", 0.7);   // 提高雷达权重
    this->declare_parameter<double>("uav_assign_gate_m", 5.0);

    this->declare_parameter<double>("uav_z_ema_alpha", 0.90);
    this->declare_parameter<double>("uav_z_jump_max", 0.80);

    this->declare_parameter<int>("uav_lost_timeout_ms", 500);
    this->declare_parameter<double>("uav_q_pos", 2.0);
    this->declare_parameter<double>("uav_q_vel", 5.0);
    this->declare_parameter<double>("uav_r_meas_xy", 0.25);
    this->declare_parameter<double>("uav_r_meas_z", 0.16);
    this->declare_parameter<double>("uav_gate_maha", 500);
    

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


    uav_default_out_x_ = this->get_parameter("uav_default_out_x").as_double();
    uav_default_out_y_ = this->get_parameter("uav_default_out_y").as_double();
    uav_default_out_z_ = this->get_parameter("uav_default_out_z").as_double();
}
Resolve::~Resolve()
{
    if (parser_) {
        delete parser_;
        parser_ = nullptr;
    }
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
    uav_other_clusters_.clear();
    uav_other_clusters_.reserve(cloud.points.size());

    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        uav_other_clusters_.push_back(p);
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

    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            uav_P_[idx][r][c] = 0.0;
        }
    }

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

    if (maha > uav_gate_maha_) {
        return;
    }

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
    if (!msg || !parser_) {
        return;
    }

    vision_interface::msg::DetectResult send_data;
    send_data.header = msg->header;
    if (is_stamp_zero(send_data.header.stamp)) {
        send_data.header.stamp = this->now();
    }

    geometry_msgs::msg::PoseArray uav_out;
    uav_out.header = msg->header;
    uav_out.header.frame_id = "rm_frame";
    if (is_stamp_zero(uav_out.header.stamp)) {
        uav_out.header.stamp = this->now();
    }
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
    auto set_default_pose = [&](geometry_msgs::msg::Pose& p) {
    p.position.x = uav_default_out_x_;
    p.position.y = uav_default_out_y_;
    p.position.z = uav_default_out_z_;
    p.orientation.x = 0.0;
    p.orientation.y = 0.0;
    p.orientation.z = 0.0;
    p.orientation.w = 1.0;
    };
    
    set_default_pose(uav_out.poses[0]);
    set_default_pose(uav_out.poses[1]);

    // ========== 阶段一：接收检测消息 ==========
    const bool has_blue_uav = (msg->blue_x[6] != 0.0f && msg->blue_y[6] != 0.0f);
    const bool has_red_uav = (msg->red_x[6] != 0.0f && msg->red_y[6] != 0.0f);
    
    if (has_blue_uav || has_red_uav) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段一：接收检测消息] ==========");
        if (has_blue_uav) {
            RCLCPP_INFO(this->get_logger(), "  蓝方UAV相机坐标: (%.2f, %.2f)", msg->blue_x[6], msg->blue_y[6]);
        } else {
            RCLCPP_INFO(this->get_logger(), "  蓝方UAV: 无检测");
        }
        if (has_red_uav) {
            RCLCPP_INFO(this->get_logger(), "  红方UAV相机坐标: (%.2f, %.2f)", msg->red_x[6], msg->red_y[6]);
        } else {
            RCLCPP_INFO(this->get_logger(), "  红方UAV: 无检测");
        }
    }

    // ========== 阶段二：读取雷达聚类数据 ==========
    std::vector<pcl::PointXYZ> clusters;
    bool other_fresh = false;

    {
        std::lock_guard<std::mutex> lk(uav_other_mutex_);
        clusters = uav_other_clusters_;

        const rclcpp::Time t_cam = is_stamp_zero(msg->header.stamp)
                                    ? this->now()
                                    : rclcpp::Time(msg->header.stamp);

        const auto dt_ns = (t_cam - uav_other_stamp_).nanoseconds();
        const auto dt_ms = std::llabs(dt_ns) / 1000000LL;
        other_fresh = (dt_ms <= uav_other_timeout_ms_);

        if (has_blue_uav || has_red_uav) {
            RCLCPP_INFO(this->get_logger(), "========== [阶段二：读取雷达聚类数据] ==========");
            RCLCPP_INFO(this->get_logger(),
                        "  聚类中心数量: %zu, 时间差: %lld ms, 数据新鲜: %s",
                        clusters.size(), (long long)dt_ms, other_fresh ? "是" : "否");

            if (other_fresh && !clusters.empty()) {
                for (size_t i = 0; i < clusters.size(); ++i) {
                    RCLCPP_INFO(this->get_logger(),
                                "  雷达聚类[%zu]: (%.3f, %.3f, %.3f)",
                                i, clusters[i].x, clusters[i].y, clusters[i].z);
                }
            } else if (clusters.empty()) {
                RCLCPP_WARN(this->get_logger(), "  ⚠️ 无可用聚类中心");
            } else {
                RCLCPP_WARN(this->get_logger(), "  ⚠️ 聚类数据超时");
            }
        }
    }

    // Lambda: 使用透视变换方法(对地面车辆已验证准确)
    auto solve_xy_on_z = [&](float px, float py, float z_plane) -> cv::Point2f {
        cv::Point2f pixel(px, py);
        // 使用透视变换: height参数是地面高度,需要减去ARMOR_HEIGHT=0.15
        // 对于UAV: z_plane是绝对高度,所以传入 z_plane - 0.15 作为地面高度
        cv::Point2f world_xy = parser_->parse_with_height(pixel, z_plane - 0.15f);
        // 坐标系转换: y轴偏移
        float y_offset = world_xy.y + 15.0f;
        float clamp_x = std::clamp(world_xy.x, 0.0f, 28.0f);
        float clamp_y = std::clamp(y_offset, 0.0f, 15.0f);
        return cv::Point2f(clamp_x, clamp_y);
    };
    auto solve_xyz_on_z = [&](float px, float py, float z_plane) -> cv::Point3f {
    cv::Point2f pixel(px, py);
    cv::Point2f world_xy = parser_->parse_with_height(pixel, z_plane - 0.15f);

    float y_offset = world_xy.y + 15.0f;
    float clamp_x = std::clamp(world_xy.x, 0.0f, 28.0f);
    float clamp_y = std::clamp(y_offset, 0.0f, 15.0f);
    float clamp_z = z_plane;   // UAV 当前假设高度

    return cv::Point3f(clamp_x, clamp_y, clamp_z);
    };

    // ========== 阶段三：相机-雷达数据关联（基于2.3m三维最近邻） ==========
    int assign_blue = -1, assign_red = -1;
    double dist_blue = 1e9, dist_red = 1e9;

    cv::Point3f blue_cam_3d(0.f, 0.f, 0.f);
    cv::Point3f red_cam_3d(0.f, 0.f, 0.f);

    if (other_fresh && !clusters.empty() && (has_blue_uav || has_red_uav)) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段三：相机-雷达数据关联] ==========");
        RCLCPP_INFO(this->get_logger(),
                    "  [三维匹配] 使用默认z=%.2fm反解相机UAV三维坐标，再做最近邻空间距离匹配",
                    uav_default_z_);

        auto dist3 = [](const cv::Point3f& a, const pcl::PointXYZ& b) -> double {
            const double dx = static_cast<double>(a.x) - b.x;
            const double dy = static_cast<double>(a.y) - b.y;
            const double dz = static_cast<double>(a.z) - b.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        const double gate = std::max(0.1, uav_assign_gate_m_);

        // 蓝方
        if (has_blue_uav) {
            blue_cam_3d = solve_xyz_on_z(msg->blue_x[6], msg->blue_y[6],
                                        static_cast<float>(uav_default_z_));

            RCLCPP_INFO(this->get_logger(),
                        "    蓝方相机(%.1f, %.1f) z=%.2f → 三维点(%.3f, %.3f, %.3f)",
                        msg->blue_x[6], msg->blue_y[6], uav_default_z_,
                        blue_cam_3d.x, blue_cam_3d.y, blue_cam_3d.z);

            for (size_t i = 0; i < clusters.size(); ++i) {
                double dist = dist3(blue_cam_3d, clusters[i]);
                RCLCPP_INFO(this->get_logger(),
                            "      vs 聚类[%zu](%.3f, %.3f, %.3f): 空间距离 %.3fm",
                            i, clusters[i].x, clusters[i].y, clusters[i].z, dist);

                if (dist < dist_blue) {
                    dist_blue = dist;
                    assign_blue = static_cast<int>(i);
                }
            }

            if (assign_blue >= 0 && dist_blue > gate) {
                RCLCPP_WARN(this->get_logger(),
                            "  蓝方: 最近聚类[%d]空间距离 %.3fm > 门限 %.3fm, ❌拒绝",
                            assign_blue, dist_blue, gate);
                assign_blue = -1;
            }
        }

        // 红方
        if (has_red_uav) {
            red_cam_3d = solve_xyz_on_z(msg->red_x[6], msg->red_y[6],
                                        static_cast<float>(uav_default_z_));

            RCLCPP_INFO(this->get_logger(),
                        "    红方相机(%.1f, %.1f) z=%.2f → 三维点(%.3f, %.3f, %.3f)",
                        msg->red_x[6], msg->red_y[6], uav_default_z_,
                        red_cam_3d.x, red_cam_3d.y, red_cam_3d.z);

            for (size_t i = 0; i < clusters.size(); ++i) {
                double dist = dist3(red_cam_3d, clusters[i]);
                RCLCPP_INFO(this->get_logger(),
                            "      vs 聚类[%zu](%.3f, %.3f, %.3f): 空间距离 %.3fm",
                            i, clusters[i].x, clusters[i].y, clusters[i].z, dist);

                if (dist < dist_red) {
                    dist_red = dist;
                    assign_red = static_cast<int>(i);
                }
            }

            if (assign_red >= 0 && dist_red > gate) {
                RCLCPP_WARN(this->get_logger(),
                            "  红方: 最近聚类[%d]空间距离 %.3fm > 门限 %.3fm, ❌拒绝",
                            assign_red, dist_red, gate);
                assign_red = -1;
            }
        }

        // 蓝红冲突消解
        if (has_blue_uav && has_red_uav && assign_blue >= 0 && assign_red >= 0) {
            if (assign_blue == assign_red) {
                if (dist_blue <= dist_red) {
                    RCLCPP_WARN(this->get_logger(),
                                "  ⚠️ 冲突: 蓝红都选聚类[%d], 保留蓝方(%.3fm < %.3fm)",
                                assign_blue, dist_blue, dist_red);
                    assign_red = -1;
                } else {
                    RCLCPP_WARN(this->get_logger(),
                                "  ⚠️ 冲突: 蓝红都选聚类[%d], 保留红方(%.3fm < %.3fm)",
                                assign_red, dist_red, dist_blue);
                    assign_blue = -1;
                }
            }
        }

        if (assign_blue >= 0) {
            RCLCPP_INFO(this->get_logger(), "  ✓ 蓝方 -> 聚类[%d] 空间距离 %.3fm",
                        assign_blue, dist_blue);
        } else if (has_blue_uav) {
            RCLCPP_WARN(this->get_logger(), "  ❌ 蓝方: 无匹配聚类");
        }

        if (assign_red >= 0) {
            RCLCPP_INFO(this->get_logger(), "  ✓ 红方 -> 聚类[%d] 空间距离 %.3fm",
                        assign_red, dist_red);
        } else if (has_red_uav) {
            RCLCPP_WARN(this->get_logger(), "  ❌ 红方: 无匹配聚类");
        }
    }

    // ========== 阶段四：相机-雷达融合测量 ==========
    bool   meas_ok[2]{false, false};
    double meas_x[2]{0, 0}, meas_y[2]{0, 0}, meas_z[2]{0, 0};

    auto compute_meas = [&](int idx, float px, float py, int assigned) {
        const char* side_name = (idx == 0) ? "蓝方" : "红方";
        bool has_cluster =
        (other_fresh &&
        assigned >= 0 &&
        assigned < static_cast<int>(clusters.size()));
        
        if (!has_cluster) {
            float z_plane = static_cast<float>(uav_default_z_);
            cv::Point2f pv = solve_xy_on_z(px, py, z_plane);

            meas_ok[idx] = true;
            meas_x[idx] = pv.x;
            meas_y[idx] = pv.y;
            meas_z[idx] = uav_default_z_;

            RCLCPP_WARN(this->get_logger(),
                        "  %s: 无合法聚类( other_fresh=%s, assigned=%d, clusters=%zu )，纯相机模式 z=%.2f -> (%.3f, %.3f, %.3f)",
                        side_name,
                        other_fresh ? "true" : "false",
                        assigned,
                        clusters.size(),
                        z_plane,
                        meas_x[idx], meas_y[idx], meas_z[idx]);
            return;
        }
        
        // 融合模式: 智能选择z值
        double xL = clusters[assigned].x;
        double yL = clusters[assigned].y;
        double zL = clusters[assigned].z;
        
        // 步骤3A: 先用默认z投影,计算距离
        float z_default = static_cast<float>(uav_default_z_);
        cv::Point2f pv_default = solve_xy_on_z(px, py, z_default);
        double dx_def = pv_default.x - xL, dy_def = pv_default.y - yL;
        double dist_default = std::sqrt(dx_def * dx_def + dy_def * dy_def);
        
        // 步骤3B: 用雷达真实z投影,计算距离
        float z_lidar = GetFilteredZPlane(idx, zL);
        cv::Point2f pv_lidar = solve_xy_on_z(px, py, z_lidar);
        double dx_lid = pv_lidar.x - xL, dy_lid = pv_lidar.y - yL;
        double dist_lidar = std::sqrt(dx_lid * dx_lid + dy_lid * dy_lid);
        
        // 步骤3C: 选择距离更小的z值
        float z_best;
        cv::Point2f pv_best;
        double dist_best;
        if (dist_lidar < dist_default) {
            z_best = z_lidar;
            pv_best = pv_lidar;
            dist_best = dist_lidar;
            RCLCPP_INFO(this->get_logger(), "  %s: [智能选z] 雷达z=%.3f距离%.3fm < 默认z=%.2f距离%.3fm, ✓选雷达z", 
                side_name, z_lidar, dist_lidar, z_default, dist_default);
        } else {
            z_best = z_default;
            pv_best = pv_default;
            dist_best = dist_default;
            RCLCPP_INFO(this->get_logger(), "  %s: [智能选z] 默认z=%.2f距离%.3fm < 雷达z=%.3f距离%.3fm, ✓选默认z", 
                side_name, z_default, dist_default, z_lidar, dist_lidar);
        }
        
        double xV = pv_best.x, yV = pv_best.y;
        
        RCLCPP_INFO(this->get_logger(), "  %s: [精确投影] 相机像素(%.1f,%.1f) 用z=%.3f → 精确XY(%.3f,%.3f)", 
            side_name, px, py, z_best, xV, yV);
        
        // 步骤4: 相机XY与雷达XY加权融合
        const double dx = xV - xL, dy = yV - yL;
        const double r_xy = std::sqrt(dx * dx + dy * dy);
        const double delta = std::max(1e-6, uav_huber_delta_xy_);
        const double huber = (r_xy <= delta) ? 1.0 : (delta / std::max(1e-6, r_xy));
        const double wV = std::max(1e-6, uav_w_vision_xy_);
        const double wL = std::max(1e-6, uav_w_lidar_xy_) * huber;
        
        meas_ok[idx] = true;
        const double w_sum = std::max(1e-6, wV + wL);
        meas_x[idx] = (wV * xV + wL * xL) / w_sum;
        meas_y[idx] = (wV * yV + wL * yL) / w_sum;
        meas_z[idx] = z_lidar;
        
        // 计算归一化后的权重(用于日志显示)
        const double wV_norm = wV / w_sum;
        const double wL_norm = wL / w_sum;

        RCLCPP_INFO(this->get_logger(),
            "  %s: 使用聚类[%d] = (%.3f, %.3f, %.3f)",
            side_name, assigned, xL, yL, zL);
        
        RCLCPP_INFO(this->get_logger(), "  %s: [融合] 相机(%.2f,%.2f)*%.3f + 雷达(%.2f,%.2f)*%.3f = 最终(%.3f,%.3f,%.3f)", 
            side_name, xV, yV, wV_norm, xL, yL, wL_norm, meas_x[idx], meas_y[idx], meas_z[idx]);
    };

    if (has_blue_uav || has_red_uav) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段四：相机-雷达融合测量] ==========");
    }
    if (has_blue_uav) {
        compute_meas(0, msg->blue_x[6], msg->blue_y[6], assign_blue);
    }
    if (has_red_uav) {
        compute_meas(1, msg->red_x[6], msg->red_y[6], assign_red);
    }

    // ========== 阶段五：卡尔曼滤波预测与更新 ==========
    const rclcpp::Time t_now = is_stamp_zero(msg->header.stamp) 
                               ? this->now() 
                               : rclcpp::Time(msg->header.stamp);
    
    bool any_kf_activity = false;
    for (int idx = 0; idx < 2; ++idx) {
        if (uav_kf_inited_[idx] || meas_ok[idx]) {
            any_kf_activity = true;
            break;
        }
    }
    
    if (any_kf_activity) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段五：卡尔曼滤波] ==========");
    }
    
    for (int idx = 0; idx < 2; ++idx) {
        const char* side_name = (idx == 0) ? "蓝方" : "红方";
        
        if (uav_kf_inited_[idx]) {
            UavKalmanPredict(idx, t_now);
            if (meas_ok[idx]) {
                UavKalmanUpdate(idx, meas_x[idx], meas_y[idx], meas_z[idx]);
                uav_kf_last_update_[idx] = t_now;
                RCLCPP_INFO(this->get_logger(), "  %s: KF更新 -> (%.3f, %.3f, %.3f)", 
                    side_name, uav_x_[idx], uav_y_[idx], uav_z_[idx]);
            } else {
                const double dt_ms =
                    std::fabs((t_now - uav_kf_last_update_[idx]).seconds()) * 1000.0;
                if (uav_kf_last_update_[idx].nanoseconds() != 0 &&
                    dt_ms > (double)uav_lost_timeout_ms_) {
                    UavKalmanReset(idx);
                    RCLCPP_WARN(this->get_logger(), "  %s: ⚠️ 超时%.1fms，重置KF", side_name, dt_ms);
                }
            }
        } else {
            if (meas_ok[idx]) {
                UavKalmanInit(idx, meas_x[idx], meas_y[idx], meas_z[idx], t_now);
                RCLCPP_INFO(this->get_logger(), "  %s: ✓ KF初始化 at (%.3f, %.3f, %.3f)", 
                    side_name, uav_x_[idx], uav_y_[idx], uav_z_[idx]);
            }
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

    // ========== 阶段六：最终输出坐标 ==========
    bool has_output = false;
    for (int idx = 0; idx < 2; ++idx) {
        if (uav_kf_inited_[idx] || meas_ok[idx]) {
            has_output = true;
            break;
        }
    }
    
    if (has_output) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段六：最终输出] ==========");
        if (uav_kf_inited_[0] || meas_ok[0]) {
            double out_x = uav_kf_inited_[0] ? uav_x_[0] : meas_x[0];
            double out_y = uav_kf_inited_[0] ? uav_y_[0] : meas_y[0];
            double out_z = uav_kf_inited_[0] ? uav_z_[0] : meas_z[0];
            RCLCPP_INFO(this->get_logger(), "  蓝方UAV输出: (%.3f, %.3f, %.3f) [%s]", 
                out_x, out_y, out_z, uav_kf_inited_[0] ? "KF" : "测量");
        }
        if (uav_kf_inited_[1] || meas_ok[1]) {
            double out_x = uav_kf_inited_[1] ? uav_x_[1] : meas_x[1];
            double out_y = uav_kf_inited_[1] ? uav_y_[1] : meas_y[1];
            double out_z = uav_kf_inited_[1] ? uav_z_[1] : meas_z[1];
            RCLCPP_INFO(this->get_logger(), "  红方UAV输出: (%.3f, %.3f, %.3f) [%s]", 
                out_x, out_y, out_z, uav_kf_inited_[1] ? "KF" : "测量");
        }
    }

    // Now keep original car outputs (i=0..5) and fill UAV slot from KF/measurement
    for (int i = 0; i < 7; i++) {
        // blue
        if (msg->blue_x[i] != 0 && msg->blue_y[i] != 0) {
            cv::Point2f p(msg->blue_x[i], msg->blue_y[i]);
            if (i == 6) {
                // UAV: 直接使用KF或测量值,无需重新转换
                if (uav_kf_inited_[0]) {
                    send_data.blue_x[i] = (float)uav_x_[0];
                    send_data.blue_y[i] = (float)uav_y_[0];
                } else if (meas_ok[0]) {
                    send_data.blue_x[i] = (float)meas_x[0];
                    send_data.blue_y[i] = (float)meas_y[0];
                } else {
                    send_data.blue_x[i] = 0;
                    send_data.blue_y[i] = 0;
                }
            } else {
                float h = parser_->get_height(p);
                auto cp = parser_->parse_with_height(p, h);
                
                if (is_test_) {
                    // 测试模式：直接使用投影原始值，不做y偏移和clamp
                    send_data.blue_x[i] = cp.x;
                    send_data.blue_y[i] = cp.y;
                } else {
                    // 原逻辑：做y偏移和clamp
                    float y_offset = cp.y + 15.0f;
                    float clamp_x = std::clamp(cp.x, 0.0f, 28.0f);
                    float clamp_y = std::clamp(y_offset, 0.0f, 15.0f);
                    send_data.blue_x[i] = clamp_x;
                    send_data.blue_y[i] = clamp_y;
                }
            }
        } else {
            send_data.blue_x[i] = 0;
            send_data.blue_y[i] = 0;
        }

        // red
        if (msg->red_x[i] != 0 && msg->red_y[i] != 0) {
            cv::Point2f p(msg->red_x[i], msg->red_y[i]);
            if (i == 6) {
                // UAV: 直接使用KF或测量值,无需重新转换
                if (uav_kf_inited_[1]) {
                    send_data.red_x[i] = (float)uav_x_[1];
                    send_data.red_y[i] = (float)uav_y_[1];
                } else if (meas_ok[1]) {
                    send_data.red_x[i] = (float)meas_x[1];
                    send_data.red_y[i] = (float)meas_y[1];
                } else {
                    send_data.red_x[i] = 0;
                    send_data.red_y[i] = 0;
                }
            } else {
                
                float h = parser_->get_height(p);
                auto cp = parser_->parse_with_height(p, h);
                if (is_test_) {
                    // 测试模式：直接使用投影原始值，不做y偏移和clamp
                    send_data.red_x[i] = cp.x;
                    send_data.red_y[i] = cp.y;
                } else {
                    // 原逻辑：做y偏移和clamp
                    float y_offset = cp.y + 15.0f;
                    float clamp_x = std::clamp(cp.x, 0.0f, 28.0f);
                    float clamp_y = std::clamp(y_offset, 0.0f, 15.0f);
                    send_data.red_x[i] = clamp_x;
                    send_data.red_y[i] = clamp_y;
                }
            }
        } else {
            send_data.red_x[i] = 0;
            send_data.red_y[i] = 0;
        }
    }

    // publish resolve_result (debug_map depends on this)
    pub_radar->publish(send_data);

    // publish /uav_positions every frame (debug_map depends on timely
    // updates)
    
    // ================= 临时：/uav_positions 的第2个目标改为“蓝方2号小车” =================
// 约定：blue[1] 是蓝方2号车（数组索引 0..5 对应 1..6号车，1=>2号车）
{
    const int car_idx = 1;  // 蓝方2号小车
    const float x = send_data.blue_x[car_idx];
    const float y = send_data.blue_y[car_idx];

    if (x != 0.0f && y != 0.0f) {
        uav_out.poses[1].position.x = static_cast<double>(x);
        uav_out.poses[1].position.y = static_cast<double>(y);
        uav_out.poses[1].position.z = 0.0;   // 地面车先给0高度

        // 如需固定姿态，可打开下面4行
        // uav_out.poses[1].orientation.w = 1.0;
        // uav_out.poses[1].orientation.x = 0.0;
        // uav_out.poses[1].orientation.y = 0.0;
        // uav_out.poses[1].orientation.z = 0.0;
    } else {
        // 蓝方2号车无坐标时，保持原逻辑（KF/测量/默认值）
        // 如需强制无效，也可改成：
        // set_nan_pose(uav_out.poses[1]);
    }
}


    pub_uav_positions_->publish(uav_out);
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Resolve)
