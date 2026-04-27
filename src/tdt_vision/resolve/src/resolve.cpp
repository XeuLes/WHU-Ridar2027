#include "resolve.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

namespace tdt_radar {

static inline bool is_stamp_zero(const builtin_interfaces::msg::Time& t)
{
    return (t.sec == 0 && t.nanosec == 0);
}

static double compute_median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double median = values[mid];
    if ((values.size() & 1U) == 0U) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        median = 0.5 * (median + values[mid - 1]);
    }
    return median;
}

static double compute_mad(const std::vector<double>& values, double center)
{
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (double value : values) {
        abs_dev.push_back(std::fabs(value - center));
    }
    return compute_median(std::move(abs_dev));
}

static cv::Mat makeCameraMatrix64(const cv::Mat& camera_matrix)
{
    cv::Mat K;
    camera_matrix.convertTo(K, CV_64F);
    return K;
}

static cv::Mat makeDistCoeffs64(const cv::Mat& dist_coeffs)
{
    if (dist_coeffs.empty()) {
        return cv::Mat::zeros(1, 5, CV_64F);
    }
    cv::Mat D;
    dist_coeffs.convertTo(D, CV_64F);
    return D.reshape(1, 1);
}

static bool hasNonZeroDistCoeffs(const cv::Mat& dist_coeffs)
{
    if (dist_coeffs.empty()) {
        return false;
    }

    const cv::Mat D = makeDistCoeffs64(dist_coeffs);
    for (int i = 0; i < D.cols; ++i) {
        if (std::abs(D.at<double>(0, i)) > 1e-12) {
            return true;
        }
    }
    return false;
}

static bool pixelToCameraRay(const cv::Point2f& pixel,
                             const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs,
                             cv::Vec3d& ray_camera)
{
    if (camera_matrix.empty()) {
        return false;
    }

    const cv::Mat K = makeCameraMatrix64(camera_matrix);
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    if (std::abs(fx) < 1e-9 || std::abs(fy) < 1e-9) {
        return false;
    }

    if (hasNonZeroDistCoeffs(dist_coeffs)) {
        std::vector<cv::Point2f> src{pixel};
        std::vector<cv::Point2f> dst;
        cv::undistortPoints(src, dst, K, makeDistCoeffs64(dist_coeffs));
        if (dst.empty()) {
            return false;
        }
        ray_camera = cv::Vec3d(dst.front().x, dst.front().y, 1.0);
    } else {
        ray_camera = cv::Vec3d((pixel.x - cx) / fx, (pixel.y - cy) / fy, 1.0);
    }

    const double norm = cv::norm(ray_camera);
    if (norm <= 1e-9) {
        return false;
    }
    ray_camera /= norm;
    return true;
}

static void projectPointsCameraModel(const std::vector<cv::Point3f>& world_points,
                                     const cv::Mat& world_rvec,
                                     const cv::Mat& world_tvec,
                                     const cv::Mat& camera_matrix,
                                     const cv::Mat& dist_coeffs,
                                     std::vector<cv::Point2f>& image_points)
{
    image_points.clear();
    if (world_points.empty()) {
        return;
    }

    cv::projectPoints(world_points,
                      world_rvec,
                      world_tvec,
                      makeCameraMatrix64(camera_matrix),
                      makeDistCoeffs64(dist_coeffs),
                      image_points);
}

Resolve::Resolve(const rclcpp::NodeOptions& node_options)
    : Node("radar_resolve_node", node_options)
{ 
    // -------------------- parser config paths (ROS2 parameters) --------------------
    this->declare_parameter<std::string>("out_matrix_path", "config/out_matrix.yaml");
    this->declare_parameter<std::string>("camera_params_path", "config/camera_params.yaml");
    this->declare_parameter<std::string>("region_points_path", "config/RM2025_Points.yaml");

    this->declare_parameter<bool>("is_test", false);
    this->declare_parameter<bool>("uav_publish_default_output", true);

    this->declare_parameter<double>("uav_default_out_blue_x", 0.0);
    this->declare_parameter<double>("uav_default_out_blue_y", 0.0);
    this->declare_parameter<double>("uav_default_out_blue_z", 0.0);
    this->declare_parameter<double>("uav_default_out_red_x", 0.0);
    this->declare_parameter<double>("uav_default_out_red_y", 0.0);
    this->declare_parameter<double>("uav_default_out_red_z", 0.0);

    // minimap path (optional)
    this->declare_parameter<std::string>("minimap_path", "config/RM2025.png");

    const std::string out_matrix_path   = this->get_parameter("out_matrix_path").as_string();
    const std::string camera_params_path = this->get_parameter("camera_params_path").as_string();
    const std::string region_points_path = this->get_parameter("region_points_path").as_string();
    const std::string minimap_path       = this->get_parameter("minimap_path").as_string();

    is_test_ = this->get_parameter("is_test").as_bool();
    uav_publish_default_output_ =
        this->get_parameter("uav_publish_default_output").as_bool();

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
    pub_uav_intersection_debug_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/uav_intersection_debug", rclcpp::SensorDataQoS());
    pub_uav_cluster_debug_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/uav_cluster_debug", rclcpp::SensorDataQoS());

    sub_uav_other_cloud_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar_other", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::UavOtherCloudCallback, this,
                      std::placeholders::_1));

    // params
    this->declare_parameter<int>("uav_other_timeout_ms", 500);
    this->declare_parameter<double>("uav_default_z", 2.1);
    this->declare_parameter<double>("uav_huber_delta_xy", 0.8);
    this->declare_parameter<double>("uav_w_vision_xy", 0.5);  // 降低相机权重(透视变换外推不准)
    this->declare_parameter<double>("uav_w_lidar_xy", 0.7);   // 提高雷达权重
    this->declare_parameter<double>("uav_assign_gate_m", 5.0);
    this->declare_parameter<double>("uav_bbox_roi_x_ratio", 0.30);
    this->declare_parameter<double>("uav_bbox_roi_y_ratio", 0.00);
    this->declare_parameter<double>("uav_bbox_roi_w_ratio", 0.40);
    this->declare_parameter<double>("uav_bbox_roi_h_ratio", 0.40);
    this->declare_parameter<double>("uav_proj_pixel_margin", 6.0);
    this->declare_parameter<double>("uav_point_min_z", 1.2);
    this->declare_parameter<double>("uav_point_max_z", 4.5);
    this->declare_parameter<double>("uav_cluster_tolerance", 0.60);
    this->declare_parameter<int>("uav_cluster_min_size", 6);
    this->declare_parameter<int>("uav_cluster_max_size", 2000);
    this->declare_parameter<double>("uav_cluster_score_dist_weight", 0.02);
    this->declare_parameter<double>("uav_cluster_score_ray_weight", 1.0);
    this->declare_parameter<double>("uav_cluster_score_spread_weight", 0.6);
    this->declare_parameter<double>("uav_cluster_score_size_weight", 0.08);
    this->declare_parameter<double>("uav_center_ray_min_depth", 0.2);
    this->declare_parameter<double>("uav_rep_ray_weight", 1.0);
    this->declare_parameter<double>("uav_rep_depth_weight", 0.8);
    this->declare_parameter<double>("uav_quality_ray_sigma", 0.45);
    this->declare_parameter<double>("uav_quality_spread_sigma", 0.25);
    this->declare_parameter<double>("uav_quality_size_ref", 12.0);
    this->declare_parameter<double>("uav_quality_measurement_threshold", 0.60);
    this->declare_parameter<double>("uav_quality_z_prior_threshold", 0.75);
    this->declare_parameter<double>(
        "uav_world_to_field_y_offset",
        std::numeric_limits<double>::quiet_NaN());
    this->declare_parameter<double>("uav_field_min_x", 0.0);
    this->declare_parameter<double>("uav_field_max_x", 28.0);
    this->declare_parameter<double>("uav_field_min_y", 0.0);
    this->declare_parameter<double>("uav_field_max_y", 15.0);
    this->declare_parameter<double>("uav_field_margin_m", 0.20);

    this->declare_parameter<double>("uav_z_ema_alpha", 0.90);
    this->declare_parameter<double>("uav_z_jump_max", 0.80);
    this->declare_parameter<double>("uav_z_prior_alpha", 0.90);
    this->declare_parameter<int>("uav_z_prior_window_ms", 1200);
    this->declare_parameter<int>("uav_z_prior_update_min_interval_ms", 250);

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
    uav_bbox_roi_x_ratio_ =
        this->get_parameter("uav_bbox_roi_x_ratio").as_double();
    uav_bbox_roi_y_ratio_ =
        this->get_parameter("uav_bbox_roi_y_ratio").as_double();
    uav_bbox_roi_w_ratio_ =
        this->get_parameter("uav_bbox_roi_w_ratio").as_double();
    uav_bbox_roi_h_ratio_ =
        this->get_parameter("uav_bbox_roi_h_ratio").as_double();
    uav_proj_pixel_margin_ =
        this->get_parameter("uav_proj_pixel_margin").as_double();
    uav_point_min_z_ = this->get_parameter("uav_point_min_z").as_double();
    uav_point_max_z_ = this->get_parameter("uav_point_max_z").as_double();
    uav_cluster_tolerance_ =
        this->get_parameter("uav_cluster_tolerance").as_double();
    uav_cluster_min_size_ =
        this->get_parameter("uav_cluster_min_size").as_int();
    uav_cluster_max_size_ =
        this->get_parameter("uav_cluster_max_size").as_int();
    uav_cluster_score_dist_weight_ =
        this->get_parameter("uav_cluster_score_dist_weight").as_double();
    uav_cluster_score_ray_weight_ =
        this->get_parameter("uav_cluster_score_ray_weight").as_double();
    uav_cluster_score_spread_weight_ =
        this->get_parameter("uav_cluster_score_spread_weight").as_double();
    uav_cluster_score_size_weight_ =
        this->get_parameter("uav_cluster_score_size_weight").as_double();
    uav_center_ray_min_depth_ =
        this->get_parameter("uav_center_ray_min_depth").as_double();
    uav_rep_ray_weight_ =
        this->get_parameter("uav_rep_ray_weight").as_double();
    uav_rep_depth_weight_ =
        this->get_parameter("uav_rep_depth_weight").as_double();
    uav_quality_ray_sigma_ =
        this->get_parameter("uav_quality_ray_sigma").as_double();
    uav_quality_spread_sigma_ =
        this->get_parameter("uav_quality_spread_sigma").as_double();
    uav_quality_size_ref_ =
        this->get_parameter("uav_quality_size_ref").as_double();
    uav_quality_measurement_threshold_ =
        this->get_parameter("uav_quality_measurement_threshold").as_double();
    uav_quality_z_prior_threshold_ =
        this->get_parameter("uav_quality_z_prior_threshold").as_double();
    const double uav_world_to_field_y_offset_param =
        this->get_parameter("uav_world_to_field_y_offset").as_double();
    uav_field_min_x_ = this->get_parameter("uav_field_min_x").as_double();
    uav_field_max_x_ = this->get_parameter("uav_field_max_x").as_double();
    uav_field_min_y_ = this->get_parameter("uav_field_min_y").as_double();
    uav_field_max_y_ = this->get_parameter("uav_field_max_y").as_double();
    uav_field_margin_m_ = this->get_parameter("uav_field_margin_m").as_double();

    uav_z_ema_alpha_ = this->get_parameter("uav_z_ema_alpha").as_double();
    uav_z_jump_max_ = this->get_parameter("uav_z_jump_max").as_double();
    uav_z_prior_alpha_ = this->get_parameter("uav_z_prior_alpha").as_double();
    uav_z_prior_window_ms_ = this->get_parameter("uav_z_prior_window_ms").as_int();
    uav_z_prior_update_min_interval_ms_ =
        this->get_parameter("uav_z_prior_update_min_interval_ms").as_int();
    if (uav_z_ema_alpha_ < 0.0)
        uav_z_ema_alpha_ = 0.0;
    if (uav_z_ema_alpha_ > 0.99)
        uav_z_ema_alpha_ = 0.99;
    if (uav_z_jump_max_ < 0.0)
        uav_z_jump_max_ = 0.0;
    uav_cluster_score_ray_weight_ = std::max(0.0, uav_cluster_score_ray_weight_);
    uav_cluster_score_spread_weight_ = std::max(0.0, uav_cluster_score_spread_weight_);
    uav_cluster_score_size_weight_ = std::max(0.0, uav_cluster_score_size_weight_);
    uav_center_ray_min_depth_ = std::max(0.0, uav_center_ray_min_depth_);
    uav_rep_ray_weight_ = std::max(0.0, uav_rep_ray_weight_);
    uav_rep_depth_weight_ = std::max(0.0, uav_rep_depth_weight_);
    uav_quality_ray_sigma_ = std::max(0.05, uav_quality_ray_sigma_);
    uav_quality_spread_sigma_ = std::max(0.05, uav_quality_spread_sigma_);
    uav_quality_size_ref_ = std::max(1.0, uav_quality_size_ref_);
    uav_quality_measurement_threshold_ =
        std::clamp(uav_quality_measurement_threshold_, 0.0, 1.0);
    uav_quality_z_prior_threshold_ =
        std::clamp(uav_quality_z_prior_threshold_, 0.0, 1.0);
    if (uav_field_min_x_ > uav_field_max_x_) {
        std::swap(uav_field_min_x_, uav_field_max_x_);
    }
    if (uav_field_min_y_ > uav_field_max_y_) {
        std::swap(uav_field_min_y_, uav_field_max_y_);
    }
    uav_field_margin_m_ = std::max(0.0, uav_field_margin_m_);
    if (std::isfinite(uav_world_to_field_y_offset_param)) {
        uav_world_to_field_y_offset_ = uav_world_to_field_y_offset_param;
    } else {
        InferUavFieldTransform();
    }
    uav_z_prior_alpha_ = std::clamp(uav_z_prior_alpha_, 0.0, 0.99);
    uav_z_prior_window_ms_ = std::max(100, uav_z_prior_window_ms_);
    uav_z_prior_update_min_interval_ms_ =
        std::max(0, uav_z_prior_update_min_interval_ms_);
    uav_bbox_roi_x_ratio_ = std::clamp(uav_bbox_roi_x_ratio_, 0.0, 0.95);
    uav_bbox_roi_y_ratio_ = std::clamp(uav_bbox_roi_y_ratio_, 0.0, 0.95);
    uav_bbox_roi_w_ratio_ = std::clamp(uav_bbox_roi_w_ratio_, 0.05, 1.0);
    uav_bbox_roi_h_ratio_ = std::clamp(uav_bbox_roi_h_ratio_, 0.05, 1.0);
    if (uav_bbox_roi_x_ratio_ + uav_bbox_roi_w_ratio_ > 1.0) {
        uav_bbox_roi_x_ratio_ = std::max(0.0, 1.0 - uav_bbox_roi_w_ratio_);
    }
    if (uav_bbox_roi_y_ratio_ + uav_bbox_roi_h_ratio_ > 1.0) {
        uav_bbox_roi_y_ratio_ = std::max(0.0, 1.0 - uav_bbox_roi_h_ratio_);
    }
    if (uav_proj_pixel_margin_ < 0.0) {
        uav_proj_pixel_margin_ = 0.0;
    }
    if (uav_point_min_z_ > uav_point_max_z_) {
        std::swap(uav_point_min_z_, uav_point_max_z_);
    }
    if (uav_cluster_tolerance_ < 0.05) {
        uav_cluster_tolerance_ = 0.05;
    }
    if (uav_cluster_min_size_ < 1) {
        uav_cluster_min_size_ = 1;
    }
    if (uav_cluster_max_size_ < uav_cluster_min_size_) {
        uav_cluster_max_size_ = uav_cluster_min_size_;
    }

    uav_lost_timeout_ms_ =
        this->get_parameter("uav_lost_timeout_ms").as_int();
    uav_q_pos_ = this->get_parameter("uav_q_pos").as_double();
    uav_q_vel_ = this->get_parameter("uav_q_vel").as_double();
    uav_r_meas_xy_ = this->get_parameter("uav_r_meas_xy").as_double();
    uav_r_meas_z_ = this->get_parameter("uav_r_meas_z").as_double();
    uav_gate_maha_ = this->get_parameter("uav_gate_maha").as_double();

    uav_default_out_x_[0] =
        this->get_parameter("uav_default_out_blue_x").as_double();
    uav_default_out_y_[0] =
        this->get_parameter("uav_default_out_blue_y").as_double();
    uav_default_out_z_[0] =
        this->get_parameter("uav_default_out_blue_z").as_double();
    uav_default_out_x_[1] =
        this->get_parameter("uav_default_out_red_x").as_double();
    uav_default_out_y_[1] =
        this->get_parameter("uav_default_out_red_y").as_double();
    uav_default_out_z_[1] =
        this->get_parameter("uav_default_out_red_z").as_double();
    for (int idx = 0; idx < 2; ++idx) {
        uav_z_prior_[idx] = ClampUavZ(uav_default_z_);
        uav_z_prior_inited_[idx] = true;
    }

    RCLCPP_INFO(this->get_logger(),
                "[INIT] UAV frame: field=(camera_world.x, camera_world.y + %.3f), bounds x=[%.2f, %.2f] y=[%.2f, %.2f], margin=%.2f",
                uav_world_to_field_y_offset_,
                uav_field_min_x_, uav_field_max_x_,
                uav_field_min_y_, uav_field_max_y_,
                uav_field_margin_m_);
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
    p.y = center_point.y + static_cast<float>(uav_world_to_field_y_offset_);
    p.z = 1.0;
    cloud->points.push_back(p);

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "rm_frame";
    pub->publish(output);
}

void Resolve::UavOtherCloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
        std::lock_guard<std::mutex> lk(uav_other_mutex_);
        uav_other_points_.clear();
        uav_other_stamp_ = is_stamp_zero(msg->header.stamp)
                               ? this->now()
                               : rclcpp::Time(msg->header.stamp);
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::lock_guard<std::mutex> lk(uav_other_mutex_);
    uav_other_points_.clear();
    uav_other_points_.reserve(cloud.points.size());

    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        uav_other_points_.push_back(p);
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

cv::Rect2f Resolve::BuildUavRoiFromBbox(float x, float y, float w, float h) const
{
    if (!(w > 1.0f && h > 1.0f)) {
        return cv::Rect2f();
    }

    const float roi_x = x + static_cast<float>(uav_bbox_roi_x_ratio_) * w;
    const float roi_y = y + static_cast<float>(uav_bbox_roi_y_ratio_) * h;
    const float roi_w = std::max(1.0f, static_cast<float>(uav_bbox_roi_w_ratio_) * w);
    const float roi_h = std::max(1.0f, static_cast<float>(uav_bbox_roi_h_ratio_) * h);
    return cv::Rect2f(roi_x, roi_y, roi_w, roi_h);
}

double Resolve::ClampUavZ(double z) const
{
    return std::clamp(z, uav_point_min_z_, uav_point_max_z_);
}

void Resolve::InferUavFieldTransform()
{
    uav_world_to_field_y_offset_ = 15.0;
    if (parser_ == nullptr || parser_->points_map.empty()) {
        return;
    }

    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& kv : parser_->points_map) {
        if (kv.second == nullptr) {
            continue;
        }
        for (const auto& point : kv.second->Points_3D) {
            min_y = std::min(min_y, static_cast<double>(point.y));
            max_y = std::max(max_y, static_cast<double>(point.y));
        }
    }

    if (!std::isfinite(min_y) || !std::isfinite(max_y)) {
        return;
    }

    // RM2025 calibration points use y in [-15, 0], while some newer maps
    // already use the public rm_frame y axis. Keep the old +15 transform only
    // for the negative-y calibration convention.
    if (min_y < -1.0 && max_y <= 1.0) {
        uav_world_to_field_y_offset_ = std::max(0.0, uav_field_max_y_);
    } else {
        uav_world_to_field_y_offset_ = 0.0;
    }
}

cv::Point3f Resolve::UavFieldToCameraWorld(const pcl::PointXYZ& point) const
{
    return cv::Point3f(point.x,
                       static_cast<float>(point.y - uav_world_to_field_y_offset_),
                       point.z);
}

cv::Vec3d Resolve::UavCameraWorldToField(const cv::Vec3d& point) const
{
    return cv::Vec3d(point[0],
                     point[1] + uav_world_to_field_y_offset_,
                     point[2]);
}

bool Resolve::IsUavFieldPointInside(double x, double y) const
{
    const double margin = std::max(0.0, uav_field_margin_m_);
    return x >= uav_field_min_x_ - margin &&
           x <= uav_field_max_x_ + margin &&
           y >= uav_field_min_y_ - margin &&
           y <= uav_field_max_y_ + margin;
}

bool Resolve::IntersectUavPixelWithZ(const cv::Point2f& pixel,
                                     double z_plane,
                                     cv::Vec3d& point_field,
                                     cv::Vec3d* point_camera_world) const
{
    cv::Vec3d origin_world;
    cv::Vec3d dir_world;
    if (!BuildUavCenterRay(pixel, origin_world, dir_world)) {
        return false;
    }

    if (std::abs(dir_world[2]) < 1e-9) {
        return false;
    }

    const double depth = (z_plane - origin_world[2]) / dir_world[2];
    if (depth <= 1e-6 || !std::isfinite(depth)) {
        return false;
    }

    const cv::Vec3d point_world = origin_world + dir_world * depth;
    if (point_camera_world != nullptr) {
        *point_camera_world = point_world;
    }
    point_field = UavCameraWorldToField(point_world);
    return std::isfinite(point_field[0]) &&
           std::isfinite(point_field[1]) &&
           std::isfinite(point_field[2]);
}

bool Resolve::BuildUavCenterRay(const cv::Point2f& pixel,
                                cv::Vec3d& origin_world,
                                cv::Vec3d& dir_world) const
{
    if (parser_ == nullptr) {
        return false;
    }

    cv::Vec3d cam_dir_vec;
    if (!pixelToCameraRay(pixel,
                          parser_->camera_matrix,
                          parser_->dist_coeffs,
                          cam_dir_vec)) {
        return false;
    }

    cv::Mat rot_mat;
    cv::Rodrigues(parser_->world_rvec, rot_mat);

    cv::Mat rot64;
    cv::Mat tvec64 = parser_->world_tvec;
    rot_mat.convertTo(rot64, CV_64F);
    tvec64.convertTo(tvec64, CV_64F);
    if (tvec64.rows == 1 && tvec64.cols == 3) {
        tvec64 = tvec64.t();
    }
    if (tvec64.rows != 3 || tvec64.cols != 1) {
        return false;
    }

    const cv::Mat cam_dir =
        (cv::Mat_<double>(3, 1) << cam_dir_vec[0], cam_dir_vec[1], cam_dir_vec[2]);
    const cv::Mat world_dir_mat = rot64.t() * cam_dir;
    const cv::Mat cam_origin_mat = -rot64.t() * tvec64;

    dir_world = cv::Vec3d(world_dir_mat.at<double>(0, 0),
                          world_dir_mat.at<double>(1, 0),
                          world_dir_mat.at<double>(2, 0));
    const double dir_norm = cv::norm(dir_world);
    if (dir_norm <= 1e-6) {
        return false;
    }
    dir_world /= dir_norm;

    origin_world = cv::Vec3d(cam_origin_mat.at<double>(0, 0),
                             cam_origin_mat.at<double>(1, 0),
                             cam_origin_mat.at<double>(2, 0));
    return true;
}

double Resolve::EvaluateUavMeasurementQuality(double conf,
                                              const UavFrustumMeasurement& meas) const
{
    if (!meas.valid || !std::isfinite(meas.ray_dist_median) ||
        !std::isfinite(meas.ray_dist_spread) || meas.cluster_size <= 0) {
        return 0.0;
    }

    const double q_vis = std::clamp(static_cast<double>(conf), 0.0, 1.0);
    const double q_ray =
        std::exp(-std::pow(meas.ray_dist_median / uav_quality_ray_sigma_, 2.0));
    const double q_spread =
        std::exp(-std::pow(meas.ray_dist_spread / uav_quality_spread_sigma_, 2.0));
    const double q_size =
        std::clamp(static_cast<double>(meas.cluster_size) / uav_quality_size_ref_,
                   0.0, 1.0);
    return 0.35 * q_vis + 0.35 * q_ray + 0.15 * q_size + 0.15 * q_spread;
}

double Resolve::GetUavFallbackZ(int idx) const
{
    if (idx >= 0 && idx < 2 && uav_z_prior_inited_[idx]) {
        return ClampUavZ(uav_z_prior_[idx]);
    }
    return ClampUavZ(uav_default_z_);
}

void Resolve::UpdateUavZPrior(int idx, double z_raw, const rclcpp::Time& stamp)
{
    if (idx < 0 || idx >= 2) {
        return;
    }

    const rclcpp::Time safe_stamp =
        (stamp.nanoseconds() == 0) ? this->now() : stamp;
    const double z = ClampUavZ(z_raw);

    if (!uav_z_prior_inited_[idx]) {
        uav_z_prior_[idx] = z;
        uav_z_prior_inited_[idx] = true;
    }

    if (uav_z_prior_last_update_[idx].nanoseconds() != 0) {
        const double dt_ms =
            std::fabs((safe_stamp - uav_z_prior_last_update_[idx]).seconds()) * 1000.0;
        if (dt_ms < static_cast<double>(uav_z_prior_update_min_interval_ms_)) {
            return;
        }
    }

    auto& samples = uav_z_prior_samples_[idx];
    samples.emplace_back(safe_stamp, z);

    while (!samples.empty()) {
        const double age_ms =
            (safe_stamp - samples.front().first).seconds() * 1000.0;
        if (age_ms <= static_cast<double>(uav_z_prior_window_ms_)) {
            break;
        }
        samples.pop_front();
    }

    std::vector<double> z_values;
    z_values.reserve(samples.size());
    for (const auto& sample : samples) {
        z_values.push_back(sample.second);
    }
    const double z_median = compute_median(std::move(z_values));
    uav_z_prior_[idx] =
        ClampUavZ(uav_z_prior_alpha_ * uav_z_prior_[idx] +
                  (1.0 - uav_z_prior_alpha_) * z_median);
    uav_z_prior_last_update_[idx] = safe_stamp;
}

bool Resolve::SolveUavFrustumMeasurement(
    const cv::Rect2f& roi,
    const std::vector<pcl::PointXYZ>& points,
    UavFrustumMeasurement& out) const
{
    out = UavFrustumMeasurement{};
    out.input_size = static_cast<int>(points.size());
    if (roi.width <= 1.0f || roi.height <= 1.0f || points.empty() || parser_ == nullptr) {
        return false;
    }

    const cv::Point2f roi_center(roi.x + roi.width * 0.5f,
                                 roi.y + roi.height * 0.5f);
    cv::Vec3d ray_origin_world;
    cv::Vec3d ray_dir_world;
    if (!BuildUavCenterRay(roi_center, ray_origin_world, ray_dir_world)) {
        return false;
    }

    std::vector<cv::Point3f> world_points;
    std::vector<pcl::PointXYZ> valid_points;
    world_points.reserve(points.size());
    valid_points.reserve(points.size());
    for (const auto& p : points) {
        if (p.z < uav_point_min_z_ || p.z > uav_point_max_z_) {
            continue;
        }
        valid_points.push_back(p);
        world_points.push_back(UavFieldToCameraWorld(p));
    }
    out.z_valid_size = static_cast<int>(valid_points.size());
    if (world_points.empty()) {
        return false;
    }

    std::vector<cv::Point2f> image_points;
    projectPointsCameraModel(world_points,
                             parser_->world_rvec,
                             parser_->world_tvec,
                             parser_->camera_matrix,
                             parser_->dist_coeffs,
                             image_points);

    const float margin = static_cast<float>(std::max(0.0, uav_proj_pixel_margin_));
    const cv::Rect2f expanded_roi(
        roi.x - margin,
        roi.y - margin,
        roi.width + 2.0f * margin,
        roi.height + 2.0f * margin);

    pcl::PointCloud<pcl::PointXYZ>::Ptr intersected(new pcl::PointCloud<pcl::PointXYZ>);
    intersected->reserve(world_points.size());
    std::vector<cv::Point3f> intersected_world;
    std::vector<cv::Point2f> intersected_uvs;
    intersected_world.reserve(world_points.size());
    intersected_uvs.reserve(world_points.size());

    for (size_t i = 0; i < valid_points.size(); ++i) {
        const auto& p = valid_points[i];
        const auto& uv = image_points[i];
        if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            continue;
        }
        if (!expanded_roi.contains(uv)) {
            continue;
        }
        intersected->push_back(p);
        intersected_world.push_back(world_points[i]);
        intersected_uvs.push_back(uv);
        out.debug_uvs.push_back(uv);
    }
    out.roi_point_size = static_cast<int>(intersected->size());

    if (intersected->size() < static_cast<size_t>(std::max(1, uav_cluster_min_size_))) {
        return false;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(intersected);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(uav_cluster_tolerance_);
    ec.setMinClusterSize(std::max(1, uav_cluster_min_size_));
    ec.setMaxClusterSize(std::max(uav_cluster_min_size_, uav_cluster_max_size_));
    ec.setSearchMethod(tree);
    ec.setInputCloud(intersected);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);
    out.cluster_count = static_cast<int>(cluster_indices.size());
    if (cluster_indices.empty()) {
        return false;
    }

    double best_score = std::numeric_limits<double>::infinity();

    for (const auto& indices : cluster_indices) {
        std::vector<double> ray_dists;
        std::vector<double> ray_depths;
        std::vector<double> z_values;
        std::vector<int> valid_cluster_indices;
        std::vector<cv::Point2f> cluster_uvs;
        ray_dists.reserve(indices.indices.size());
        ray_depths.reserve(indices.indices.size());
        z_values.reserve(indices.indices.size());
        valid_cluster_indices.reserve(indices.indices.size());
        cluster_uvs.reserve(indices.indices.size());

        for (int idx : indices.indices) {
            const auto& world_p = intersected_world[idx];
            const cv::Vec3d point_world(world_p.x, world_p.y, world_p.z);
            const cv::Vec3d delta = point_world - ray_origin_world;
            const double ray_depth = delta.dot(ray_dir_world);
            if (ray_depth <= uav_center_ray_min_depth_) {
                continue;
            }

            const cv::Vec3d lateral = delta - ray_dir_world * ray_depth;
            const double ray_dist = cv::norm(lateral);
            if (!std::isfinite(ray_dist)) {
                continue;
            }

            ray_dists.push_back(ray_dist);
            ray_depths.push_back(ray_depth);
            z_values.push_back(intersected->points[idx].z);
            valid_cluster_indices.push_back(idx);
            cluster_uvs.push_back(intersected_uvs[idx]);
        }

        if (valid_cluster_indices.empty()) {
            continue;
        }

        const double ray_dist_median = compute_median(ray_dists);
        const double ray_dist_spread = compute_mad(ray_dists, ray_dist_median);
        const double depth_median = compute_median(ray_depths);
        const double depth_spread = compute_mad(ray_depths, depth_median);
        const double cluster_z_median = compute_median(z_values);
        const double cluster_size_term =
            std::min(static_cast<double>(valid_cluster_indices.size()),
                     uav_quality_size_ref_);
        const double score =
            ray_dist_median * uav_cluster_score_ray_weight_ +
            ray_dist_spread * uav_cluster_score_spread_weight_ -
            cluster_size_term * uav_cluster_score_size_weight_;

        if (score < best_score) {
            const double ray_scale =
                std::max(0.05, ray_dist_median + std::max(0.02, ray_dist_spread));
            const double depth_scale =
                std::max(0.10, depth_spread + 0.05);

            double best_point_cost = std::numeric_limits<double>::infinity();
            pcl::PointXYZ best_point{};
            for (size_t i = 0; i < valid_cluster_indices.size(); ++i) {
                const double point_cost =
                    uav_rep_ray_weight_ * (ray_dists[i] / ray_scale) +
                    uav_rep_depth_weight_ *
                        (std::fabs(ray_depths[i] - depth_median) / depth_scale);
                if (point_cost < best_point_cost) {
                    best_point_cost = point_cost;
                    best_point = intersected->points[valid_cluster_indices[i]];
                }
            }

            best_score = score;
            out.valid = true;
            out.x = best_point.x;
            out.y = best_point.y;
            out.z = best_point.z;
            out.cluster_z_median = cluster_z_median;
            out.ray_dist_median = ray_dist_median;
            out.ray_dist_spread = ray_dist_spread;
            out.depth_median = depth_median;
            out.cluster_size = static_cast<int>(valid_cluster_indices.size());
            out.cluster_uvs = std::move(cluster_uvs);
        }
    }

    return out.valid;
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
    for (int i = 0; i < 7; ++i) {
        send_data.blue_car_box_x1[i] = msg->blue_car_box_x1[i];
        send_data.blue_car_box_y1[i] = msg->blue_car_box_y1[i];
        send_data.blue_car_box_x2[i] = msg->blue_car_box_x2[i];
        send_data.blue_car_box_y2[i] = msg->blue_car_box_y2[i];
        send_data.red_car_box_x1[i] = msg->red_car_box_x1[i];
        send_data.red_car_box_y1[i] = msg->red_car_box_y1[i];
        send_data.red_car_box_x2[i] = msg->red_car_box_x2[i];
        send_data.red_car_box_y2[i] = msg->red_car_box_y2[i];
    }

    geometry_msgs::msg::PoseArray uav_out;
    uav_out.header = msg->header;
    uav_out.header.frame_id = "rm_frame";
    if (is_stamp_zero(uav_out.header.stamp)) {
        uav_out.header.stamp = this->now();
    }
    uav_out.poses.resize(2);

    auto set_zero_pose = [](geometry_msgs::msg::Pose& p) {
        p.position.x = 0.0;
        p.position.y = 0.0;
        p.position.z = 0.0;
        p.orientation.x = 0.0;
        p.orientation.y = 0.0;
        p.orientation.z = 0.0;
        p.orientation.w = 1.0;
    };
    auto set_default_pose = [&](int idx, geometry_msgs::msg::Pose& p) {
        p.position.x = uav_default_out_x_[idx];
        p.position.y = uav_default_out_y_[idx];
        p.position.z = uav_default_out_z_[idx];
        p.orientation.x = 0.0;
        p.orientation.y = 0.0;
        p.orientation.z = 0.0;
        p.orientation.w = 1.0;
    };

    if (uav_publish_default_output_) {
        set_default_pose(0, uav_out.poses[0]);
        set_default_pose(1, uav_out.poses[1]);
    } else {
        set_zero_pose(uav_out.poses[0]);
        set_zero_pose(uav_out.poses[1]);
    }

    // ========== 阶段一：接收检测消息 ==========
    const bool has_blue_bbox =
        (msg->blue_uav_bbox_w > 1.0f && msg->blue_uav_bbox_h > 1.0f);
    const bool has_red_bbox =
        (msg->red_uav_bbox_w > 1.0f && msg->red_uav_bbox_h > 1.0f);
    const bool has_blue_uav =
        has_blue_bbox || (msg->blue_x[6] != 0.0f && msg->blue_y[6] != 0.0f);
    const bool has_red_uav =
        has_red_bbox || (msg->red_x[6] != 0.0f && msg->red_y[6] != 0.0f);
    
    if (has_blue_uav || has_red_uav) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段一：接收检测消息] ==========");
        if (has_blue_uav) {
            RCLCPP_INFO(this->get_logger(),
                        "  蓝方UAV观测: center=(%.2f, %.2f), bbox=(%.2f, %.2f, %.2f, %.2f), conf=%.2f",
                        msg->blue_x[6], msg->blue_y[6],
                        msg->blue_uav_bbox_x, msg->blue_uav_bbox_y,
                        msg->blue_uav_bbox_w, msg->blue_uav_bbox_h,
                        msg->blue_uav_conf);
        } else {
            RCLCPP_INFO(this->get_logger(), "  蓝方UAV: 无检测");
        }
        if (has_red_uav) {
            RCLCPP_INFO(this->get_logger(),
                        "  红方UAV观测: center=(%.2f, %.2f), bbox=(%.2f, %.2f, %.2f, %.2f), conf=%.2f",
                        msg->red_x[6], msg->red_y[6],
                        msg->red_uav_bbox_x, msg->red_uav_bbox_y,
                        msg->red_uav_bbox_w, msg->red_uav_bbox_h,
                        msg->red_uav_conf);
        } else {
            RCLCPP_INFO(this->get_logger(), "  红方UAV: 无检测");
        }
    }

    // ========== 阶段二：读取原始 lidar_other 点云 ==========
    std::vector<pcl::PointXYZ> other_points;
    bool other_fresh = false;
    const rclcpp::Time detect_stamp = is_stamp_zero(msg->header.stamp)
                                          ? this->now()
                                          : rclcpp::Time(msg->header.stamp);

    {
        std::lock_guard<std::mutex> lk(uav_other_mutex_);
        other_points = uav_other_points_;

        const bool has_other_stamp = (uav_other_stamp_.nanoseconds() != 0);
        long long dt_ms = -1;
        if (has_other_stamp) {
            const auto dt_ns = (detect_stamp - uav_other_stamp_).nanoseconds();
            dt_ms = std::llabs(dt_ns) / 1000000LL;
        }
        other_fresh = has_other_stamp && (dt_ms <= uav_other_timeout_ms_);

        if (has_blue_uav || has_red_uav) {
            RCLCPP_INFO(this->get_logger(), "========== [阶段二：读取 lidar_other 原始点云] ==========");
            if (has_other_stamp) {
                RCLCPP_INFO(this->get_logger(),
                            "  点云数量: %zu, 时间差: %lld ms, 数据新鲜: %s",
                            other_points.size(), dt_ms, other_fresh ? "是" : "否");
            } else {
                RCLCPP_WARN(this->get_logger(),
                            "  尚未收到 /livox/lidar_other，无法计算点云-图像时间差");
            }

            if (other_points.empty()) {
                RCLCPP_WARN(this->get_logger(), "  ⚠️ 无可用 lidar_other 点");
            } else {
                RCLCPP_INFO(this->get_logger(),
                            "  lidar_other 使用原始点云做视锥体投影筛点，聚类将在解算阶段完成");
            }
        }
    }

    // Lambda: pure vision fallback uses the same calibrated ray as frustum scoring.
    auto solve_xy_on_z = [&](float px,
                             float py,
                             float z_plane,
                             cv::Vec3d& point_field) -> bool {
        return IntersectUavPixelWithZ(cv::Point2f(px, py),
                                      static_cast<double>(z_plane),
                                      point_field);
    };
    // ========== 阶段三：视锥体求交 + 聚类中心 ==========
    bool   meas_ok[2]{false, false};
    bool   meas_from_lidar[2]{false, false};
    double meas_quality[2]{0.0, 0.0};
    double meas_x[2]{0, 0}, meas_y[2]{0, 0}, meas_z[2]{0, 0};
    std::vector<cv::Point2f> debug_uvs_all;
    std::vector<cv::Point2f> debug_cluster_uvs_all;
    debug_uvs_all.reserve(256);
    debug_cluster_uvs_all.reserve(128);

    auto compute_meas = [&](int idx,
                            bool has_target,
                            bool has_bbox,
                            float px,
                            float py,
                            float bbox_x,
                            float bbox_y,
                            float bbox_w,
                            float bbox_h,
                            float bbox_conf) {
        const char* side_name = (idx == 0) ? "蓝方" : "红方";
        if (!has_target) {
            return;
        }

        const cv::Rect2f roi = has_bbox
            ? BuildUavRoiFromBbox(bbox_x, bbox_y, bbox_w, bbox_h)
            : cv::Rect2f();

        if (has_bbox && other_fresh && !other_points.empty()) {
            bool has_cluster_candidate = false;
            UavFrustumMeasurement frustum_meas;
            if (SolveUavFrustumMeasurement(roi, other_points, frustum_meas)) {
                has_cluster_candidate = true;
                debug_uvs_all.insert(debug_uvs_all.end(),
                                     frustum_meas.debug_uvs.begin(),
                                     frustum_meas.debug_uvs.end());
                debug_cluster_uvs_all.insert(debug_cluster_uvs_all.end(),
                                             frustum_meas.cluster_uvs.begin(),
                                             frustum_meas.cluster_uvs.end());

                const double quality =
                    EvaluateUavMeasurementQuality(bbox_conf, frustum_meas);
                if (quality >= uav_quality_measurement_threshold_) {
                    meas_ok[idx] = true;
                    meas_from_lidar[idx] = true;
                    meas_quality[idx] = quality;
                    meas_x[idx] = frustum_meas.x;
                    meas_y[idx] = frustum_meas.y;
                    meas_z[idx] = GetFilteredZPlane(idx, frustum_meas.z);

                    if (quality >= uav_quality_z_prior_threshold_) {
                        UpdateUavZPrior(idx, frustum_meas.cluster_z_median, detect_stamp);
                    }

                    const cv::Point2f roi_center(
                        roi.x + roi.width * 0.5f,
                        roi.y + roi.height * 0.5f);
                    cv::Vec3d pv(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
                    (void)solve_xy_on_z(roi_center.x,
                                        roi_center.y,
                                        static_cast<float>(meas_z[idx]),
                                        pv);

                    RCLCPP_INFO(
                        this->get_logger(),
                        "  %s: 视锥求交成功 roi=(%.1f, %.1f, %.1f, %.1f) -> lidar=(%.3f, %.3f, %.3f), q=%.3f, ray_med=%.3f, ray_spread=%.3f, n=%d, z_prior=%.3f, vision@z=(%.3f, %.3f)",
                        side_name,
                        roi.x, roi.y, roi.width, roi.height,
                        meas_x[idx], meas_y[idx], meas_z[idx],
                        quality,
                        frustum_meas.ray_dist_median,
                        frustum_meas.ray_dist_spread,
                        frustum_meas.cluster_size,
                        GetUavFallbackZ(idx),
                        pv[0], pv[1]);
                    return;
                }

                RCLCPP_WARN(
                    this->get_logger(),
                    "  %s: 视锥求交得到点簇但质量不足 q=%.3f < %.3f (ray_med=%.3f, ray_spread=%.3f, n=%d, conf=%.2f)，转入纯视觉 fallback",
                    side_name,
                    quality,
                    uav_quality_measurement_threshold_,
                    frustum_meas.ray_dist_median,
                    frustum_meas.ray_dist_spread,
                    frustum_meas.cluster_size,
                    bbox_conf);
            }

            if (!has_cluster_candidate) {
                RCLCPP_WARN(this->get_logger(),
                            "  %s: ROI 视锥求交未得到有效聚类 input=%d, z_ok=%d, roi_ok=%d, clusters=%d, min_cluster=%d，转入纯视觉 fallback",
                            side_name,
                            frustum_meas.input_size,
                            frustum_meas.z_valid_size,
                            frustum_meas.roi_point_size,
                            frustum_meas.cluster_count,
                            uav_cluster_min_size_);
            }
        }

        if (px != 0.0f && py != 0.0f) {
            const float z_plane = static_cast<float>(GetUavFallbackZ(idx));
            cv::Vec3d pv;
            if (!solve_xy_on_z(px, py, z_plane, pv)) {
                RCLCPP_WARN(this->get_logger(),
                            "  %s: 纯视觉 fallback center=(%.1f, %.1f) z_prior=%.2f 射线无法与高度平面求交",
                            side_name, px, py, z_plane);
                return;
            }

            if (!IsUavFieldPointInside(pv[0], pv[1])) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "  %s: 纯视觉 fallback center=(%.1f, %.1f) z_prior=%.2f -> raw=(%.3f, %.3f, %.3f) 场外，丢弃该视觉测量",
                    side_name, px, py, z_plane,
                    pv[0], pv[1], pv[2]);
                return;
            }

            meas_ok[idx] = true;
            meas_x[idx] = pv[0];
            meas_y[idx] = pv[1];
            meas_z[idx] = z_plane;

            RCLCPP_WARN(this->get_logger(),
                        "  %s: 纯视觉 fallback center=(%.1f, %.1f) z_prior=%.2f -> (%.3f, %.3f, %.3f)",
                        side_name, px, py, z_plane,
                        meas_x[idx], meas_y[idx], meas_z[idx]);
        }
    };

    if (has_blue_uav || has_red_uav) {
        RCLCPP_INFO(this->get_logger(), "========== [阶段三：视锥体求交测量] ==========");
    }
    compute_meas(0, has_blue_uav, has_blue_bbox,
                 msg->blue_x[6], msg->blue_y[6],
                 msg->blue_uav_bbox_x, msg->blue_uav_bbox_y,
                 msg->blue_uav_bbox_w, msg->blue_uav_bbox_h,
                 msg->blue_uav_conf);
    compute_meas(1, has_red_uav, has_red_bbox,
                 msg->red_x[6], msg->red_y[6],
                 msg->red_uav_bbox_x, msg->red_uav_bbox_y,
                 msg->red_uav_bbox_w, msg->red_uav_bbox_h,
                 msg->red_uav_conf);

    {
        pcl::PointCloud<pcl::PointXYZ> debug_cloud;
        debug_cloud.points.reserve(debug_uvs_all.size());
        for (const auto& uv : debug_uvs_all) {
            pcl::PointXYZ p;
            p.x = uv.x;
            p.y = uv.y;
            p.z = 0.0f;
            debug_cloud.points.push_back(p);
        }

        sensor_msgs::msg::PointCloud2 debug_msg;
        pcl::toROSMsg(debug_cloud, debug_msg);
        debug_msg.header = msg->header;
        debug_msg.header.frame_id = "camera_image";
        if (is_stamp_zero(debug_msg.header.stamp)) {
            debug_msg.header.stamp = this->now();
        }
        pub_uav_intersection_debug_->publish(debug_msg);
    }
    {
        pcl::PointCloud<pcl::PointXYZ> debug_cloud;
        debug_cloud.points.reserve(debug_cluster_uvs_all.size());
        for (const auto& uv : debug_cluster_uvs_all) {
            pcl::PointXYZ p;
            p.x = uv.x;
            p.y = uv.y;
            p.z = 0.0f;
            debug_cloud.points.push_back(p);
        }

        sensor_msgs::msg::PointCloud2 debug_msg;
        pcl::toROSMsg(debug_cloud, debug_msg);
        debug_msg.header = msg->header;
        debug_msg.header.frame_id = "camera_image";
        if (is_stamp_zero(debug_msg.header.stamp)) {
            debug_msg.header.stamp = this->now();
        }
        pub_uav_cluster_debug_->publish(debug_msg);
    }

    // ========== 阶段四：卡尔曼滤波预测与更新 ==========
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
            RCLCPP_INFO(this->get_logger(), "  蓝方UAV输出: (%.3f, %.3f, %.3f) [%s/%s q=%.3f z_prior=%.3f]", 
                out_x, out_y, out_z, uav_kf_inited_[0] ? "KF" : "测量",
                meas_from_lidar[0] ? "激光" : "视觉",
                meas_quality[0], GetUavFallbackZ(0));
        }
        if (uav_kf_inited_[1] || meas_ok[1]) {
            double out_x = uav_kf_inited_[1] ? uav_x_[1] : meas_x[1];
            double out_y = uav_kf_inited_[1] ? uav_y_[1] : meas_y[1];
            double out_z = uav_kf_inited_[1] ? uav_z_[1] : meas_z[1];
            RCLCPP_INFO(this->get_logger(), "  红方UAV输出: (%.3f, %.3f, %.3f) [%s/%s q=%.3f z_prior=%.3f]", 
                out_x, out_y, out_z, uav_kf_inited_[1] ? "KF" : "测量",
                meas_from_lidar[1] ? "激光" : "视觉",
                meas_quality[1], GetUavFallbackZ(1));
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
                    float y_offset = cp.y + static_cast<float>(uav_world_to_field_y_offset_);
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
                    float y_offset = cp.y + static_cast<float>(uav_world_to_field_y_offset_);
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
    
//     // ================= 临时：/uav_positions 的第2个目标改为“蓝方2号小车” =================
// // 约定：blue[1] 是蓝方2号车（数组索引 0..5 对应 1..6号车，1=>2号车）
// {
//     const int car_idx = 1;  // 蓝方2号小车
//     const float x = send_data.blue_x[car_idx];
//     const float y = send_data.blue_y[car_idx];

//     if (x != 0.0f && y != 0.0f) {
//         uav_out.poses[1].position.x = static_cast<double>(x);
//         uav_out.poses[1].position.y = static_cast<double>(y);
//         uav_out.poses[1].position.z = 0.30;   // 地面车先给0高度

//         // 如需固定姿态，可打开下面4行
//         // uav_out.poses[1].orientation.w = 1.0;
//         // uav_out.poses[1].orientation.x = 0.0;
//         // uav_out.poses[1].orientation.y = 0.0;
//         // uav_out.poses[1].orientation.z = 0.0;
//     } else {
//         // 蓝方2号车无坐标时，保持原逻辑（KF/测量/默认值）
//         // 如需强制无效，也可改成：
//         // set_nan_pose(uav_out.poses[1]);
//     }
// }


    pub_uav_positions_->publish(uav_out);
}

}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Resolve)
