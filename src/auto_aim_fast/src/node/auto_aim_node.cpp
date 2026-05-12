#include "auto_aim_fast/node/auto_aim_node.hpp"
#include "auto_aim_fast/util/time.hpp"
#include "auto_aim_fast/pipeline/frame.hpp"
#include "auto_aim_fast/tracking/target_tracker.hpp"
#include "auto_aim_fast/legacy/detector.hpp"
#include "radar_decision_comm/common/team_identity.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

// cv_bridge for publishing debug images
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace auto_aim_fast
{

static constexpr float kRad2Deg = 57.29577951308232f;
static constexpr float kDeg2Rad = 0.017453292519943295f;
static constexpr const char* kRedBold = "\033[1;31m";
static constexpr const char* kColorReset = "\033[0m";

static cv::Mat makeK(double fx, double fy, double cx, double cy)
{
  return (cv::Mat_<double>(3,3) << fx, 0.0, cx,
                                  0.0, fy, cy,
                                  0.0, 0.0, 1.0);
}

static cv::Mat makeD(const std::vector<double> & d)
{
  cv::Mat D = cv::Mat::zeros(1, static_cast<int>(d.size()), CV_64F);
  for (int i = 0; i < static_cast<int>(d.size()); ++i) {
    D.at<double>(0, i) = d[i];
  }
  return D;
}

static rclcpp::QoS makeImageQoS(const rclcpp::Node & node)
{
  const bool use_sensor = node.get_parameter("qos.use_sensor_data_qos").as_bool();
  const int depth = static_cast<int>(node.get_parameter("qos.depth").as_int());
  const bool keep_all = node.get_parameter("qos.history_keep_all").as_bool();
  const auto rel = node.get_parameter("qos.reliability").as_string();

  rclcpp::QoS qos = use_sensor ? rclcpp::SensorDataQoS()
                               : rclcpp::QoS(rclcpp::KeepLast(std::max(1, depth)));

  if (keep_all) qos.keep_all();
  else qos.keep_last(std::max(1, depth));

  if (rel == "reliable") qos.reliable();
  else qos.best_effort();

  return qos;
}

static rclcpp::QoS makeInfoQoS()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
}

AutoAimNode::AutoAimNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("auto_aim_node", options)
{
  cv::setNumThreads(3);

  this->declare_parameter<std::string>("image_topic", "/camera2/image");
  this->declare_parameter<std::string>("camera_info_topic", "/camera2/camera_info");
  this->declare_parameter<bool>("use_referee_identity_color", true);
  this->declare_parameter<std::string>("referee_identity_topic", "/radar/referee/robot_identity");
  this->declare_parameter<bool>("gimbal.require_ready", true);
  this->declare_parameter<std::string>("gimbal.angles_topic", "/gimbal/angles_true");
  this->declare_parameter<double>("gimbal.max_msg_age_s", 10.0);

  this->declare_parameter<bool>("qos.use_sensor_data_qos", true);
  this->declare_parameter<int>("qos.depth", 1);
  this->declare_parameter<bool>("qos.history_keep_all", false);
  this->declare_parameter<std::string>("qos.reliability", "best_effort");

  this->declare_parameter<int>("publish_hz", 200);

  this->declare_parameter<std::string>("output.angle_unit", "rad"); // "rad" or "deg"

  // prediction params
  this->declare_parameter<bool>("prediction.enable", true);
  this->declare_parameter<double>("prediction.rate_smoothing", 0.8);
  this->declare_parameter<double>("prediction.max_rate_deg_s", 720.0);
  this->declare_parameter<int64_t>("prediction.extra_latency_ns", 0);

  // debug image publish
  this->declare_parameter<bool>("debug.publish_debug_image", false);
  this->declare_parameter<bool>("debug.publish_binary_image", false);
  this->declare_parameter<int>("debug.decimation", 3);   // 每3帧生成/发布1次
  this->declare_parameter<std::string>("debug.debug_topic", "/auto_aim/debug_image");
  this->declare_parameter<std::string>("debug.binary_topic", "/auto_aim/binary_image");

  this->declare_parameter<bool>("camera.use_camera_info", false);
  this->declare_parameter<bool>("camera.set_once", true);

  // tracker
  this->declare_parameter<int>("tracker.confirm_track_frames", 3);
  this->declare_parameter<int>("tracker.lost_to_lost_frames", 5);
  this->declare_parameter<int>("tracker.lost_to_search_frames", 20);
  this->declare_parameter<double>("tracker.min_conf_to_track", 0.5);

  this->declare_parameter<int>("tracker.roi_base_size", 200);
  this->declare_parameter<double>("tracker.roi_expand_lost", 1.3);
  this->declare_parameter<int>("tracker.roi_max_size", 800);

  // budget
  this->declare_parameter<int>("budget.max_lights", 64);
  this->declare_parameter<int>("budget.max_pairs", 128);
  this->declare_parameter<int>("budget.max_pnp", 2);

  // fallback camera
  this->declare_parameter<double>("camera.fx", 1.0);
  this->declare_parameter<double>("camera.fy", 1.0);
  this->declare_parameter<double>("camera.cx", 0.0);
  this->declare_parameter<double>("camera.cy", 0.0);
  this->declare_parameter<std::vector<double>>("camera.dist", std::vector<double>{0,0,0,0,0});

  // detector
  this->declare_parameter<int>("detector.binary_thresh", 255);
  this->declare_parameter<bool>("detector.use_sigma_thresh", true);
  this->declare_parameter<double>("detector.sigma_k", 2.5);
  this->declare_parameter<int>("detector.min_dynamic_thresh", 80);
  this->declare_parameter<int>("detector.max_dynamic_thresh", 220);

  this->declare_parameter<double>("detector.spot_min_area", 3.0);
  this->declare_parameter<double>("detector.spot_max_area", 1200.0);
  this->declare_parameter<double>("detector.spot_min_ratio", 0.3);
  this->declare_parameter<double>("detector.spot_max_ratio", 1.01);

  this->declare_parameter<int>("detector.min_row_spots", 2);
  this->declare_parameter<double>("detector.row_center_x_tol_ratio", 0.6);
  this->declare_parameter<double>("detector.row_width_similarity_ratio", 0.8);
  this->declare_parameter<double>("detector.pattern_min_aspect", 0.6);
  this->declare_parameter<double>("detector.pattern_max_aspect", 4.0);

  this->declare_parameter<int>("detector.target_color", 0);
  this->declare_parameter<int>("detector.color_threshold", 30);

  this->declare_parameter<bool>("detector.show_contours", false);
  this->declare_parameter<bool>("detector.show_pairs", true);
  this->declare_parameter<std::string>("detector.mode", "legacy");
  this->declare_parameter<std::string>("detector.yolo.engine_path", "");
  this->declare_parameter<int>("detector.yolo.input_w", 640);
  this->declare_parameter<int>("detector.yolo.input_h", 640);
  this->declare_parameter<int>("detector.yolo.max_det", 300);
  this->declare_parameter<int>("detector.yolo.target_class_id", 0);
  this->declare_parameter<double>("detector.yolo.conf_thresh", 0.5);
  this->declare_parameter<double>("detector.yolo.nms_thresh", 0.45);
  this->declare_parameter<double>("detector.yolo.kpt_conf_thresh", 0.5);
  this->declare_parameter<double>("detector.yolo.min_w_ratio", 0.045);
  this->declare_parameter<double>("detector.yolo.min_h_ratio", 0.055);
  this->declare_parameter<double>("detector.yolo.min_area_ratio", 0.0025);
  this->declare_parameter<bool>("detector.yolo.require_outer_corners", true);
  this->declare_parameter<bool>("detector.yolo.use_fp16_input", true);
  this->declare_parameter<bool>("detector.yolo.swap_rb_input", false);
  this->declare_parameter<std::vector<int64_t>>("detector.yolo.outer_corner_indices", std::vector<int64_t>{0, 4, 7, 3});
  this->declare_parameter<bool>("detector.laser_spot_filter.enable", true);
  this->declare_parameter<double>("detector.laser_spot_filter.max_near_circle_aspect", 1.35);
  this->declare_parameter<double>("detector.laser_spot_filter.min_side_px", 20.0);
  this->declare_parameter<double>("detector.laser_spot_filter.min_vertical_cv", 0.12);
  this->declare_parameter<double>("detector.laser_spot_filter.min_vertical_gradient", 0.04);
  this->declare_parameter<double>("detector.laser_spot_filter.roi_padding_ratio", 0.0);

  // 配对模式参数
  this->declare_parameter<int>("detector.pair_mode", 1);
  this->declare_parameter<double>("detector.max_x_diff_ratio", 0.5);
  this->declare_parameter<double>("detector.max_y_diff_ratio", 0.5);
  this->declare_parameter<double>("detector.min_vertical_dist", 0.8);
  this->declare_parameter<double>("detector.max_vertical_dist", 4.0);
  this->declare_parameter<double>("detector.min_connect_angle", 60.0);
  this->declare_parameter<double>("detector.max_connect_angle", 120.0);


  image_topic_ = this->get_parameter("image_topic").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  use_referee_identity_color_ = this->get_parameter("use_referee_identity_color").as_bool();
  referee_identity_topic_ = this->get_parameter("referee_identity_topic").as_string();
  require_gimbal_ready_ = this->get_parameter("gimbal.require_ready").as_bool();
  gimbal_angles_topic_ = this->get_parameter("gimbal.angles_topic").as_string();
  gimbal_max_msg_age_s_ =
    std::max(0.01, this->get_parameter("gimbal.max_msg_age_s").as_double());
  use_camera_info_ = this->get_parameter("camera.use_camera_info").as_bool();

  // tracker inject
  tracking::TrackerParams tp;
  tp.confirm_track_frames = this->get_parameter("tracker.confirm_track_frames").as_int();
  tp.lost_to_lost_frames = this->get_parameter("tracker.lost_to_lost_frames").as_int();
  tp.lost_to_search_frames = this->get_parameter("tracker.lost_to_search_frames").as_int();
  tp.min_conf_to_track = static_cast<float>(this->get_parameter("tracker.min_conf_to_track").as_double());
  tp.roi_base_size = this->get_parameter("tracker.roi_base_size").as_int();
  tp.roi_expand_lost = static_cast<float>(this->get_parameter("tracker.roi_expand_lost").as_double());
  tp.roi_max_size = this->get_parameter("tracker.roi_max_size").as_int();
  pipeline_.setTrackerParams(tp);

  // detector inject
  SimpleLightDetector::Params dp;
  dp.binary_thresh = this->get_parameter("detector.binary_thresh").as_int();

  dp.use_sigma_thresh =
      this->get_parameter("detector.use_sigma_thresh").as_bool();
  dp.sigma_k =
      static_cast<float>(this->get_parameter("detector.sigma_k").as_double());
  dp.min_dynamic_thresh =
      this->get_parameter("detector.min_dynamic_thresh").as_int();
  dp.max_dynamic_thresh =
      this->get_parameter("detector.max_dynamic_thresh").as_int();

  dp.spot_min_area =
      this->get_parameter("detector.spot_min_area").as_double();
  dp.spot_max_area =
      this->get_parameter("detector.spot_max_area").as_double();
  dp.spot_min_ratio =
      static_cast<float>(this->get_parameter("detector.spot_min_ratio").as_double());
  dp.spot_max_ratio =
      static_cast<float>(this->get_parameter("detector.spot_max_ratio").as_double());

  dp.min_row_spots =
      this->get_parameter("detector.min_row_spots").as_int();
  dp.row_center_x_tol_ratio =
      static_cast<float>(this->get_parameter("detector.row_center_x_tol_ratio").as_double());
  dp.row_width_similarity_ratio =
      static_cast<float>(this->get_parameter("detector.row_width_similarity_ratio").as_double());
  dp.pattern_min_aspect =
      static_cast<float>(this->get_parameter("detector.pattern_min_aspect").as_double());
  dp.pattern_max_aspect =
      static_cast<float>(this->get_parameter("detector.pattern_max_aspect").as_double());

  dp.target_color =
      this->get_parameter("detector.target_color").as_int();
  dp.color_threshold =
      this->get_parameter("detector.color_threshold").as_int();

  dp.show_contours =
      this->get_parameter("detector.show_contours").as_bool();
  dp.show_pairs =
      this->get_parameter("detector.show_pairs").as_bool();

  dp.pair_mode =
      this->get_parameter("detector.pair_mode").as_int();
  dp.max_x_diff_ratio =
      static_cast<float>(this->get_parameter("detector.max_x_diff_ratio").as_double());
  dp.max_y_diff_ratio =
      static_cast<float>(this->get_parameter("detector.max_y_diff_ratio").as_double());
  dp.min_vertical_dist =
      static_cast<float>(this->get_parameter("detector.min_vertical_dist").as_double());
  dp.max_vertical_dist =
      static_cast<float>(this->get_parameter("detector.max_vertical_dist").as_double());
  dp.min_connect_angle =
      static_cast<float>(this->get_parameter("detector.min_connect_angle").as_double());
  dp.max_connect_angle =
      static_cast<float>(this->get_parameter("detector.max_connect_angle").as_double());

  detector_params_ = dp;
  pipeline_.setDetectorParams(dp);

  const auto detector_mode_str = this->get_parameter("detector.mode").as_string();
  const auto detector_mode =
    (detector_mode_str == "yolo" || detector_mode_str == "YOLO")
      ? vision::DetectorMode::YOLO
      : vision::DetectorMode::LEGACY;
  pipeline_.setDetectorMode(detector_mode);

  vision::YoloDetectorParams yp;
  yp.engine_path = this->get_parameter("detector.yolo.engine_path").as_string();
  yp.input_w = this->get_parameter("detector.yolo.input_w").as_int();
  yp.input_h = this->get_parameter("detector.yolo.input_h").as_int();
  yp.max_det = this->get_parameter("detector.yolo.max_det").as_int();
  yp.target_class_id = this->get_parameter("detector.yolo.target_class_id").as_int();
  yp.conf_thresh =
      static_cast<float>(this->get_parameter("detector.yolo.conf_thresh").as_double());
  yp.nms_thresh =
      static_cast<float>(this->get_parameter("detector.yolo.nms_thresh").as_double());
  yp.kpt_conf_thresh =
      static_cast<float>(this->get_parameter("detector.yolo.kpt_conf_thresh").as_double());
  yp.min_w_ratio =
      static_cast<float>(this->get_parameter("detector.yolo.min_w_ratio").as_double());
  yp.min_h_ratio =
      static_cast<float>(this->get_parameter("detector.yolo.min_h_ratio").as_double());
  yp.min_area_ratio =
      static_cast<float>(this->get_parameter("detector.yolo.min_area_ratio").as_double());
  yp.require_outer_corners =
      this->get_parameter("detector.yolo.require_outer_corners").as_bool();
  yp.use_fp16_input =
      this->get_parameter("detector.yolo.use_fp16_input").as_bool();
  yp.swap_rb_input =
      this->get_parameter("detector.yolo.swap_rb_input").as_bool();
  {
    const auto idx = this->get_parameter("detector.yolo.outer_corner_indices").as_integer_array();
    if (idx.size() == 4) {
      yp.outer_corner_indices = {
        static_cast<int>(idx[0]),
        static_cast<int>(idx[1]),
        static_cast<int>(idx[2]),
        static_cast<int>(idx[3])
      };
    }
  }
  yolo_params_ = yp;
  pipeline_.setYoloDetectorParams(yp);

  pipeline::LaserSpotFilterParams laser_filter;
  laser_filter.enable =
    this->get_parameter("detector.laser_spot_filter.enable").as_bool();
  laser_filter.max_near_circle_aspect =
    static_cast<float>(
      std::max(
        1.0,
        this->get_parameter("detector.laser_spot_filter.max_near_circle_aspect").as_double()));
  laser_filter.min_side_px =
    static_cast<float>(
      std::max(1.0, this->get_parameter("detector.laser_spot_filter.min_side_px").as_double()));
  laser_filter.min_vertical_cv =
    static_cast<float>(
      std::max(0.0, this->get_parameter("detector.laser_spot_filter.min_vertical_cv").as_double()));
  laser_filter.min_vertical_gradient =
    static_cast<float>(
      std::max(
        0.0,
        this->get_parameter("detector.laser_spot_filter.min_vertical_gradient").as_double()));
  laser_filter.roi_padding_ratio =
    static_cast<float>(
      std::clamp(
        this->get_parameter("detector.laser_spot_filter.roi_padding_ratio").as_double(),
        0.0,
        0.5));
  pipeline_.setLaserSpotFilterParams(laser_filter);
  pipeline_.setDebugImageEnabled(
    this->get_parameter("debug.publish_debug_image").as_bool() ||
    this->get_parameter("debug.publish_binary_image").as_bool());

  // camera fallback
  if (!use_camera_info_) {
    const double fx = this->get_parameter("camera.fx").as_double();
    const double fy = this->get_parameter("camera.fy").as_double();
    const double cx = this->get_parameter("camera.cx").as_double();
    const double cy = this->get_parameter("camera.cy").as_double();
    const auto dist = this->get_parameter("camera.dist").as_double_array();
    pipeline_.setCamera(makeK(fx, fy, cx, cy), makeD(dist));
    camera_set_ = true;
  }

  // publishers
  pub_target_ = this->create_publisher<auto_aim_fast::msg::AimTarget>("/auto_aim/target", rclcpp::QoS(10));
  pub_stats_  = this->create_publisher<auto_aim_fast::msg::DebugStats>("/auto_aim/stats", rclcpp::QoS(10));

  // debug publishers (created always; whether to publish depends on params)
  pub_debug_image_ = this->create_publisher<sensor_msgs::msg::Image>(
    this->get_parameter("debug.debug_topic").as_string(), rclcpp::QoS(1));
  pub_binary_image_ = this->create_publisher<sensor_msgs::msg::Image>(
    this->get_parameter("debug.binary_topic").as_string(), rclcpp::QoS(1));

  // subscriptions
  if (require_gimbal_ready_) {
    sub_gimbal_angles_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      gimbal_angles_topic_, rclcpp::SensorDataQoS(),
      std::bind(&AutoAimNode::onGimbalAngles, this, std::placeholders::_1));
  }

  sub_identity_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
    referee_identity_topic_, 10,
    std::bind(&AutoAimNode::onRefereeIdentity, this, std::placeholders::_1));

  pipeline_.start();
  updateImageSubscriptions(!require_gimbal_ready_);

  const int publish_hz = std::max(1, static_cast<int>(this->get_parameter("publish_hz").as_int()));
  publish_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000 / publish_hz)),
    std::bind(&AutoAimNode::onPublish, this));

  report_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&AutoAimNode::onReport, this));

  RCLCPP_INFO(this->get_logger(),
              "auto_aim_node started. detector_mode=%s debug_pub=%d binary_pub=%d gimbal_gate=%s topic=%s timeout=%.3fs",
              this->get_parameter("detector.mode").as_string().c_str(),
              this->get_parameter("debug.publish_debug_image").as_bool() ? 1 : 0,
              this->get_parameter("debug.publish_binary_image").as_bool() ? 1 : 0,
              require_gimbal_ready_ ? "enabled" : "disabled",
              gimbal_angles_topic_.c_str(),
              gimbal_max_msg_age_s_);
}

void AutoAimNode::onRefereeIdentity(
  const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
{
  if (!use_referee_identity_color_ || !msg || !msg->valid) {
    return;
  }

  const int next_target_color =
    radar_decision_comm::team_identity::enemy_auto_aim_target_color_from_self_color(
      msg->self_color);
  const bool next_swap_rb =
    radar_decision_comm::team_identity::enemy_yolo_swap_rb_from_self_color(
      msg->self_color);

  if (detector_params_.target_color == next_target_color &&
      yolo_params_.swap_rb_input == next_swap_rb)
  {
    return;
  }

  detector_params_.target_color = next_target_color;
  yolo_params_.swap_rb_input = next_swap_rb;
  pipeline_.setDetectorParams(detector_params_);
  pipeline_.setYoloDetectorParams(yolo_params_);

  RCLCPP_INFO(
    this->get_logger(),
    "AutoAimNode switched target_color=%d swap_rb_input=%s from referee robot_id=%u",
    detector_params_.target_color,
    yolo_params_.swap_rb_input ? "true" : "false",
    msg->robot_id);
}

AutoAimNode::~AutoAimNode()
{
  pipeline_.stop();
}

void AutoAimNode::updateImageSubscriptions(bool active)
{
  std::lock_guard<std::mutex> lk(subscription_mutex_);
  if (active) {
    if (image_subscriptions_active_) {
      return;
    }

    sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, makeImageQoS(*this),
      std::bind(&AutoAimNode::onImage, this, std::placeholders::_1));

    if (use_camera_info_) {
      sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, makeInfoQoS(),
        std::bind(&AutoAimNode::onCameraInfo, this, std::placeholders::_1));
    }

    image_subscriptions_active_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "auto_aim image input active: image=%s camera_info=%s",
      image_topic_.c_str(),
      use_camera_info_ ? camera_info_topic_.c_str() : "disabled");
    return;
  }

  if (!image_subscriptions_active_ && !sub_image_ && !sub_info_) {
    return;
  }

  sub_image_.reset();
  sub_info_.reset();
  image_subscriptions_active_ = false;
  pipeline_.clearPending();
  RCLCPP_WARN(
    this->get_logger(),
    "auto_aim image input suspended; waiting for initialized gimbal feedback.");
}

void AutoAimNode::onGimbalAngles(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (!msg || msg->data.size() < 2 ||
      !std::isfinite(msg->data[0]) ||
      !std::isfinite(msg->data[1]))
  {
    return;
  }

  bool became_ready = false;
  {
    std::lock_guard<std::mutex> lk(gimbal_mutex_);
    became_ready = !gimbal_ready_;
    gimbal_ready_ = true;
    last_gimbal_msg_time_ = this->get_clock()->now();
  }

  if (became_ready) {
    RCLCPP_INFO(
      this->get_logger(),
      "Initialized gimbal feedback received; auto_aim inference can resume.");
  }
  updateImageSubscriptions(true);
}

bool AutoAimNode::isGimbalReady(const rclcpp::Time & now, double * age_s)
{
  if (!require_gimbal_ready_) {
    if (age_s) {
      *age_s = 0.0;
    }
    return true;
  }

  bool timed_out = false;
  double age = std::numeric_limits<double>::infinity();
  {
    std::lock_guard<std::mutex> lk(gimbal_mutex_);
    if (!gimbal_ready_) {
      if (age_s) {
        *age_s = age;
      }
      return false;
    }

    age = (now - last_gimbal_msg_time_).seconds();
    if (!std::isfinite(age) || age < 0.0) {
      age = 0.0;
    }

    if (age <= gimbal_max_msg_age_s_) {
      if (age_s) {
        *age_s = age;
      }
      return true;
    }

    gimbal_ready_ = false;
    timed_out = true;
  }

  if (timed_out) {
    if (age_s) {
      *age_s = age;
    }
    updateImageSubscriptions(false);
    RCLCPP_ERROR(
      this->get_logger(),
      "%sGimbal feedback timed out after %.3fs; auto_aim inference is paused.%s",
      kRedBold, age, kColorReset);
  }

  return false;
}

void AutoAimNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!use_camera_info_) return;

  const bool set_once = this->get_parameter("camera.set_once").as_bool();
  if (set_once && camera_set_) return;

  const auto & k = msg->k;
  cv::Mat K = (cv::Mat_<double>(3,3) << k[0], k[1], k[2],
                                       k[3], k[4], k[5],
                                       k[6], k[7], k[8]);
  cv::Mat D = makeD(msg->d);

  pipeline_.setCamera(K, D);
  camera_set_ = true;
}

void AutoAimNode::onImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
  double gimbal_age_s = 0.0;
  if (!isGimbalReady(this->get_clock()->now(), &gimbal_age_s)) {
    gimbal_wait_skipped_frames_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  pipeline::Frame f;
  f.image = msg;
  f.stamp_ros = rclcpp::Time(msg->header.stamp);
  f.stamp_ns =
    static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(msg->header.stamp.nanosec);
  f.recv_ns = util::nowSteadyNs();
  f.frame_id = msg->header.frame_id;

  f.budget.max_lights = static_cast<int>(this->get_parameter("budget.max_lights").as_int());
  f.budget.max_pairs  = static_cast<int>(this->get_parameter("budget.max_pairs").as_int());
  f.budget.max_pnp    = static_cast<int>(this->get_parameter("budget.max_pnp").as_int());

  pipeline_.pushFrame(f);
}

void AutoAimNode::publishWaitingForGimbal(const rclcpp::Time & now, double gimbal_age_s)
{
  pipeline_.setDebugImageEnabled(false);
  const auto out = pipeline_.getLatestOutput();
  const std::string unit = this->get_parameter("output.angle_unit").as_string();

  auto_aim_fast::msg::AimTarget tgt;
  tgt.stamp = now;
  tgt.state = "WAIT_GIMBAL";
  tgt.has_target = false;
  tgt.yaw = 0.0f;
  tgt.pitch = 0.0f;
  tgt.yaw_pred = 0.0f;
  tgt.pitch_pred = 0.0f;
  tgt.angle_unit = unit;
  tgt.distance = 0.0f;
  tgt.confidence = 0.0f;
  tgt.has_pose = false;
  tgt.tx = 0.0f;
  tgt.ty = 0.0f;
  tgt.tz = 0.0f;
  tgt.latency_ns = 0;
  tgt.extra_latency_ns = this->get_parameter("prediction.extra_latency_ns").as_int();
  tgt.roi_used = false;
  tgt.roi_x = 0;
  tgt.roi_y = 0;
  tgt.roi_w = 0;
  tgt.roi_h = 0;
  tgt.u = std::numeric_limits<float>::quiet_NaN();
  tgt.v = std::numeric_limits<float>::quiet_NaN();
  tgt.pushed_frames = out.pushed_frames;
  tgt.processed_frames = out.processed_frames;
  tgt.dropped_frames = out.dropped_frames;
  pub_target_->publish(tgt);

  auto_aim_fast::msg::DebugStats st;
  st.stamp = now;
  st.state = "WAIT_GIMBAL";
  st.pushed_frames = out.pushed_frames;
  st.processed_frames = out.processed_frames;
  st.dropped_frames = out.dropped_frames;
  st.good_count = 0;
  st.miss_count = 0;
  st.roi_used = false;
  st.roi_x = 0;
  st.roi_y = 0;
  st.roi_w = 0;
  st.roi_h = 0;
  st.roi_img_w = out.roi_dbg.img_w;
  st.roi_img_h = out.roi_dbg.img_h;
  st.has_pose = false;
  st.tx = 0.0f;
  st.ty = 0.0f;
  st.tz = 0.0f;
  st.preprocess_ms = 0.0f;
  st.pairing_ms = 0.0f;
  st.pnp_ms = 0.0f;
  st.tracker_ms = 0.0f;
  (void)gimbal_age_s;
  pub_stats_->publish(st);
}

void AutoAimNode::onPublish()
{
  const rclcpp::Time now = this->get_clock()->now();
  double gimbal_age_s = 0.0;
  if (!isGimbalReady(now, &gimbal_age_s)) {
    pred_inited_ = false;
    yaw_rate_ = 0.0f;
    pitch_rate_ = 0.0f;
    publishWaitingForGimbal(now, gimbal_age_s);
    return;
  }

  const auto out = pipeline_.getLatestOutput();

  const bool has_img_stamp = !(out.stamp_sec == 0 && out.stamp_nanosec == 0);

  int64_t latency_ns = 0;
  if (has_img_stamp) {
    rclcpp::Time img_stamp(
      static_cast<int64_t>(out.stamp_sec) * 1000000000LL + static_cast<int64_t>(out.stamp_nanosec),
      this->get_clock()->get_clock_type());
    latency_ns = (now - img_stamp).nanoseconds();
  }

  const int64_t extra_latency_ns = this->get_parameter("prediction.extra_latency_ns").as_int();

  float yaw_m = out.yaw;
  float pitch_m = out.pitch;

  float yaw_pred = yaw_m;
  float pitch_pred = pitch_m;

  const bool pred_enable = this->get_parameter("prediction.enable").as_bool();
  const double alpha = this->get_parameter("prediction.rate_smoothing").as_double();
  const double max_rate_deg_s = this->get_parameter("prediction.max_rate_deg_s").as_double();
  const float max_rate = static_cast<float>(max_rate_deg_s * kDeg2Rad);

  const bool can_predict = (out.state == "TRACK") && out.has_pose && has_img_stamp;
  const int64_t stamp_ns = static_cast<int64_t>(out.stamp_sec) * 1000000000LL + static_cast<int64_t>(out.stamp_nanosec);

  if (pred_enable && can_predict) {
    if (!pred_inited_) {
      pred_inited_ = true;
      last_stamp_ns_ = stamp_ns;
      last_yaw_ = yaw_m;
      last_pitch_ = pitch_m;
      yaw_rate_ = 0.0f;
      pitch_rate_ = 0.0f;
    } else {
      const int64_t dt_ns = stamp_ns - last_stamp_ns_;
      if (dt_ns > 0) {
        const float dt = static_cast<float>(dt_ns) / 1e9f;

        float dyaw = yaw_m - last_yaw_;
        float dpitch = pitch_m - last_pitch_;

        while (dyaw > static_cast<float>(M_PI))  dyaw -= static_cast<float>(2.0 * M_PI);
        while (dyaw < -static_cast<float>(M_PI)) dyaw += static_cast<float>(2.0 * M_PI);

        const float yaw_rate_meas = dyaw / dt;
        const float pitch_rate_meas = dpitch / dt;

        const float a = static_cast<float>(std::clamp(alpha, 0.0, 0.99));
        yaw_rate_ = a * yaw_rate_ + (1.0f - a) * yaw_rate_meas;
        pitch_rate_ = a * pitch_rate_ + (1.0f - a) * pitch_rate_meas;

        yaw_rate_ = std::clamp(yaw_rate_, -max_rate, max_rate);
        pitch_rate_ = std::clamp(pitch_rate_, -max_rate, max_rate);

        last_stamp_ns_ = stamp_ns;
        last_yaw_ = yaw_m;
        last_pitch_ = pitch_m;
      }
    }

    const float dt_pred = static_cast<float>(latency_ns + extra_latency_ns) / 1e9f;
    yaw_pred = yaw_m + yaw_rate_ * dt_pred;
    pitch_pred = pitch_m + pitch_rate_ * dt_pred;
  } else {
    pred_inited_ = false;
    yaw_rate_ = 0.0f;
    pitch_rate_ = 0.0f;
  }

  const std::string unit = this->get_parameter("output.angle_unit").as_string();
  float yaw_out = yaw_m, pitch_out = pitch_m, yaw_pred_out = yaw_pred, pitch_pred_out = pitch_pred;
  if (unit == "deg") {
    yaw_out *= kRad2Deg;
    pitch_out *= kRad2Deg;
    yaw_pred_out *= kRad2Deg;
    pitch_pred_out *= kRad2Deg;
  }

  const bool strict_has_target = (out.state == "TRACK") && out.has_pose;

  auto_aim_fast::msg::AimTarget tgt;

  if (has_img_stamp) {
    tgt.stamp.sec = out.stamp_sec;
    tgt.stamp.nanosec = out.stamp_nanosec;
  } else {
    tgt.stamp = now;
  }

  tgt.state = out.state;
  tgt.has_target = strict_has_target;

  tgt.yaw = yaw_out;
  tgt.pitch = pitch_out;
  tgt.yaw_pred = yaw_pred_out;
  tgt.pitch_pred = pitch_pred_out;
  tgt.angle_unit = unit;

  tgt.distance = out.dist;
  tgt.confidence = out.confidence;

  tgt.has_pose = out.has_pose;
  tgt.tx = out.tx;
  tgt.ty = out.ty;
  tgt.tz = out.tz;

  tgt.latency_ns = has_img_stamp ? latency_ns : 0;
  tgt.extra_latency_ns = extra_latency_ns;

  tgt.roi_used = out.roi_dbg.used;
  tgt.roi_x = out.roi_dbg.x;
  tgt.roi_y = out.roi_dbg.y;
  tgt.roi_w = out.roi_dbg.w;
  tgt.roi_h = out.roi_dbg.h;

  if (out.has_target_center) {
    tgt.u = out.target_u;
    tgt.v = out.target_v;
  } else {
    tgt.u = std::numeric_limits<float>::quiet_NaN();
    tgt.v = std::numeric_limits<float>::quiet_NaN();
  }

  tgt.pushed_frames = out.pushed_frames;
  tgt.processed_frames = out.processed_frames;
  tgt.dropped_frames = out.dropped_frames;

  pub_target_->publish(tgt);

  auto_aim_fast::msg::DebugStats st;
  st.stamp = now;
  st.state = out.state;

  st.pushed_frames = out.pushed_frames;
  st.processed_frames = out.processed_frames;
  st.dropped_frames = out.dropped_frames;

  st.good_count = out.good_count;
  st.miss_count = out.miss_count;

  st.roi_used = out.roi_dbg.used;
  st.roi_x = out.roi_dbg.x;
  st.roi_y = out.roi_dbg.y;
  st.roi_w = out.roi_dbg.w;
  st.roi_h = out.roi_dbg.h;
  st.roi_img_w = out.roi_dbg.img_w;
  st.roi_img_h = out.roi_dbg.img_h;

  st.has_pose = out.has_pose;
  st.tx = out.tx;
  st.ty = out.ty;
  st.tz = out.tz;

  st.preprocess_ms = static_cast<float>(out.stats.preprocess_ns / 1e6);
  st.pairing_ms = static_cast<float>(out.stats.pairing_ns / 1e6);
  st.pnp_ms = static_cast<float>(out.stats.pnp_ns / 1e6);
  st.tracker_ms = static_cast<float>(out.stats.tracker_ns / 1e6);

  pub_stats_->publish(st);

  // ---- debug image publish (复现优先) ----
  const bool pub_dbg = this->get_parameter("debug.publish_debug_image").as_bool();
  const bool pub_bin = this->get_parameter("debug.publish_binary_image").as_bool();
  pipeline_.setDebugImageEnabled(pub_dbg || pub_bin);
  const int64_t debug_decimation_raw = this->get_parameter("debug.decimation").as_int();
  const int debug_decimation = static_cast<int>(std::max<int64_t>(1, debug_decimation_raw));

  if (pub_dbg || pub_bin) {
    cv::Mat dbg_bgr, bin;
    int32_t debug_stamp_sec = 0;
    uint32_t debug_stamp_nanosec = 0;
    std::string debug_frame_id;
    uint64_t debug_sequence = 0;
    if (pipeline_.getLastDebugImages(
          dbg_bgr, bin, debug_stamp_sec, debug_stamp_nanosec, debug_frame_id, debug_sequence)) {
      static uint64_t last_debug_published_sequence = 0;
      static uint64_t debug_frame_counter = 0;
      if (debug_sequence == 0 || debug_sequence == last_debug_published_sequence) {
        return;
      }
      last_debug_published_sequence = debug_sequence;
      debug_frame_counter++;
      const bool debug_this_frame =
        ((debug_frame_counter % static_cast<uint64_t>(debug_decimation)) == 0);
      if (!debug_this_frame) {
        return;
      }

      std_msgs::msg::Header debug_header;
      if (debug_stamp_sec != 0 || debug_stamp_nanosec != 0) {
        debug_header.stamp.sec = debug_stamp_sec;
        debug_header.stamp.nanosec = debug_stamp_nanosec;
      } else {
        debug_header.stamp = now;
      }
      debug_header.frame_id = debug_frame_id;

      if (pub_dbg && !dbg_bgr.empty()) {
        auto msg = cv_bridge::CvImage(
          debug_header,
          sensor_msgs::image_encodings::BGR8,
          dbg_bgr).toImageMsg();
        pub_debug_image_->publish(*msg);
      }

      if (pub_bin && !bin.empty()) {
        cv::Mat mono = bin;
        if (bin.channels() != 1) {
          cv::cvtColor(bin, mono, cv::COLOR_BGR2GRAY);
        }
        auto msg = cv_bridge::CvImage(
          debug_header,
          sensor_msgs::image_encodings::MONO8,
          mono).toImageMsg();
        pub_binary_image_->publish(*msg);
      }
    }
  }
}

void AutoAimNode::onReport()
{
  double gimbal_age_s = 0.0;
  if (!isGimbalReady(this->get_clock()->now(), &gimbal_age_s)) {
    // Noisy periodic status; keep disabled unless debugging gimbal freshness.
    // RCLCPP_INFO(
    //   this->get_logger(),
    //   "state=WAIT_GIMBAL skipped_frames=%lu gimbal_age=%.3f timeout=%.3f",
    //   static_cast<unsigned long>(gimbal_wait_skipped_frames_.load(std::memory_order_relaxed)),
    //   std::isfinite(gimbal_age_s) ? gimbal_age_s : -1.0,
    //   gimbal_max_msg_age_s_);
    return;
  }

  // Noisy periodic status; keep disabled unless debugging auto aim throughput.
  // const auto out = pipeline_.getLatestOutput();
  // RCLCPP_INFO(this->get_logger(),
  //             "state=%s pairs=%d->%d roi_used=%d skipped_wait=%lu",
  //             out.state.c_str(),
  //             out.budget.pairs_in,
  //             out.budget.pairs_used,
  //             out.roi_dbg.used ? 1 : 0,
  //             static_cast<unsigned long>(gimbal_wait_skipped_frames_.load(std::memory_order_relaxed)));
}

}  // namespace auto_aim_fast

RCLCPP_COMPONENTS_REGISTER_NODE(auto_aim_fast::AutoAimNode)
