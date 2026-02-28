#include "auto_aim_fast/node/auto_aim_node.hpp"
#include "auto_aim_fast/util/time.hpp"
#include "auto_aim_fast/pipeline/frame.hpp"
#include "auto_aim_fast/tracking/target_tracker.hpp"
#include "auto_aim_fast/legacy/detector.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

// cv_bridge for publishing debug images
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

namespace auto_aim_fast
{

static constexpr float kRad2Deg = 57.29577951308232f;
static constexpr float kDeg2Rad = 0.017453292519943295f;

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
  this->declare_parameter<std::string>("image_topic", "/camera0/image");
  this->declare_parameter<std::string>("camera_info_topic", "/camera0/camera_info");

  this->declare_parameter<bool>("qos.use_sensor_data_qos", true);
  this->declare_parameter<int>("qos.depth", 1);
  this->declare_parameter<bool>("qos.history_keep_all", false);
  this->declare_parameter<std::string>("qos.reliability", "best_effort");

  this->declare_parameter<int>("publish_hz", 100);

  this->declare_parameter<std::string>("output.angle_unit", "rad"); // "rad" or "deg"

  // prediction params
  this->declare_parameter<bool>("prediction.enable", true);
  this->declare_parameter<double>("prediction.rate_smoothing", 0.8);
  this->declare_parameter<double>("prediction.max_rate_deg_s", 720.0);
  this->declare_parameter<int64_t>("prediction.extra_latency_ns", 0);

  // debug image publish
  this->declare_parameter<bool>("debug.publish_debug_image", false);
  this->declare_parameter<bool>("debug.publish_binary_image", false);
  this->declare_parameter<std::string>("debug.debug_topic", "/auto_aim/debug_image");
  this->declare_parameter<std::string>("debug.binary_topic", "/auto_aim/binary_image");

  this->declare_parameter<bool>("camera.use_camera_info", false);
  this->declare_parameter<bool>("camera.set_once", true);

  // tracker
  this->declare_parameter<int>("tracker.confirm_track_frames", 3);
  this->declare_parameter<int>("tracker.lost_to_lost_frames", 5);
  this->declare_parameter<int>("tracker.lost_to_search_frames", 20);
  this->declare_parameter<double>("tracker.min_conf_to_track", 0.3);

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
  this->declare_parameter<int>("detector.binary_thresh", 150);
  this->declare_parameter<double>("detector.min_area", 10.0);
  this->declare_parameter<double>("detector.max_area", 10000.0);
  this->declare_parameter<double>("detector.min_ratio", 0.1);
  this->declare_parameter<double>("detector.max_ratio", 0.4);
  this->declare_parameter<double>("detector.max_angle", 90.0);

  this->declare_parameter<int>("detector.target_color", 0);
  this->declare_parameter<int>("detector.color_threshold", 30);

  this->declare_parameter<double>("detector.min_light_ratio", 0.6);
  this->declare_parameter<double>("detector.min_center_dist", 1.0);
  this->declare_parameter<double>("detector.max_center_dist", 5.0);
  this->declare_parameter<double>("detector.max_pair_angle", 10.0);
  
  // 配对模式参数
  this->declare_parameter<int>("detector.pair_mode", 1);
  this->declare_parameter<double>("detector.max_x_diff_ratio", 0.5);
  this->declare_parameter<double>("detector.max_y_diff_ratio", 0.5);
  this->declare_parameter<double>("detector.min_vertical_dist", 0.8);
  this->declare_parameter<double>("detector.max_vertical_dist", 4.0);
  this->declare_parameter<double>("detector.min_connect_angle", 60.0);
  this->declare_parameter<double>("detector.max_connect_angle", 120.0);

  this->declare_parameter<bool>("detector.show_contours", false);
  this->declare_parameter<bool>("detector.show_pairs", true);

  const auto image_topic = this->get_parameter("image_topic").as_string();
  const auto info_topic = this->get_parameter("camera_info_topic").as_string();

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
  dp.min_area = this->get_parameter("detector.min_area").as_double();
  dp.max_area = this->get_parameter("detector.max_area").as_double();
  dp.min_ratio = static_cast<float>(this->get_parameter("detector.min_ratio").as_double());
  dp.max_ratio = static_cast<float>(this->get_parameter("detector.max_ratio").as_double());
  dp.max_angle = static_cast<float>(this->get_parameter("detector.max_angle").as_double());
  dp.target_color = this->get_parameter("detector.target_color").as_int();
  dp.color_threshold = this->get_parameter("detector.color_threshold").as_int();
  dp.min_light_ratio = static_cast<float>(this->get_parameter("detector.min_light_ratio").as_double());
  dp.min_center_dist = static_cast<float>(this->get_parameter("detector.min_center_dist").as_double());
  dp.max_center_dist = static_cast<float>(this->get_parameter("detector.max_center_dist").as_double());
  dp.max_pair_angle = static_cast<float>(this->get_parameter("detector.max_pair_angle").as_double());
  
  // 配对模式参数
  dp.pair_mode = this->get_parameter("detector.pair_mode").as_int();
  dp.max_x_diff_ratio = static_cast<float>(this->get_parameter("detector.max_x_diff_ratio").as_double());
  dp.max_y_diff_ratio = static_cast<float>(this->get_parameter("detector.max_y_diff_ratio").as_double());
  dp.min_vertical_dist = static_cast<float>(this->get_parameter("detector.min_vertical_dist").as_double());
  dp.max_vertical_dist = static_cast<float>(this->get_parameter("detector.max_vertical_dist").as_double());
  dp.min_connect_angle = static_cast<float>(this->get_parameter("detector.min_connect_angle").as_double());
  dp.max_connect_angle = static_cast<float>(this->get_parameter("detector.max_connect_angle").as_double());
  
  dp.show_contours = this->get_parameter("detector.show_contours").as_bool();
  dp.show_pairs = this->get_parameter("detector.show_pairs").as_bool();
  pipeline_.setDetectorParams(dp);

  // camera fallback
  if (!this->get_parameter("camera.use_camera_info").as_bool()) {
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
  sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
    image_topic, makeImageQoS(*this),
    std::bind(&AutoAimNode::onImage, this, std::placeholders::_1));

  sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    info_topic, makeInfoQoS(),
    std::bind(&AutoAimNode::onCameraInfo, this, std::placeholders::_1));

  pipeline_.start();

  const int publish_hz = std::max(1, static_cast<int>(this->get_parameter("publish_hz").as_int()));
  publish_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000 / publish_hz)),
    std::bind(&AutoAimNode::onPublish, this));

  report_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&AutoAimNode::onReport, this));

  RCLCPP_INFO(this->get_logger(),
              "auto_aim_node started. debug_pub=%d binary_pub=%d",
              this->get_parameter("debug.publish_debug_image").as_bool() ? 1 : 0,
              this->get_parameter("debug.publish_binary_image").as_bool() ? 1 : 0);
}

AutoAimNode::~AutoAimNode()
{
  pipeline_.stop();
}

void AutoAimNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!this->get_parameter("camera.use_camera_info").as_bool()) return;

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
  pipeline::Frame f;
  f.image = msg;
  f.stamp_ros = rclcpp::Time(msg->header.stamp);
  f.stamp_ns =
    static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(msg->header.stamp.nanosec);
  f.recv_ns = util::nowSteadyNs();

  f.budget.max_lights = static_cast<int>(this->get_parameter("budget.max_lights").as_int());
  f.budget.max_pairs  = static_cast<int>(this->get_parameter("budget.max_pairs").as_int());
  f.budget.max_pnp    = static_cast<int>(this->get_parameter("budget.max_pnp").as_int());

  pipeline_.pushFrame(f);
}

void AutoAimNode::onPublish()
{
  const auto out = pipeline_.getLatestOutput();
  const rclcpp::Time now = this->get_clock()->now();

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

  if (pub_dbg || pub_bin) {
    cv::Mat dbg_bgr, bin;
    if (pipeline_.getLastDebugImages(dbg_bgr, bin)) {
      if (pub_dbg && !dbg_bgr.empty()) {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, dbg_bgr).toImageMsg();
        pub_debug_image_->publish(*msg);
      }
      if (pub_bin && !bin.empty()) {
        // ensure mono8
        cv::Mat mono = bin;
        if (bin.channels() != 1) {
          cv::cvtColor(bin, mono, cv::COLOR_BGR2GRAY);
        }
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, mono).toImageMsg();
        pub_binary_image_->publish(*msg);
      }
    }
  }
}

void AutoAimNode::onReport()
{
  const auto out = pipeline_.getLatestOutput();
  RCLCPP_INFO(this->get_logger(),
              "state=%s pairs=%d roi_used=%d",
              out.state.c_str(),
              out.budget.pairs_in,
              out.roi_dbg.used ? 1 : 0);
}

}  // namespace auto_aim_fast
