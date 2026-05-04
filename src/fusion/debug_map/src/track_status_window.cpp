#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <radar_decision_comm/msg/referee_robot_identity.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_interface/msg/detect_result.hpp>

namespace debug_map
{

namespace
{

constexpr int kRobotSlots = 7;
constexpr int kTrackSlots = kRobotSlots * 2;

struct PointSample {
  float x{0.0f};
  float y{0.0f};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

struct TrackStatus {
  std::string state{"NO_STATUS"};
  std::string output_mode{"NO_OUTPUT"};
  float confidence{0.0f};
  int hit_count{0};
  int semantic_hit_count{0};
  int lidar_hit_count{0};
  int miss_count{0};
  bool current_matched{false};
  bool publishable{false};
  bool has_blind_guess{false};
  float blind_guess_score{0.0f};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

struct DebugCloudStatus {
  int point_count{0};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

struct UavResolveDebugStatus {
  std::string mode{"--"};
  bool frustum_valid{false};
  double frustum_x{0.0};
  double frustum_y{0.0};
  double frustum_z{0.0};
  bool cluster_valid{false};
  double cluster_x{0.0};
  double cluster_y{0.0};
  double cluster_z{0.0};
  bool vision_valid{false};
  bool vision_inside{false};
  double vision_x{0.0};
  double vision_y{0.0};
  double vision_z{0.0};
  bool final_valid{false};
  double final_x{0.0};
  double final_y{0.0};
  double final_z{0.0};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

struct PureVisionResult {
  bool computed{false};
  bool inside_field{false};
  bool ray_ok{false};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

int slotIndex(char color, int number_id)
{
  if (number_id < 0 || number_id >= kRobotSlots) {
    return -1;
  }
  return (color == 'R') ? kRobotSlots + number_id : number_id;
}

std::string robotLabel(int slot)
{
  const bool red = slot >= kRobotSlots;
  const int number = red ? slot - kRobotSlots + 1 : slot + 1;
  return std::string(red ? "R" : "B") + std::to_string(number);
}

bool validPoint(float x, float y)
{
  return std::isfinite(x) && std::isfinite(y) &&
         (x != 0.0f || y != 0.0f);
}

bool freshStamp(const rclcpp::Time& stamp,
                const rclcpp::Time& now,
                double stale_timeout_s)
{
  return stamp.nanoseconds() != 0 &&
         std::fabs((now - stamp).seconds()) <= stale_timeout_s;
}

bool freshPoint(const PointSample& sample,
                const rclcpp::Time& now,
                double stale_timeout_s)
{
  return sample.valid && freshStamp(sample.stamp, now, stale_timeout_s);
}

std::vector<std::string> split(const std::string& line, char delimiter)
{
  std::vector<std::string> tokens;
  std::stringstream stream(line);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int parseInt(const std::string& text, int fallback = 0)
{
  try {
    return std::stoi(text);
  } catch (...) {
    return fallback;
  }
}

float parseFloat(const std::string& text, float fallback = 0.0f)
{
  try {
    return std::stof(text);
  } catch (...) {
    return fallback;
  }
}

cv::Scalar stateColor(const std::string& state)
{
  if (state == "CONFIRMED") {
    return cv::Scalar(80, 210, 120);
  }
  if (state == "LOST") {
    return cv::Scalar(60, 190, 255);
  }
  if (state == "TENTATIVE") {
    return cv::Scalar(90, 210, 245);
  }
  if (state == "INACTIVE") {
    return cv::Scalar(150, 150, 150);
  }
  return cv::Scalar(120, 120, 120);
}

std::string formatPoint(const PointSample& sample,
                        const rclcpp::Time& now,
                        double stale_timeout_s)
{
  if (!freshPoint(sample, now, stale_timeout_s)) {
    return "--";
  }

  char text[64];
  std::snprintf(text, sizeof(text), "%.2f, %.2f", sample.x, sample.y);
  return text;
}

std::string formatFieldPoint(double x, double y, double z, bool inside_field)
{
  char text[96];
  std::snprintf(text, sizeof(text), "%.2f, %.2f, %.2f %s",
                x, y, z, inside_field ? "OK" : "OUT");
  return text;
}

std::string formatDebugCloud(const DebugCloudStatus& status,
                             const rclcpp::Time& now,
                             double stale_timeout_s)
{
  if (!status.valid || !freshStamp(status.stamp, now, stale_timeout_s)) {
    return "--";
  }
  return std::to_string(status.point_count);
}

std::string formatDebugPoint(bool valid, double x, double y, double z)
{
  if (!valid) {
    return "--";
  }

  char text[72];
  std::snprintf(text, sizeof(text), "%.2f, %.2f, %.2f", x, y, z);
  return text;
}

std::string formatConfidence(const TrackStatus& status,
                             const rclcpp::Time& now,
                             double stale_timeout_s)
{
  if (!status.valid || status.stamp.nanoseconds() == 0 ||
      std::fabs((now - status.stamp).seconds()) > stale_timeout_s) {
    return "--";
  }

  char text[32];
  std::snprintf(text, sizeof(text), "%.2f", status.confidence);
  return text;
}

}  // namespace

class TrackStatusWindowNode : public rclcpp::Node
{
public:
  explicit TrackStatusWindowNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("track_status_window_node", options)
  {
    detect_topic_ = declare_parameter<std::string>("detect_topic", "/detect_result");
    resolve_topic_ = declare_parameter<std::string>("resolve_topic", "/resolve_result");
    kalman_topic_ = declare_parameter<std::string>("kalman_topic", "/kalman_detect");
    debug_map_topic_ = declare_parameter<std::string>("debug_map_topic", "/debug_map_result");
    status_topic_ = declare_parameter<std::string>("status_topic", "/kalman_track_status");
    identity_topic_ =
      declare_parameter<std::string>("identity_topic", "/radar/referee/robot_identity");
    uav_intersection_debug_topic_ =
      declare_parameter<std::string>("uav_intersection_debug_topic", "/uav_intersection_debug");
    uav_cluster_debug_topic_ =
      declare_parameter<std::string>("uav_cluster_debug_topic", "/uav_cluster_debug");
    uav_resolve_debug_status_topic_ =
      declare_parameter<std::string>("uav_resolve_debug_status_topic",
                                     "/uav_resolve_debug_status");
    window_name_ = declare_parameter<std::string>("window_name", "Track Status Monitor");
    stale_timeout_s_ = declare_parameter<double>("stale_timeout_s", 1.0);
    self_color_override_ =
      normalizeSelfColor(declare_parameter<int>("self_color_override", -1), -1);
    self_color_ = self_color_override_;
    self_color_valid_ = (self_color_ == 0 || self_color_ == 2);

    out_matrix_path_ =
      declare_parameter<std::string>("out_matrix_path", "config/out_matrix.yaml");
    camera_params_path_ =
      declare_parameter<std::string>("camera_params_path", "config/camera_params.yaml");
    region_points_path_ =
      declare_parameter<std::string>("region_points_path", "config/RM2025_Points.yaml");
    uav_default_z_ = declare_parameter<double>("uav_default_z", 2.1);
    uav_world_to_field_y_offset_ =
      declare_parameter<double>("uav_world_to_field_y_offset", 15.0);
    uav_field_min_x_ = declare_parameter<double>("uav_field_min_x", 0.0);
    uav_field_max_x_ = declare_parameter<double>("uav_field_max_x", 28.0);
    uav_field_min_y_ = declare_parameter<double>("uav_field_min_y", 0.0);
    uav_field_max_y_ = declare_parameter<double>("uav_field_max_y", 15.0);
    uav_field_margin_m_ = declare_parameter<double>("uav_field_margin_m", 0.20);

    const double refresh_hz =
      std::max(1.0, declare_parameter<double>("refresh_hz", 10.0));

    loadPureVisionConfig();

    auto sensor_qos = rclcpp::SensorDataQoS();
    detect_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      detect_topic_, sensor_qos,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, detect_samples_);
      });
    resolve_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      resolve_topic_, sensor_qos,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, resolve_samples_);
      });
    kalman_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      kalman_topic_, 10,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, kalman_samples_);
      });
    debug_map_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      debug_map_topic_, 10,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, debug_map_samples_);
      });
    status_sub_ = create_subscription<std_msgs::msg::String>(
      status_topic_, 10,
      std::bind(&TrackStatusWindowNode::statusCallback, this, std::placeholders::_1));
    uav_resolve_debug_status_sub_ = create_subscription<std_msgs::msg::String>(
      uav_resolve_debug_status_topic_, 10,
      std::bind(&TrackStatusWindowNode::uavResolveDebugStatusCallback,
                this, std::placeholders::_1));
    identity_sub_ =
      create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
        identity_topic_, 10,
        std::bind(&TrackStatusWindowNode::identityCallback, this, std::placeholders::_1));
    uav_intersection_debug_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      uav_intersection_debug_topic_, sensor_qos,
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        updateDebugCloud(msg, uav_intersection_status_);
      });
    uav_cluster_debug_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      uav_cluster_debug_topic_, sensor_qos,
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        updateDebugCloud(msg, uav_cluster_status_);
      });

    refresh_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / refresh_hz)),
      std::bind(&TrackStatusWindowNode::drawWindow, this));

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, 1280, 880);

    RCLCPP_INFO(
      get_logger(),
      "Track status window started. detect=%s resolve=%s kalman=%s debug_map=%s status=%s identity=%s",
      detect_topic_.c_str(), resolve_topic_.c_str(), kalman_topic_.c_str(),
      debug_map_topic_.c_str(), status_topic_.c_str(), identity_topic_.c_str());
  }

  ~TrackStatusWindowNode() override
  {
    cv::destroyWindow(window_name_);
  }

private:
  static int8_t normalizeSelfColor(int raw_color, int8_t fallback)
  {
    if (raw_color == 0 || raw_color == 2) {
      return static_cast<int8_t>(raw_color);
    }
    return fallback;
  }

  static cv::Mat makeMatrix64(const cv::Mat& input)
  {
    cv::Mat out;
    input.convertTo(out, CV_64F);
    return out;
  }

  static bool readFirstMatrix(cv::FileStorage& fs,
                              const std::vector<std::string>& keys,
                              cv::Mat& out)
  {
    for (const auto& key : keys) {
      fs[key] >> out;
      if (!out.empty()) {
        return true;
      }
    }
    return false;
  }

  void loadPureVisionConfig()
  {
    pure_vision_ready_ = false;

    cv::FileStorage fs_ext(out_matrix_path_, cv::FileStorage::READ);
    if (!fs_ext.isOpened()) {
      RCLCPP_WARN(get_logger(),
                  "UAV pure vision debug disabled: failed to open out_matrix_path=%s",
                  out_matrix_path_.c_str());
      return;
    }
    fs_ext["world_rvec"] >> world_rvec_;
    fs_ext["world_tvec"] >> world_tvec_;
    fs_ext.release();

    cv::FileStorage fs_cam(camera_params_path_, cv::FileStorage::READ);
    if (!fs_cam.isOpened()) {
      RCLCPP_WARN(get_logger(),
                  "UAV pure vision debug disabled: failed to open camera_params_path=%s",
                  camera_params_path_.c_str());
      return;
    }
    if (!readFirstMatrix(fs_cam, {"camera_matrix", "K"}, camera_matrix_)) {
      RCLCPP_WARN(get_logger(),
                  "UAV pure vision debug disabled: missing camera_matrix/K in %s",
                  camera_params_path_.c_str());
      return;
    }
    if (!readFirstMatrix(fs_cam, {"dist_coeffs", "dist_coeff", "distortion_coefficients", "D"},
                         dist_coeffs_)) {
      dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
    }
    fs_cam.release();

    inferUavFieldYOffset();

    pure_vision_ready_ =
      !world_rvec_.empty() && !world_tvec_.empty() && !camera_matrix_.empty();
    if (!pure_vision_ready_) {
      RCLCPP_WARN(get_logger(), "UAV pure vision debug disabled: empty camera matrices");
      return;
    }

    RCLCPP_INFO(get_logger(),
                "UAV pure vision debug ready. z=%.2f y_offset=%.2f field=[%.1f,%.1f]x[%.1f,%.1f]",
                uav_default_z_, uav_world_to_field_y_offset_,
                uav_field_min_x_, uav_field_max_x_,
                uav_field_min_y_, uav_field_max_y_);
  }

  void inferUavFieldYOffset()
  {
    cv::FileStorage fs(region_points_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      return;
    }

    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    const cv::FileNode root = fs.root();
    for (auto it = root.begin(); it != root.end(); ++it) {
      const cv::FileNode points = *it;
      if (!points.isSeq()) {
        continue;
      }
      for (auto pit = points.begin(); pit != points.end(); ++pit) {
        double y = 0.0;
        (*pit)["y"] >> y;
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
      }
    }
    fs.release();

    if (!std::isfinite(min_y) || !std::isfinite(max_y)) {
      return;
    }

    if (min_y < -1.0 && max_y <= 1.0) {
      uav_world_to_field_y_offset_ = std::max(0.0, uav_field_max_y_);
    } else {
      uav_world_to_field_y_offset_ = 0.0;
    }
  }

  bool pixelToWorldRay(const cv::Point2f& pixel,
                       cv::Vec3d& origin_world,
                       cv::Vec3d& dir_world) const
  {
    if (!pure_vision_ready_ || camera_matrix_.empty()) {
      return false;
    }

    const cv::Mat K = makeMatrix64(camera_matrix_);
    cv::Vec3d ray_camera;
    cv::Mat D = dist_coeffs_.empty() ? cv::Mat::zeros(1, 5, CV_64F) : makeMatrix64(dist_coeffs_);
    bool has_dist = false;
    for (int row = 0; row < D.rows; ++row) {
      for (int col = 0; col < D.cols; ++col) {
        if (std::fabs(D.at<double>(row, col)) > 1e-12) {
          has_dist = true;
        }
      }
    }

    if (has_dist) {
      std::vector<cv::Point2f> src{pixel};
      std::vector<cv::Point2f> dst;
      cv::undistortPoints(src, dst, K, D);
      if (dst.empty()) {
        return false;
      }
      ray_camera = cv::Vec3d(dst.front().x, dst.front().y, 1.0);
    } else {
      const double fx = K.at<double>(0, 0);
      const double fy = K.at<double>(1, 1);
      const double cx = K.at<double>(0, 2);
      const double cy = K.at<double>(1, 2);
      if (std::fabs(fx) < 1e-9 || std::fabs(fy) < 1e-9) {
        return false;
      }
      ray_camera = cv::Vec3d((pixel.x - cx) / fx, (pixel.y - cy) / fy, 1.0);
    }

    const double ray_norm = cv::norm(ray_camera);
    if (ray_norm <= 1e-9) {
      return false;
    }
    ray_camera /= ray_norm;

    cv::Mat rvec = makeMatrix64(world_rvec_);
    cv::Mat tvec = makeMatrix64(world_tvec_);
    if (tvec.rows == 1 && tvec.cols == 3) {
      tvec = tvec.t();
    }
    if (tvec.rows != 3 || tvec.cols != 1) {
      return false;
    }

    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);
    rot_mat.convertTo(rot_mat, CV_64F);

    const cv::Mat cam_dir =
      (cv::Mat_<double>(3, 1) << ray_camera[0], ray_camera[1], ray_camera[2]);
    const cv::Mat world_dir_mat = rot_mat.t() * cam_dir;
    const cv::Mat cam_origin_mat = -rot_mat.t() * tvec;

    dir_world = cv::Vec3d(world_dir_mat.at<double>(0, 0),
                          world_dir_mat.at<double>(1, 0),
                          world_dir_mat.at<double>(2, 0));
    const double dir_norm = cv::norm(dir_world);
    if (dir_norm <= 1e-9) {
      return false;
    }
    dir_world /= dir_norm;

    origin_world = cv::Vec3d(cam_origin_mat.at<double>(0, 0),
                             cam_origin_mat.at<double>(1, 0),
                             cam_origin_mat.at<double>(2, 0));
    return true;
  }

  PureVisionResult computePureVision(const PointSample& sample,
                                     const rclcpp::Time& t) const
  {
    PureVisionResult result;
    if (!freshPoint(sample, t, stale_timeout_s_)) {
      return result;
    }

    cv::Vec3d origin_world;
    cv::Vec3d dir_world;
    if (!pixelToWorldRay(cv::Point2f(sample.x, sample.y), origin_world, dir_world)) {
      return result;
    }
    result.ray_ok = true;

    if (std::fabs(dir_world[2]) < 1e-9) {
      return result;
    }
    const double depth = (uav_default_z_ - origin_world[2]) / dir_world[2];
    if (depth <= 1e-6 || !std::isfinite(depth)) {
      return result;
    }

    const cv::Vec3d point_world = origin_world + dir_world * depth;
    result.x = point_world[0];
    result.y = point_world[1] + uav_world_to_field_y_offset_;
    result.z = uav_default_z_;
    const double margin = std::max(0.0, uav_field_margin_m_);
    result.inside_field =
      result.x >= uav_field_min_x_ - margin &&
      result.x <= uav_field_max_x_ + margin &&
      result.y >= uav_field_min_y_ - margin &&
      result.y <= uav_field_max_y_ + margin;
    result.computed = true;
    return result;
  }

  void updateSamples(const vision_interface::msg::DetectResult::SharedPtr& msg,
                     std::array<PointSample, kTrackSlots>& samples)
  {
    const rclcpp::Time stamp = now();

    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kRobotSlots; ++i) {
      updateOneSample(samples[slotIndex('B', i)], msg->blue_x[i], msg->blue_y[i], stamp);
      updateOneSample(samples[slotIndex('R', i)], msg->red_x[i], msg->red_y[i], stamp);
    }
  }

  void updateOneSample(PointSample& sample, float x, float y, const rclcpp::Time& stamp)
  {
    sample.x = x;
    sample.y = y;
    sample.valid = validPoint(x, y);
    sample.stamp = stamp;
  }

  void identityCallback(
    const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (msg->valid) {
      self_color_ = normalizeSelfColor(msg->self_color, self_color_);
      self_color_valid_ = (self_color_ == 0 || self_color_ == 2);
      self_color_stamp_ = now();
    } else if (self_color_override_ == 0 || self_color_override_ == 2) {
      self_color_ = self_color_override_;
      self_color_valid_ = true;
      self_color_stamp_ = now();
    }
  }

  void updateDebugCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg,
                        DebugCloudStatus& status)
  {
    const rclcpp::Time stamp = now();
    std::lock_guard<std::mutex> lock(mutex_);
    status.point_count = static_cast<int>(msg->width * msg->height);
    status.valid = true;
    status.stamp = stamp;
  }

  void statusCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    const rclcpp::Time stamp = now();
    std::stringstream stream(msg->data);
    std::string line;

    std::lock_guard<std::mutex> lock(mutex_);
    while (std::getline(stream, line)) {
      if (line.empty() || line.rfind("stamp,", 0) == 0) {
        continue;
      }

      const auto fields = split(line, ',');
      if (fields.size() < 8 || fields[0].empty()) {
        continue;
      }

      const char color = fields[0][0];
      const int number_id = parseInt(fields[1], -1);
      const int slot = slotIndex(color, number_id);
      if (slot < 0) {
        continue;
      }

      TrackStatus next;
      next.state = fields[2];
      next.confidence = parseFloat(fields[3]);
      next.hit_count = parseInt(fields[4]);
      next.semantic_hit_count = parseInt(fields[5]);
      next.lidar_hit_count = parseInt(fields[6]);
      next.miss_count = parseInt(fields[7]);
      if (fields.size() >= 16) {
        next.current_matched = parseInt(fields[14]) != 0;
        next.publishable = parseInt(fields[15]) != 0;
      }
      if (fields.size() >= 19) {
        next.output_mode = fields[16];
        next.has_blind_guess = parseInt(fields[17]) != 0;
        next.blind_guess_score = parseFloat(fields[18]);
      }
      next.valid = true;
      next.stamp = stamp;

      const auto& old = statuses_[slot];
      if (old.valid && old.state != next.state) {
        pushLog(robotLabel(slot) + " " + old.state + " -> " + next.state);
      }
      statuses_[slot] = next;
    }
  }

  void uavResolveDebugStatusCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    const rclcpp::Time stamp = now();
    std::stringstream stream(msg->data);
    std::string line;

    std::lock_guard<std::mutex> lock(mutex_);
    while (std::getline(stream, line)) {
      if (line.empty() || line.rfind("stamp,", 0) == 0) {
        continue;
      }

      const auto fields = split(line, ',');
      if (fields.size() < 19 || fields[0].empty()) {
        continue;
      }

      const char color = fields[0][0];
      const int idx = (color == 'B') ? 0 : ((color == 'R') ? 1 : -1);
      if (idx < 0) {
        continue;
      }

      UavResolveDebugStatus next;
      next.mode = fields[1];
      next.frustum_valid = parseInt(fields[2]) != 0;
      next.frustum_x = parseFloat(fields[3]);
      next.frustum_y = parseFloat(fields[4]);
      next.frustum_z = parseFloat(fields[5]);
      next.cluster_valid = parseInt(fields[6]) != 0;
      next.cluster_x = parseFloat(fields[7]);
      next.cluster_y = parseFloat(fields[8]);
      next.cluster_z = parseFloat(fields[9]);
      next.vision_valid = parseInt(fields[10]) != 0;
      next.vision_x = parseFloat(fields[11]);
      next.vision_y = parseFloat(fields[12]);
      next.vision_z = parseFloat(fields[13]);
      next.vision_inside = parseInt(fields[14]) != 0;
      next.final_valid = parseInt(fields[15]) != 0;
      next.final_x = parseFloat(fields[16]);
      next.final_y = parseFloat(fields[17]);
      next.final_z = parseFloat(fields[18]);
      next.valid = true;
      next.stamp = stamp;
      uav_resolve_debug_statuses_[idx] = next;
    }
  }

  void pushLog(const std::string& line)
  {
    event_log_.push_back(line);
    while (event_log_.size() > 8) {
      event_log_.pop_front();
    }
  }

  void drawWindow()
  {
    std::array<PointSample, kTrackSlots> detect;
    std::array<PointSample, kTrackSlots> resolve;
    std::array<PointSample, kTrackSlots> kalman;
    std::array<PointSample, kTrackSlots> debug_map;
    std::array<TrackStatus, kTrackSlots> statuses;
    std::array<UavResolveDebugStatus, 2> uav_resolve_debug_statuses;
    std::deque<std::string> event_log;
    int8_t self_color = -1;
    bool self_color_valid = false;
    rclcpp::Time self_color_stamp{0, 0, RCL_ROS_TIME};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      detect = detect_samples_;
      resolve = resolve_samples_;
      kalman = kalman_samples_;
      debug_map = debug_map_samples_;
      statuses = statuses_;
      uav_resolve_debug_statuses = uav_resolve_debug_statuses_;
      event_log = event_log_;
      self_color = self_color_;
      self_color_valid = self_color_valid_;
      self_color_stamp = self_color_stamp_;
    }

    const rclcpp::Time t = now();
    cv::Mat canvas(880, 1280, CV_8UC3, cv::Scalar(25, 27, 30));
    drawTitle(canvas, t, self_color, self_color_valid, self_color_stamp);
    drawTable(canvas, t, detect, resolve, kalman, debug_map, statuses);
    drawUavDebug(canvas, t, detect, resolve, uav_resolve_debug_statuses);
    drawEventLog(canvas, event_log);

    cv::imshow(window_name_, canvas);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      rclcpp::shutdown();
    }
  }

  void drawTitle(cv::Mat& canvas,
                 const rclcpp::Time& t,
                 int8_t self_color,
                 bool self_color_valid,
                 const rclcpp::Time& self_color_stamp)
  {
    cv::putText(canvas, "Track Status Monitor", cv::Point(24, 36),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(240, 240, 240), 2,
                cv::LINE_AA);

    std::ostringstream subtitle;
    subtitle << "stale timeout " << std::fixed << std::setprecision(1)
             << stale_timeout_s_ << "s, time " << std::setprecision(3)
             << t.seconds() << ", self color "
             << selfColorLabel(self_color, self_color_valid, self_color_stamp, t);
    cv::putText(canvas, subtitle.str(), cv::Point(24, 62),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(170, 175, 180), 1,
                cv::LINE_AA);
  }

  std::string selfColorLabel(int8_t self_color,
                             bool self_color_valid,
                             const rclcpp::Time& self_color_stamp,
                             const rclcpp::Time& t) const
  {
    if (!self_color_valid) {
      return "UNKNOWN";
    }
    if (self_color_stamp.nanoseconds() != 0 &&
        std::fabs((t - self_color_stamp).seconds()) > stale_timeout_s_ &&
        !(self_color_override_ == 0 || self_color_override_ == 2)) {
      return "STALE";
    }
    if (self_color == 0) {
      return "BLUE";
    }
    if (self_color == 2) {
      return "RED";
    }
    return "UNKNOWN";
  }

  std::string modeDisplay(const std::string& mode) const
  {
    if (mode == "radar_vision_fusion") {
      return "RADAR+VISION";
    }
    if (mode == "radar_only") {
      return "RADAR";
    }
    if (mode == "pure_vision") {
      return "VISION";
    }
    if (mode == "default") {
      return "DEFAULT";
    }
    if (mode.rfind("kf_hold_", 0) == 0) {
      return "KF_HOLD";
    }
    return mode;
  }

  void drawTable(cv::Mat& canvas,
                 const rclcpp::Time& t,
                 const std::array<PointSample, kTrackSlots>& detect,
                 const std::array<PointSample, kTrackSlots>& resolve,
                 const std::array<PointSample, kTrackSlots>& kalman,
                 const std::array<PointSample, kTrackSlots>& debug_map,
                 const std::array<TrackStatus, kTrackSlots>& statuses)
  {
    const std::array<int, 7> cols{24, 92, 236, 310, 510, 710, 910};
    const int header_y = 100;
    const int row_h = 34;
    const int first_row_y = 132;

    cv::line(canvas, cv::Point(18, 76), cv::Point(1262, 76),
             cv::Scalar(60, 64, 70), 1, cv::LINE_AA);
    cv::rectangle(canvas, cv::Rect(18, 84, 1244, 30),
                  cv::Scalar(38, 42, 47), cv::FILLED);

    drawText(canvas, "ID", cols[0], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "State", cols[1], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Conf", cols[2], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Detect", cols[3], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Resolve", cols[4], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Kalman", cols[5], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "DebugMap", cols[6], header_y, 0.48, cv::Scalar(190, 198, 205), 1);

    for (int slot = 0; slot < kTrackSlots; ++slot) {
      const int y = first_row_y + slot * row_h;
      if (slot % 2 == 0) {
        cv::rectangle(canvas, cv::Rect(18, y - 22, 1244, row_h),
                      cv::Scalar(30, 33, 37), cv::FILLED);
      }
      if (slot == kRobotSlots) {
        cv::line(canvas, cv::Point(18, y - 28), cv::Point(1262, y - 28),
                 cv::Scalar(70, 76, 84), 1, cv::LINE_AA);
      }

      const bool red = slot >= kRobotSlots;
      const cv::Scalar id_color = red ? cv::Scalar(100, 120, 245)
                                      : cv::Scalar(245, 150, 80);
      const auto& status = statuses[slot];
      const bool status_fresh =
        status.valid && std::fabs((t - status.stamp).seconds()) <= stale_timeout_s_;
      std::string state = status_fresh ? status.state : "NO_STATUS";
      if (status_fresh) {
        if (!status.output_mode.empty() && status.output_mode != "NO_OUTPUT") {
          state += "/" + status.output_mode;
        }
        state += status.current_matched ? " M" : " -";
        if (status.publishable) {
          state += " P";
        }
        if (status.has_blind_guess) {
          char score[32];
          std::snprintf(score, sizeof(score), " G%.2f", status.blind_guess_score);
          state += score;
        }
      }

      drawText(canvas, robotLabel(slot), cols[0], y, 0.55, id_color, 2);
      drawText(canvas, state, cols[1], y, 0.46, stateColor(status.state), 1);
      drawText(canvas, formatConfidence(status, t, stale_timeout_s_), cols[2], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(detect[slot], t, stale_timeout_s_), cols[3], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(resolve[slot], t, stale_timeout_s_), cols[4], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(kalman[slot], t, stale_timeout_s_), cols[5], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(debug_map[slot], t, stale_timeout_s_), cols[6], y,
               0.46, cv::Scalar(218, 220, 224), 1);
    }
  }

  void drawUavDebug(cv::Mat& canvas,
                    const rclcpp::Time& t,
                    const std::array<PointSample, kTrackSlots>& detect,
                    const std::array<PointSample, kTrackSlots>& resolve,
                    const std::array<UavResolveDebugStatus, 2>& uav_statuses)
  {
    const int top = 620;
    const int header_y = top + 28;
    const int row1_y = top + 62;
    const int row2_y = top + 94;
    const std::array<int, 7> cols{24, 82, 240, 425, 610, 795, 990};

    cv::rectangle(canvas, cv::Rect(18, top - 24, 1244, 126),
                  cv::Scalar(31, 34, 38), cv::FILLED);
    cv::putText(canvas, "UAV Debug", cv::Point(24, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(190, 198, 205), 1,
                cv::LINE_AA);

    drawText(canvas, "ID", cols[0], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Mode", cols[1], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Detect(px)", cols[2], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "PureVision", cols[3], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Frustum", cols[4], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Cluster", cols[5], header_y, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Final", cols[6], header_y, 0.45, cv::Scalar(190, 198, 205), 1);

    auto draw_uav_row = [&](char color, int y) {
      const int slot = slotIndex(color, 6);
      const int side_idx = (color == 'B') ? 0 : 1;
      const bool red = color == 'R';
      const cv::Scalar id_color = red ? cv::Scalar(100, 120, 245)
                                      : cv::Scalar(245, 150, 80);
      const auto& status = uav_statuses[side_idx];
      const bool status_fresh =
        status.valid && freshStamp(status.stamp, t, stale_timeout_s_);
      const PureVisionResult pv = computePureVision(detect[slot], t);
      const bool vision_valid = status_fresh ? status.vision_valid : pv.computed;
      const bool vision_inside = status_fresh ? status.vision_inside : pv.inside_field;
      const double vision_x = status_fresh ? status.vision_x : pv.x;
      const double vision_y = status_fresh ? status.vision_y : pv.y;
      const double vision_z = status_fresh ? status.vision_z : pv.z;
      const cv::Scalar pv_color =
        vision_valid ? (vision_inside ? cv::Scalar(80, 210, 120)
                                      : cv::Scalar(90, 210, 245))
                     : cv::Scalar(150, 150, 150);
      const std::string pv_text = vision_valid
        ? formatFieldPoint(vision_x, vision_y, vision_z, vision_inside)
        : (pv.ray_ok ? "no plane hit" : "--");
      const std::string mode = status_fresh ? modeDisplay(status.mode) : "--";

      drawText(canvas, robotLabel(slot), cols[0], y, 0.52, id_color, 2);
      drawText(canvas, mode, cols[1], y, 0.41, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(detect[slot], t, stale_timeout_s_), cols[2], y,
               0.45, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, pv_text, cols[3], y, 0.41, pv_color, 1);
      drawText(canvas,
               status_fresh
                 ? formatDebugPoint(status.frustum_valid, status.frustum_x,
                                    status.frustum_y, status.frustum_z)
                 : "--",
               cols[4], y, 0.41, cv::Scalar(218, 220, 224), 1);
      drawText(canvas,
               status_fresh
                 ? formatDebugPoint(status.cluster_valid, status.cluster_x,
                                    status.cluster_y, status.cluster_z)
                 : "--",
               cols[5], y, 0.41, cv::Scalar(218, 220, 224), 1);
      drawText(canvas,
               status_fresh
                 ? formatDebugPoint(status.final_valid, status.final_x,
                                    status.final_y, status.final_z)
                 : formatPoint(resolve[slot], t, stale_timeout_s_),
               cols[6], y, 0.41, cv::Scalar(218, 220, 224), 1);
    };

    draw_uav_row('B', row1_y);
    draw_uav_row('R', row2_y);
  }

  void drawEventLog(cv::Mat& canvas, const std::deque<std::string>& event_log)
  {
    const int top = 775;
    cv::rectangle(canvas, cv::Rect(18, top - 24, 1244, 136),
                  cv::Scalar(31, 34, 38), cv::FILLED);
    cv::putText(canvas, "State Changes", cv::Point(24, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(190, 198, 205), 1,
                cv::LINE_AA);

    int y = top + 24;
    for (auto it = event_log.rbegin(); it != event_log.rend(); ++it) {
      drawText(canvas, *it, 28, y, 0.44, cv::Scalar(218, 220, 224), 1);
      y += 20;
    }
  }

  void drawText(cv::Mat& canvas,
                const std::string& text,
                int x,
                int y,
                double scale,
                const cv::Scalar& color,
                int thickness)
  {
    cv::putText(canvas, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv::LINE_AA);
  }

  std::string detect_topic_;
  std::string resolve_topic_;
  std::string kalman_topic_;
  std::string debug_map_topic_;
  std::string status_topic_;
  std::string identity_topic_;
  std::string uav_intersection_debug_topic_;
  std::string uav_cluster_debug_topic_;
  std::string uav_resolve_debug_status_topic_;
  std::string window_name_;
  double stale_timeout_s_{1.0};
  int8_t self_color_override_{-1};

  std::string out_matrix_path_;
  std::string camera_params_path_;
  std::string region_points_path_;
  double uav_default_z_{2.1};
  double uav_world_to_field_y_offset_{15.0};
  double uav_field_min_x_{0.0};
  double uav_field_max_x_{28.0};
  double uav_field_min_y_{0.0};
  double uav_field_max_y_{15.0};
  double uav_field_margin_m_{0.20};
  bool pure_vision_ready_{false};
  cv::Mat world_rvec_;
  cv::Mat world_tvec_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  std::mutex mutex_;
  std::array<PointSample, kTrackSlots> detect_samples_{};
  std::array<PointSample, kTrackSlots> resolve_samples_{};
  std::array<PointSample, kTrackSlots> kalman_samples_{};
  std::array<PointSample, kTrackSlots> debug_map_samples_{};
  std::array<TrackStatus, kTrackSlots> statuses_{};
  DebugCloudStatus uav_intersection_status_{};
  DebugCloudStatus uav_cluster_status_{};
  std::array<UavResolveDebugStatus, 2> uav_resolve_debug_statuses_{};
  int8_t self_color_{-1};
  bool self_color_valid_{false};
  rclcpp::Time self_color_stamp_{0, 0, RCL_ROS_TIME};
  std::deque<std::string> event_log_;

  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr detect_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr resolve_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr kalman_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr debug_map_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr
    uav_resolve_debug_status_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr
    identity_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    uav_intersection_debug_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    uav_cluster_debug_sub_;
  rclcpp::TimerBase::SharedPtr refresh_timer_;
};

}  // namespace debug_map

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<debug_map::TrackStatusWindowNode>());
  rclcpp::shutdown();
  return 0;
}
