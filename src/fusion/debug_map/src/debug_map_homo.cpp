#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <radar_decision_comm/msg/referee_robot_identity.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace debug_map_homo
{

namespace
{

int normalize_self_color(int raw_color, int fallback)
{
  if (raw_color == 0) {
    return 0;
  }
  if (raw_color == 1 || raw_color == 2) {
    return 2;
  }
  return (fallback == 0) ? 0 : 2;
}

}  // namespace

class DebugMapHomoNode : public rclcpp::Node
{
public:
  explicit DebugMapHomoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("debug_map_homo_node", options)
  {
    declare_parameter<std::string>("map_path", "");
    declare_parameter<int>("target_w", 700);   // 28 * 25
    declare_parameter<int>("target_h", 375);   // 15 * 25

    declare_parameter<std::string>("kalman_topic", "/kalman_detect");
    declare_parameter<std::string>("resolve_topic", "/resolve_result");
    declare_parameter<std::string>("uav_topic", "/uav_positions");
    declare_parameter<std::string>("match_info_topic", "/match_info");
    declare_parameter<std::string>("debug_image_topic", "/debug_map_homo");
    declare_parameter<std::string>("debug_map_result_topic", "/debug_map_result");

    declare_parameter<std::string>("homography_path", "");
    declare_parameter<bool>("auto_load_homography", true);
    declare_parameter<bool>("auto_save_homography", true);

    declare_parameter<int>("draw_timeout_ms", 500);
    declare_parameter<bool>("show_window", true);
    declare_parameter<bool>("prefer_kalman", true);
    declare_parameter<bool>("use_match_info_color", true);
    declare_parameter<int>("self_color_override", 0);
    declare_parameter<bool>("use_referee_identity_color", true);
    declare_parameter<std::string>("referee_identity_topic", "/radar/referee/robot_identity");

    map_path_ = get_parameter("map_path").as_string();
    target_w_ = get_parameter("target_w").as_int();
    target_h_ = get_parameter("target_h").as_int();

    kalman_topic_ = get_parameter("kalman_topic").as_string();
    resolve_topic_ = get_parameter("resolve_topic").as_string();
    uav_topic_ = get_parameter("uav_topic").as_string();
    match_info_topic_ = get_parameter("match_info_topic").as_string();
    debug_image_topic_ = get_parameter("debug_image_topic").as_string();
    debug_map_result_topic_ = get_parameter("debug_map_result_topic").as_string();

    homography_path_ = get_parameter("homography_path").as_string();
    auto_load_homography_ = get_parameter("auto_load_homography").as_bool();
    auto_save_homography_ = get_parameter("auto_save_homography").as_bool();

    draw_timeout_ms_ = get_parameter("draw_timeout_ms").as_int();
    show_window_ = get_parameter("show_window").as_bool();
    prefer_kalman_ = get_parameter("prefer_kalman").as_bool();
    use_match_info_color_ = get_parameter("use_match_info_color").as_bool();
    self_color_override_ =
      normalize_self_color(get_parameter("self_color_override").as_int(), 0);
    use_referee_identity_color_ = get_parameter("use_referee_identity_color").as_bool();
    referee_identity_topic_ = get_parameter("referee_identity_topic").as_string();

    if (map_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "parameter 'map_path' is empty");
      throw std::runtime_error("map_path is empty");
    }

    map_ = cv::imread(map_path_);
    if (map_.empty()) {
      RCLCPP_FATAL(get_logger(), "failed to load map image: %s", map_path_.c_str());
      throw std::runtime_error("failed to load map image");
    }

    if (target_w_ > 0 && target_h_ > 0) {
      cv::resize(map_, map_, cv::Size(target_w_, target_h_));
    }

    resetAllStates();

    if (auto_load_homography_ && !homography_path_.empty()) {
    loadHomographyFromFile();
    }

    kalman_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      kalman_topic_, 10,
      std::bind(&DebugMapHomoNode::onKalmanDetect, this, std::placeholders::_1));

    resolve_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      resolve_topic_, rclcpp::SensorDataQoS(),
      std::bind(&DebugMapHomoNode::onResolveDetect, this, std::placeholders::_1));

    match_info_sub_ = create_subscription<vision_interface::msg::MatchInfo>(
      match_info_topic_, 10,
      std::bind(&DebugMapHomoNode::onMatchInfo, this, std::placeholders::_1));

    referee_identity_sub_ = create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      referee_identity_topic_, 10,
      std::bind(&DebugMapHomoNode::onRefereeIdentity, this, std::placeholders::_1));

    uav_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      uav_topic_, rclcpp::SensorDataQoS(),
      std::bind(&DebugMapHomoNode::onUav, this, std::placeholders::_1));

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_image_topic_, 10);
    debug_map_result_pub_ =
      create_publisher<vision_interface::msg::DetectResult>(debug_map_result_topic_, 10);

    if (show_window_) {
      cv::namedWindow(kWindowName_, cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback(kWindowName_, &DebugMapHomoNode::mouseThunk, this);
    }

    using namespace std::chrono_literals;
    ui_timer_ = create_wall_timer(33ms, std::bind(&DebugMapHomoNode::onUiTimer, this));

    RCLCPP_INFO(
      get_logger(),
      "DebugMapHomoNode started.\n"
      "  map_path=%s\n"
      "  kalman_topic=%s\n"
      "  resolve_topic=%s\n"
      "  uav_topic=%s\n"
      "  match_info_topic=%s\n"
      "  debug_image_topic=%s\n"
      "  debug_map_result_topic=%s\n"
      "  Click 4 points and input world X/Y in the UI.\n"
      "  Keys: [R]=reset mapping, [ESC]=close window",
      map_path_.c_str(),
      kalman_topic_.c_str(),
      resolve_topic_.c_str(),
      uav_topic_.c_str(),
      match_info_topic_.c_str(),
      debug_image_topic_.c_str(),
      debug_map_result_topic_.c_str());
  }

private:
  static constexpr const char * kWindowName_ = "Debug Map Homo";

  struct TextInputState
  {
    bool active = false;
    int stage = 0;  // 0: input X, 1: input Y
    std::string x_str;
    std::string y_str;
    cv::Point2f clicked_px{0.0f, 0.0f};
  };

  bool loadHomographyFromFile()
{
  std::lock_guard<std::mutex> lk(mtx_);

  cv::FileStorage fs(homography_path_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN(
      get_logger(),
      "homography file not found or cannot open: %s, will enter 4-point input mode",
      homography_path_.c_str());
    return false;
  }

  cv::Mat H_p2w, H_w2p;
  fs["H_p2w"] >> H_p2w;
  fs["H_w2p"] >> H_w2p;
  fs.release();

  if (H_p2w.empty() || H_w2p.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "homography file invalid: %s, will enter 4-point input mode",
      homography_path_.c_str());
    return false;
  }

  H_p2w_ = H_p2w;
  H_w2p_ = H_w2p;
  mapping_ready_ = true;

  RCLCPP_INFO(
    get_logger(),
    "loaded homography from: %s",
    homography_path_.c_str());
  return true;
}

bool saveHomographyToFile()
{
  std::lock_guard<std::mutex> lk(mtx_);

  if (!mapping_ready_ || H_p2w_.empty() || H_w2p_.empty()) {
    RCLCPP_WARN(get_logger(), "homography not ready, skip save");
    return false;
  }

  if (homography_path_.empty()) {
    RCLCPP_WARN(get_logger(), "homography_path is empty, skip save");
    return false;
  }

  cv::FileStorage fs(homography_path_, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    RCLCPP_ERROR(get_logger(), "failed to open homography file for write: %s", homography_path_.c_str());
    return false;
  }

  fs << "H_p2w" << H_p2w_;
  fs << "H_w2p" << H_w2p_;
  fs.release();

  RCLCPP_INFO(get_logger(), "saved homography to: %s", homography_path_.c_str());
  return true;
}

  void resetAllStates()
  {
    std::lock_guard<std::mutex> lk(mtx_);

    for (int i = 0; i < 7; ++i) {
      kalman_blue_valid_[i] = false;
      kalman_red_valid_[i] = false;
      resolve_blue_valid_[i] = false;
      resolve_red_valid_[i] = false;

      kalman_blue_xy_[i] = cv::Point2f(0.0f, 0.0f);
      kalman_red_xy_[i] = cv::Point2f(0.0f, 0.0f);
      resolve_blue_xy_[i] = cv::Point2f(0.0f, 0.0f);
      resolve_red_xy_[i] = cv::Point2f(0.0f, 0.0f);

      kalman_blue_t_[i] = 0.0;
      kalman_red_t_[i] = 0.0;
      resolve_blue_t_[i] = 0.0;
      resolve_red_t_[i] = 0.0;
    }

    uav_valid_[0] = false;
    uav_valid_[1] = false;
    uav_xy_[0] = cv::Point2f(0.0f, 0.0f);
    uav_xy_[1] = cv::Point2f(0.0f, 0.0f);
    uav_t_[0] = 0.0;
    uav_t_[1] = 0.0;

    self_color_ = self_color_override_;

    mapping_ready_ = false;
    H_p2w_.release();
    H_w2p_.release();
    calib_px_.clear();
    calib_wxy_.clear();

    input_state_ = TextInputState{};
  }

  static double nowSec()
  {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
             system_clock::now().time_since_epoch()).count() / 1000.0;
  }

  static bool isFinite3(double x, double y, double z)
  {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  static bool looksLikeNumber(const std::string & s)
  {
    if (s.empty()) return false;
    bool has_digit = false;
    for (char c : s) {
      if (std::isdigit(static_cast<unsigned char>(c))) {
        has_digit = true;
      } else if (c == '.' || c == '-' || c == '+') {
      } else {
        return false;
      }
    }
    return has_digit;
  }

  void onKalmanDetect(const vision_interface::msg::DetectResult::SharedPtr msg)
  {
    if (!msg) return;
    const double t = nowSec();

    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < 7; ++i) {
      if (msg->blue_x[i] != 0.0f || msg->blue_y[i] != 0.0f) {
        kalman_blue_xy_[i] = cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
        kalman_blue_valid_[i] = true;
        kalman_blue_t_[i] = t;
      }
      if (msg->red_x[i] != 0.0f || msg->red_y[i] != 0.0f) {
        kalman_red_xy_[i] = cv::Point2f(msg->red_x[i], msg->red_y[i]);
        kalman_red_valid_[i] = true;
        kalman_red_t_[i] = t;
      }
    }
  }

  void onResolveDetect(const vision_interface::msg::DetectResult::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < 7; ++i) {
      resolve_blue_xy_[i] = cv::Point2f(0.0f, 0.0f);
      resolve_blue_valid_[i] = false;
      resolve_blue_t_[i] = 0.0;
      resolve_red_xy_[i] = cv::Point2f(0.0f, 0.0f);
      resolve_red_valid_[i] = false;
      resolve_red_t_[i] = 0.0;
    }
  }

  void onMatchInfo(const vision_interface::msg::MatchInfo::SharedPtr msg)
  {
    if (!msg) return;
    if (!use_match_info_color_) return;
    if (use_referee_identity_color_ && has_referee_identity_color_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    self_color_ = normalize_self_color(msg->self_color, self_color_override_);
  }

  void onRefereeIdentity(const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
  {
    if (!use_referee_identity_color_ || !msg || !msg->valid) return;
    std::lock_guard<std::mutex> lk(mtx_);
    has_referee_identity_color_ = true;
    self_color_ = normalize_self_color(msg->self_color, self_color_override_);
  }

  void onUav(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (!msg || msg->poses.size() < 2) return;
    const double t = nowSec();

    std::lock_guard<std::mutex> lk(mtx_);
    for (int idx = 0; idx < 2; ++idx) {
      const auto & p = msg->poses[idx].position;
      if (!isFinite3(p.x, p.y, p.z)) {
        uav_valid_[idx] = false;
        uav_xy_[idx] = cv::Point2f(0.0f, 0.0f);
        uav_t_[idx] = 0.0;
        continue;
      }
      uav_valid_[idx] = true;
      uav_xy_[idx] = cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
      uav_t_[idx] = t;
    }
  }

  static void mouseThunk(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    if (!userdata) return;
    auto * self = static_cast<DebugMapHomoNode *>(userdata);
    self->onMouse(event, x, y);
  }

  void onMouse(int event, int x, int y)
  {
    if (!show_window_) return;
    if (event != cv::EVENT_LBUTTONDOWN) return;

    std::lock_guard<std::mutex> lk(mtx_);
    if (mapping_ready_) return;
    if (input_state_.active) return;
    if (calib_px_.size() >= 4) return;

    input_state_.active = true;
    input_state_.stage = 0;
    input_state_.x_str.clear();
    input_state_.y_str.clear();
    input_state_.clicked_px = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
  }

void buildHomographyLocked()
{
  cv::Mat H = cv::findHomography(calib_px_, calib_wxy_, 0);
  if (H.empty()) {
    mapping_ready_ = false;
    calib_px_.clear();
    calib_wxy_.clear();
    input_state_ = TextInputState{};
    return;
  }

  H_p2w_ = H;
  H_w2p_ = H.inv();
  mapping_ready_ = true;

  if (auto_save_homography_) {
    // 注意：这里不要再次加锁调用带锁版本
    cv::FileStorage fs(homography_path_, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
      fs << "H_p2w" << H_p2w_;
      fs << "H_w2p" << H_w2p_;
      fs.release();
      RCLCPP_INFO(get_logger(), "saved homography to: %s", homography_path_.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "failed to save homography: %s", homography_path_.c_str());
    }
  }
}

  void resetMappingLocked()
  {
    mapping_ready_ = false;
    H_p2w_.release();
    H_w2p_.release();
    calib_px_.clear();
    calib_wxy_.clear();
    input_state_ = TextInputState{};
  }

  bool worldToPixelLocked(float wx, float wy, cv::Point2f & out_uv) const
  {
    if (!mapping_ready_ || H_w2p_.empty()) return false;

    cv::Mat X = (cv::Mat_<double>(3, 1) << static_cast<double>(wx), static_cast<double>(wy), 1.0);
    cv::Mat U = H_w2p_ * X;
    const double w = U.at<double>(2, 0);
    if (std::fabs(w) < 1e-12) return false;

    out_uv.x = static_cast<float>(U.at<double>(0, 0) / w);
    out_uv.y = static_cast<float>(U.at<double>(1, 0) / w);
    return true;
  }

  cv::Point2f applySelfColorFlipLocked(const cv::Point2f & pt) const
  {
    if (self_color_ == 0) {
      return cv::Point2f(28.0f - pt.x, 15.0f - pt.y);
    }
    return pt;
  }

  cv::Point2f worldToPixelFallback(const cv::Point2f & world) const
  {
    return cv::Point2f(
      map_.cols * world.x / 28.0f,
      map_.rows * (15.0f - world.y) / 15.0f);
  }

  bool chooseVehiclePointLocked(
    bool is_blue, int idx, cv::Point2f & out_world, bool & from_kalman) const
  {
    const double t = nowSec();
    const double timeout_s = static_cast<double>(draw_timeout_ms_) / 1000.0;

    bool kalman_ok = false;
    bool resolve_ok = false;

    if (is_blue) {
      kalman_ok = kalman_blue_valid_[idx] && (t - kalman_blue_t_[idx] < timeout_s);
      resolve_ok = resolve_blue_valid_[idx] && (t - resolve_blue_t_[idx] < timeout_s);
      if (prefer_kalman_) {
        if (kalman_ok) {
          out_world = kalman_blue_xy_[idx];
          from_kalman = true;
          return true;
        }
        if (resolve_ok) {
          out_world = resolve_blue_xy_[idx];
          from_kalman = false;
          return true;
        }
      } else {
        if (resolve_ok) {
          out_world = resolve_blue_xy_[idx];
          from_kalman = false;
          return true;
        }
        if (kalman_ok) {
          out_world = kalman_blue_xy_[idx];
          from_kalman = true;
          return true;
        }
      }
    } else {
      kalman_ok = kalman_red_valid_[idx] && (t - kalman_red_t_[idx] < timeout_s);
      resolve_ok = resolve_red_valid_[idx] && (t - resolve_red_t_[idx] < timeout_s);
      if (prefer_kalman_) {
        if (kalman_ok) {
          out_world = kalman_red_xy_[idx];
          from_kalman = true;
          return true;
        }
        if (resolve_ok) {
          out_world = resolve_red_xy_[idx];
          from_kalman = false;
          return true;
        }
      } else {
        if (resolve_ok) {
          out_world = resolve_red_xy_[idx];
          from_kalman = false;
          return true;
        }
        if (kalman_ok) {
          out_world = kalman_red_xy_[idx];
          from_kalman = true;
          return true;
        }
      }
    }

    return false;
  }

  vision_interface::msg::DetectResult buildDebugMapResultLocked(
    const rclcpp::Time & stamp) const
  {
    vision_interface::msg::DetectResult result;
    result.header.stamp = stamp;
    result.header.frame_id = "map";

    for (int i = 0; i < 7; ++i) {
      cv::Point2f world;
      bool from_kalman = false;
      if (chooseVehiclePointLocked(true, i, world, from_kalman)) {
        (void)from_kalman;
        world = applySelfColorFlipLocked(world);
        result.blue_x[i] = world.x;
        result.blue_y[i] = world.y;
      }
      if (chooseVehiclePointLocked(false, i, world, from_kalman)) {
        (void)from_kalman;
        world = applySelfColorFlipLocked(world);
        result.red_x[i] = world.x;
        result.red_y[i] = world.y;
      }
    }

    return result;
  }

  void drawVehicleLocked(cv::Mat & canvas, bool is_blue, int idx)
  {
    cv::Point2f world;
    bool from_kalman = false;
    if (!chooseVehiclePointLocked(is_blue, idx, world, from_kalman)) {
      return;
    }

    world = applySelfColorFlipLocked(world);

    cv::Point2f pixel;
    if (!worldToPixelLocked(world.x, world.y, pixel)) {
      pixel = worldToPixelFallback(world);
    }

    pixel.x = std::clamp(pixel.x, 0.0f, static_cast<float>(canvas.cols - 1));
    pixel.y = std::clamp(pixel.y, 0.0f, static_cast<float>(canvas.rows - 1));

    const cv::Scalar color = is_blue ? cv::Scalar(200, 0, 0) : cv::Scalar(0, 0, 200);
    cv::circle(canvas, pixel, 10, color, -1);

    // 参考旧文件：车辆编号仍然是 1~6
    const int number = idx + 1;
    cv::putText(
      canvas, std::to_string(number),
      cv::Point(static_cast<int>(pixel.x) - 6, static_cast<int>(pixel.y) + 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::putText(
      canvas, from_kalman ? "K" : "R",
      cv::Point(static_cast<int>(pixel.x) + 8, static_cast<int>(pixel.y) - 8),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
  }

  void drawUavLocked(cv::Mat & canvas, int idx)
  {
    const double t = nowSec();
    const double timeout_s = static_cast<double>(draw_timeout_ms_) / 1000.0;

    if (!uav_valid_[idx]) return;
    if (t - uav_t_[idx] >= timeout_s) return;

    cv::Point2f world = applySelfColorFlipLocked(uav_xy_[idx]);

    cv::Point2f pixel;
    if (!worldToPixelLocked(world.x, world.y, pixel)) {
      pixel = worldToPixelFallback(world);
    }

    pixel.x = std::clamp(pixel.x, 0.0f, static_cast<float>(canvas.cols - 1));
    pixel.y = std::clamp(pixel.y, 0.0f, static_cast<float>(canvas.rows - 1));

    const cv::Scalar color = (idx == 0) ? cv::Scalar(200, 0, 0) : cv::Scalar(0, 0, 200);
    const std::string label = (idx == 0) ? "UAV-B" : "UAV-R";

    cv::circle(canvas, pixel, 12, color, -1);
    cv::putText(
      canvas, label,
      cv::Point(static_cast<int>(pixel.x) - 20, static_cast<int>(pixel.y) + 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
  }

  void drawCalibrationPointsLocked(cv::Mat & canvas)
  {
    for (size_t i = 0; i < calib_px_.size(); ++i) {
      cv::circle(canvas, calib_px_[i], 6, cv::Scalar(0, 255, 255), -1);
      cv::putText(
        canvas, std::to_string(static_cast<int>(i) + 1),
        calib_px_[i] + cv::Point2f(8.0f, -8.0f),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
  }

  void drawStatusBarLocked(cv::Mat & canvas)
  {
    const std::string mapping_text = mapping_ready_ ? "Mapping READY" : "Mapping NOT READY";
    const cv::Scalar mapping_color = mapping_ready_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    cv::putText(
      canvas, mapping_text + "   [R]=reset  [ESC]=close",
      cv::Point(20, canvas.rows - 15),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, mapping_color, 2);

    std::string self_color_text = "self_color=" + std::to_string(self_color_);
    cv::putText(
      canvas, self_color_text,
      cv::Point(20, 25),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);
  }

  void drawInputPanelLocked(cv::Mat & canvas)
  {
    if (!input_state_.active) return;

    cv::rectangle(canvas, cv::Rect(20, 40, canvas.cols - 40, 120), cv::Scalar(10, 10, 10), -1);
    cv::rectangle(canvas, cv::Rect(20, 40, canvas.cols - 40, 120), cv::Scalar(230, 230, 230), 2);

    const int idx = static_cast<int>(calib_px_.size()) + 1;

    cv::putText(
      canvas,
      "Homography Point " + std::to_string(idx) + "/4   Pixel=(" +
      std::to_string(static_cast<int>(input_state_.clicked_px.x)) + "," +
      std::to_string(static_cast<int>(input_state_.clicked_px.y)) + ")",
      cv::Point(35, 75),
      cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

    std::string line;
    if (input_state_.stage == 0) {
      line = "Input world X: " + input_state_.x_str;
    } else {
      line = "Input world Y: " + input_state_.y_str;
    }

    cv::putText(
      canvas, line,
      cv::Point(35, 120),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
  }

  void pumpInputKeyLocked(int key)
  {
    if (!input_state_.active) return;

    if (key == 27) {
      input_state_ = TextInputState{};
      return;
    }

    if (key == 8 || key == 127) {
      if (input_state_.stage == 0) {
        if (!input_state_.x_str.empty()) input_state_.x_str.pop_back();
      } else {
        if (!input_state_.y_str.empty()) input_state_.y_str.pop_back();
      }
      return;
    }

    if (key == 13 || key == 10) {
      if (input_state_.stage == 0) {
        if (!looksLikeNumber(input_state_.x_str)) return;
        input_state_.stage = 1;
        return;
      } else {
        if (!looksLikeNumber(input_state_.y_str)) return;

        const double wx = std::stod(input_state_.x_str);
        const double wy = std::stod(input_state_.y_str);

        calib_px_.push_back(input_state_.clicked_px);
        calib_wxy_.push_back(cv::Point2f(static_cast<float>(wx), static_cast<float>(wy)));

        input_state_ = TextInputState{};

        if (calib_px_.size() == 4) {
          buildHomographyLocked();
        }
        return;
      }
    }

    if (key >= 32 && key <= 126) {
      const char c = static_cast<char>(key);
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '+') {
        if (input_state_.stage == 0) {
          input_state_.x_str.push_back(c);
        } else {
          input_state_.y_str.push_back(c);
        }
      }
    }
  }

  void onUiTimer()
  {
    cv::Mat canvas;
    vision_interface::msg::DetectResult debug_map_result;
    {
      std::lock_guard<std::mutex> lk(mtx_);

      canvas = map_.clone();

      drawStatusBarLocked(canvas);
      drawCalibrationPointsLocked(canvas);
      drawInputPanelLocked(canvas);

      for (int i = 0; i < 6; ++i) {
        drawVehicleLocked(canvas, true, i);
        drawVehicleLocked(canvas, false, i);
      }

      // drawUavLocked(canvas, 0);
      // drawUavLocked(canvas, 1);

      debug_map_result = buildDebugMapResultLocked(now());

      std_msgs::msg::Header header;
      header.stamp = now();
      header.frame_id = "map";
      try {
        auto msg = cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg();
        image_pub_->publish(*msg);
      } catch (const cv_bridge::Exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "cv_bridge conversion failed: %s", e.what());
      }
    }

    debug_map_result_pub_->publish(debug_map_result);

    if (show_window_) {
      cv::imshow(kWindowName_, canvas);
      const int key = cv::waitKey(1);

      std::lock_guard<std::mutex> lk(mtx_);
      if (key == 'r' || key == 'R') {
        resetMappingLocked();
      } else if (input_state_.active) {
        pumpInputKeyLocked(key);
      } else if (key == 27) {
        cv::destroyWindow(kWindowName_);
        show_window_ = false;
      }
    }
  }

private:
  mutable std::mutex mtx_;

  std::string map_path_;
  int target_w_{700};
  int target_h_{375};

  std::string kalman_topic_;
  std::string resolve_topic_;
  std::string uav_topic_;
  std::string match_info_topic_;
  std::string debug_image_topic_;
  std::string debug_map_result_topic_;
  std::string referee_identity_topic_;



    std::string homography_path_;
bool auto_load_homography_{true};
bool auto_save_homography_{true};

  int draw_timeout_ms_{500};
  bool show_window_{true};
  bool prefer_kalman_{true};
  bool use_match_info_color_{true};
  bool use_referee_identity_color_{true};
  bool has_referee_identity_color_{false};
  int self_color_override_{0};

  cv::Mat map_;
  int self_color_{0};

  std::array<cv::Point2f, 7> kalman_blue_xy_{};
  std::array<cv::Point2f, 7> kalman_red_xy_{};
  std::array<bool, 7> kalman_blue_valid_{};
  std::array<bool, 7> kalman_red_valid_{};
  std::array<double, 7> kalman_blue_t_{};
  std::array<double, 7> kalman_red_t_{};

  std::array<cv::Point2f, 7> resolve_blue_xy_{};
  std::array<cv::Point2f, 7> resolve_red_xy_{};
  std::array<bool, 7> resolve_blue_valid_{};
  std::array<bool, 7> resolve_red_valid_{};
  std::array<double, 7> resolve_blue_t_{};
  std::array<double, 7> resolve_red_t_{};

  std::array<cv::Point2f, 2> uav_xy_{};
  std::array<bool, 2> uav_valid_{};
  std::array<double, 2> uav_t_{};

  bool mapping_ready_{false};
  cv::Mat H_p2w_;
  cv::Mat H_w2p_;
  std::vector<cv::Point2f> calib_px_;
  std::vector<cv::Point2f> calib_wxy_;
  TextInputState input_state_;

  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr kalman_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr resolve_sub_;
  rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr match_info_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr
    referee_identity_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr uav_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr debug_map_result_pub_;
  rclcpp::TimerBase::SharedPtr ui_timer_;
};

}  // namespace debug_map_homo

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<debug_map_homo::DebugMapHomoNode>());
  rclcpp::shutdown();
  return 0;
}
