#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "gimbal_aim_solver/angle_solver.hpp"
#include "gimbal_aim_solver/gimbal_geometry.hpp"
#include "gimbal_aim_solver/types.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kDegToRad = kPi / 180.0;

struct Angles
{
  double yaw_deg{0.0};
  double pitch_deg{0.0};
  bool valid{false};
};

struct Sample
{
  cv::Point2d pixel_cam0;
  Eigen::Vector3d point_cam0;
  double coarse_yaw_deg{0.0};
  double coarse_pitch_deg{0.0};
  double manual_delta_yaw_deg{0.0};
  double manual_delta_pitch_deg{0.0};
  double ideal_yaw_deg{0.0};
  double ideal_pitch_deg{0.0};
};

bool isFinite(double v)
{
  return std::isfinite(v);
}

double normalizeAngleDeg(double a)
{
  while (a > 180.0) a -= 360.0;
  while (a < -180.0) a += 360.0;
  return a;
}

}  // namespace

class GimbalCameraTranslationDebugNode final : public rclcpp::Node
{
public:
  GimbalCameraTranslationDebugNode()
  : Node("gimbal_camera_translation_debug_node")
  {
    declareParams();
    loadParams();

    cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic_, 10);
    enable_pub_ = create_publisher<std_msgs::msg::Bool>(enable_topic_, 10);

    cam0_sub_ = create_subscription<sensor_msgs::msg::Image>(
      cam0_image_topic_, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::SharedPtr msg) { imageCallback(msg, true); });
    cam2_sub_ = create_subscription<sensor_msgs::msg::Image>(
      cam2_image_topic_, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::SharedPtr msg) { imageCallback(msg, false); });
    cam0_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      cam0_info_topic_, rclcpp::QoS(10),
      std::bind(&GimbalCameraTranslationDebugNode::cam0InfoCallback, this, std::placeholders::_1));
    feedback_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      feedback_topic_, rclcpp::SensorDataQoS(),
      std::bind(&GimbalCameraTranslationDebugNode::feedbackCallback, this, std::placeholders::_1));

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name_, &GimbalCameraTranslationDebugNode::mouseThunk, this);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(30),
      std::bind(&GimbalCameraTranslationDebugNode::onTimer, this));

    RCLCPP_INFO(
      get_logger(),
      "gimbal camera translation debug started. cam0=%s cam2=%s cmd_topic=%s feedback=%s",
      cam0_image_topic_.c_str(), cam2_image_topic_.c_str(), cmd_topic_.c_str(),
      feedback_topic_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "Keys: mouse-left select cam0, arrows refine cam0 pixel, M zoom, WASD tune gimbal by %.3f deg, R/SPACE record, P solve, Q quit",
      manual_step_deg_);
  }

  ~GimbalCameraTranslationDebugNode() override
  {
    cv::destroyWindow(window_name_);
  }

private:
  void declareParams()
  {
    declare_parameter<std::string>("cam0_image_topic", "/camera0/image");
    declare_parameter<std::string>("cam2_image_topic", "/camera2/image");
    declare_parameter<std::string>("cam0_info_topic", "/camera0/camera_info");
    declare_parameter<std::string>("cmd_topic", "/gimbal/rough_command_angles");
    declare_parameter<std::string>("enable_topic", "/gimbal/enable");
    declare_parameter<std::string>("feedback_topic", "/gimbal/angles_true");
    declare_parameter<std::string>("gimbal_angle_order", "yaw_pitch");
    declare_parameter<int>("yaw_index", 0);
    declare_parameter<int>("pitch_index", 1);
    declare_parameter<int>("feedback_yaw_index", 0);
    declare_parameter<int>("feedback_pitch_index", 1);
    declare_parameter<int>("publish_dim", 2);
    declare_parameter<double>("default_depth_m", 20.0);
    declare_parameter<double>("manual_step_deg", 0.05);
    declare_parameter<int>("max_display_width", 1800);
    declare_parameter<int>("max_display_height", 950);
    declare_parameter<bool>("publish_enable", true);
    declare_parameter<bool>("enable_value", true);
    declare_parameter<double>("cam0_fx", 0.0);
    declare_parameter<double>("cam0_fy", 0.0);
    declare_parameter<double>("cam0_cx", 0.0);
    declare_parameter<double>("cam0_cy", 0.0);
    declare_parameter<std::vector<double>>(
      "camera_to_gimbal_xyz", std::vector<double>{-0.05, -0.20, 0.0});
    declare_parameter<double>("camera_to_gimbal_pitch_deg", 0.0);
    declare_parameter<std::vector<double>>("laser_offset_xyz", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter<double>("yaw_offset_deg", 0.0);
    declare_parameter<double>("pitch_offset_deg", 0.0);
    declare_parameter<double>("yaw_min_deg", -180.0);
    declare_parameter<double>("yaw_max_deg", 180.0);
    declare_parameter<double>("pitch_min_deg", -60.0);
    declare_parameter<double>("pitch_max_deg", 60.0);
    declare_parameter<std::string>("camera_to_lidar_yaml", "");
  }

  void loadParams()
  {
    cam0_image_topic_ = get_parameter("cam0_image_topic").as_string();
    cam2_image_topic_ = get_parameter("cam2_image_topic").as_string();
    cam0_info_topic_ = get_parameter("cam0_info_topic").as_string();
    cmd_topic_ = get_parameter("cmd_topic").as_string();
    enable_topic_ = get_parameter("enable_topic").as_string();
    feedback_topic_ = get_parameter("feedback_topic").as_string();
    gimbal_angle_order_ = get_parameter("gimbal_angle_order").as_string();
    yaw_index_ = static_cast<int>(get_parameter("yaw_index").as_int());
    pitch_index_ = static_cast<int>(get_parameter("pitch_index").as_int());
    feedback_yaw_index_ = static_cast<int>(get_parameter("feedback_yaw_index").as_int());
    feedback_pitch_index_ = static_cast<int>(get_parameter("feedback_pitch_index").as_int());
    publish_dim_ = std::max(2, static_cast<int>(get_parameter("publish_dim").as_int()));
    default_depth_m_ = std::max(0.1, get_parameter("default_depth_m").as_double());
    manual_step_deg_ = std::max(1e-4, get_parameter("manual_step_deg").as_double());
    max_display_width_ = std::max(320, static_cast<int>(get_parameter("max_display_width").as_int()));
    max_display_height_ =
      std::max(240, static_cast<int>(get_parameter("max_display_height").as_int()));
    publish_enable_ = get_parameter("publish_enable").as_bool();
    enable_value_ = get_parameter("enable_value").as_bool();

    fx_ = get_parameter("cam0_fx").as_double();
    fy_ = get_parameter("cam0_fy").as_double();
    cx_ = get_parameter("cam0_cx").as_double();
    cy_ = get_parameter("cam0_cy").as_double();
    have_cam0_info_ = fx_ > 0.0 && fy_ > 0.0;

    const auto xyz = get_parameter("camera_to_gimbal_xyz").as_double_array();
    if (xyz.size() == 3) {
      camera_to_gimbal_xyz_ = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
    }
    gimbal_geometry_.setCameraToGimbalTranslation(camera_to_gimbal_xyz_);

    const auto laser_offset = get_parameter("laser_offset_xyz").as_double_array();
    if (laser_offset.size() == 3) {
      laser_offset_xyz_ = Eigen::Vector3d(laser_offset[0], laser_offset[1], laser_offset[2]);
    }

    angle_solver_.setOffsets(
      get_parameter("yaw_offset_deg").as_double(),
      get_parameter("pitch_offset_deg").as_double());
    angle_solver_.setLimits(
      get_parameter("yaw_min_deg").as_double(),
      get_parameter("yaw_max_deg").as_double(),
      get_parameter("pitch_min_deg").as_double(),
      get_parameter("pitch_max_deg").as_double());

    double pitch_deg = get_parameter("camera_to_gimbal_pitch_deg").as_double();
    const std::string camera_to_lidar_yaml = get_parameter("camera_to_lidar_yaml").as_string();
    double yaml_pitch_deg = 0.0;
    if (!camera_to_lidar_yaml.empty() && readPitchFromYaml(camera_to_lidar_yaml, yaml_pitch_deg)) {
      pitch_deg = yaml_pitch_deg;
    }
    setCameraPitch(pitch_deg);
  }

  bool readPitchFromYaml(const std::string & path, double & pitch_deg) const
  {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      return false;
    }
    if (!fs["rotation_rpy_deg"].empty()) {
      cv::FileNode rpy = fs["rotation_rpy_deg"];
      if (rpy.size() == 3) {
        pitch_deg = static_cast<double>(rpy[1]);
        return true;
      }
    }
    if (!fs["rotation_rpy_rad"].empty()) {
      cv::FileNode rpy = fs["rotation_rpy_rad"];
      if (rpy.size() == 3) {
        pitch_deg = static_cast<double>(rpy[1]) * kRadToDeg;
        return true;
      }
    }
    if (!fs["pitch_deg"].empty()) {
      pitch_deg = static_cast<double>(fs["pitch_deg"]);
      return true;
    }
    if (!fs["pitch"].empty()) {
      pitch_deg = static_cast<double>(fs["pitch"]) * kRadToDeg;
      return true;
    }
    return false;
  }

  void setCameraPitch(double pitch_deg)
  {
    const Eigen::AngleAxisd camera_from_gimbal_pitch(
      pitch_deg * kDegToRad, Eigen::Vector3d::UnitX());
    R_gimbal_from_camera_ = camera_from_gimbal_pitch.inverse().toRotationMatrix();
    gimbal_geometry_.setCameraPitchRelativeToGimbal(pitch_deg);
    camera_to_gimbal_pitch_deg_ = pitch_deg;
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr & msg, bool cam0)
  {
    try {
      cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
      std::lock_guard<std::mutex> lk(image_mutex_);
      if (cam0) {
        cam0_image_ = image;
      } else {
        cam2_image_ = image;
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "image conversion failed: %s", e.what());
    }
  }

  void cam0InfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (!msg || msg->k[0] <= 0.0 || msg->k[4] <= 0.0) {
      return;
    }
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    have_cam0_info_ = true;
  }

  void feedbackCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (!msg || msg->data.empty()) {
      return;
    }
    const int n = static_cast<int>(msg->data.size());
    if (feedback_yaw_index_ >= n || feedback_pitch_index_ >= n) {
      return;
    }
    const double yaw = msg->data[static_cast<size_t>(feedback_yaw_index_)];
    const double pitch = msg->data[static_cast<size_t>(feedback_pitch_index_)];
    if (!isFinite(yaw) || !isFinite(pitch)) {
      return;
    }
    latest_feedback_yaw_deg_ = yaw;
    latest_feedback_pitch_deg_ = pitch;
    have_feedback_ = true;
    if (!have_command_) {
      command_yaw_deg_ = yaw;
      command_pitch_deg_ = pitch;
      have_command_ = true;
    }
  }

  static void mouseThunk(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    auto * self = static_cast<GimbalCameraTranslationDebugNode *>(userdata);
    if (self != nullptr) {
      self->onMouse(event, x, y);
    }
  }

  void onMouse(int event, int x, int y)
  {
    if (event != cv::EVENT_LBUTTONDOWN) {
      return;
    }
    cv::Mat cam0;
    {
      std::lock_guard<std::mutex> lk(image_mutex_);
      if (!cam0_image_.empty()) {
        cam0 = cam0_image_.clone();
      }
    }
    cv::Point2d raw(static_cast<double>(x), static_cast<double>(y));
    if (display_scale_ > 1e-9) {
      raw.x /= display_scale_;
      raw.y /= display_scale_;
    }

    if (raw.x < 0.0 || raw.y < 0.0 || raw.x >= cam0.cols || raw.y >= cam0.rows) {
      return;
    }
    selected_pixel_ = raw;
    coarseAimSelectedPixel();
  }

  Eigen::Vector3d pixelToCameraPoint(const cv::Point2d & px) const
  {
    if (!have_cam0_info_) {
      return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    const double z = default_depth_m_;
    const double x = (px.x - cx_) / fx_ * z;
    const double y = (px.y - cy_) / fy_ * z;
    return Eigen::Vector3d(x, y, z);
  }

  Eigen::Matrix3d rotationFromSolvedYawPitchDeg(double yaw_deg, double pitch_deg) const
  {
    const double yaw_rad = yaw_deg * kDegToRad;
    const double pitch_rad = pitch_deg * kDegToRad;
    const Eigen::AngleAxisd yaw_rot(-yaw_rad, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd pitch_rot(pitch_rad, Eigen::Vector3d::UnitX());
    return (yaw_rot * pitch_rot).toRotationMatrix();
  }

  Angles anglesForPointAndTranslation(
    const Eigen::Vector3d & p_camera,
    const Eigen::Vector3d & t_camera_to_gimbal) const
  {
    gimbal_aim_solver::GimbalGeometry geometry;
    geometry.setCameraToGimbalTranslation(t_camera_to_gimbal);
    geometry.setCameraPitchRelativeToGimbal(camera_to_gimbal_pitch_deg_);
    const Eigen::Vector3d p_gimbal = geometry.cameraToGimbal(p_camera);

    gimbal_aim_solver::TargetAngles raw_target = angle_solver_.solve(p_gimbal);
    if (!raw_target.valid) {
      return {};
    }

    const Eigen::Matrix3d R_gimbal =
      rotationFromSolvedYawPitchDeg(raw_target.yaw_deg, raw_target.pitch_deg);
    const Eigen::Vector3d laser_offset_in_static_gimbal = R_gimbal * laser_offset_xyz_;
    const Eigen::Vector3d p_from_laser = p_gimbal - laser_offset_in_static_gimbal;
    const gimbal_aim_solver::TargetAngles target = angle_solver_.solve(p_from_laser);
    if (!target.valid) {
      return {};
    }

    return Angles{target.yaw_deg, target.pitch_deg, true};
  }

  void coarseAimSelectedPixel()
  {
    if (!selected_pixel_) {
      return;
    }
    const Eigen::Vector3d p_camera = pixelToCameraPoint(*selected_pixel_);
    const Angles a = anglesForPointAndTranslation(p_camera, camera_to_gimbal_xyz_);
    if (!a.valid) {
      RCLCPP_WARN(get_logger(), "selected pixel cannot produce valid gimbal command");
      return;
    }
    coarse_yaw_deg_ = a.yaw_deg;
    coarse_pitch_deg_ = a.pitch_deg;
    manual_delta_yaw_deg_ = 0.0;
    manual_delta_pitch_deg_ = 0.0;
    have_coarse_command_ = true;
    updateCommandFromCoarseAndManual();
    have_command_ = true;
    publishCommand();
    RCLCPP_INFO(
      get_logger(),
      "coarse aim pixel=(%.1f, %.1f), p_cam=(%.3f, %.3f, %.3f), coarse yaw=%.3f pitch=%.3f",
      selected_pixel_->x, selected_pixel_->y, p_camera.x(), p_camera.y(), p_camera.z(),
      coarse_yaw_deg_, coarse_pitch_deg_);
  }

  void updateCommandFromCoarseAndManual()
  {
    if (!have_coarse_command_) {
      return;
    }
    command_yaw_deg_ = coarse_yaw_deg_ + manual_delta_yaw_deg_;
    command_pitch_deg_ = coarse_pitch_deg_ + manual_delta_pitch_deg_;
  }

  void publishCommand()
  {
    if (!have_command_) {
      return;
    }
    std_msgs::msg::Float64MultiArray msg;
    msg.data.assign(static_cast<size_t>(publish_dim_), 0.0);
    msg.data[static_cast<size_t>(std::clamp(yaw_index_, 0, publish_dim_ - 1))] = command_yaw_deg_;
    msg.data[static_cast<size_t>(std::clamp(pitch_index_, 0, publish_dim_ - 1))] = command_pitch_deg_;
    cmd_pub_->publish(msg);

    if (publish_enable_) {
      std_msgs::msg::Bool en;
      en.data = enable_value_;
      enable_pub_->publish(en);
    }
  }

  void recordSample()
  {
    if (!selected_pixel_) {
      RCLCPP_WARN(get_logger(), "select a cam0 pixel first");
      return;
    }
    if (!have_cam0_info_) {
      RCLCPP_WARN(get_logger(), "no cam0 CameraInfo or intrinsics parameters");
      return;
    }
    if (!have_coarse_command_) {
      RCLCPP_WARN(get_logger(), "no coarse command yet; click/refine a cam0 pixel first");
      return;
    }

    Sample s;
    s.pixel_cam0 = *selected_pixel_;
    s.point_cam0 = pixelToCameraPoint(*selected_pixel_);
    s.coarse_yaw_deg = coarse_yaw_deg_;
    s.coarse_pitch_deg = coarse_pitch_deg_;
    s.manual_delta_yaw_deg = manual_delta_yaw_deg_;
    s.manual_delta_pitch_deg = manual_delta_pitch_deg_;
    s.ideal_yaw_deg = command_yaw_deg_;
    s.ideal_pitch_deg = command_pitch_deg_;
    samples_.push_back(s);
    RCLCPP_INFO(
      get_logger(),
      "recorded sample %zu: pixel=(%.1f, %.1f), coarse=(%.3f, %.3f), manual_delta=(%.3f, %.3f), ideal=(%.3f, %.3f)%s",
      samples_.size(), s.pixel_cam0.x, s.pixel_cam0.y,
      s.coarse_yaw_deg, s.coarse_pitch_deg,
      s.manual_delta_yaw_deg, s.manual_delta_pitch_deg,
      s.ideal_yaw_deg, s.ideal_pitch_deg,
      have_feedback_ ? "" : " (no feedback)");
  }

  Eigen::VectorXd residualForTranslation(const Eigen::Vector3d & t) const
  {
    Eigen::VectorXd r(static_cast<int>(samples_.size() * 2));
    for (size_t i = 0; i < samples_.size(); ++i) {
      const Angles a = anglesForPointAndTranslation(samples_[i].point_cam0, t);
      if (!a.valid) {
        r(static_cast<int>(2 * i)) = 1e3;
        r(static_cast<int>(2 * i + 1)) = 1e3;
        continue;
      }
      r(static_cast<int>(2 * i)) = normalizeAngleDeg(a.yaw_deg - samples_[i].ideal_yaw_deg);
      r(static_cast<int>(2 * i + 1)) = a.pitch_deg - samples_[i].ideal_pitch_deg;
    }
    return r;
  }

  void solveTranslation()
  {
    if (samples_.size() < 20) {
      RCLCPP_WARN(get_logger(), "need at least 20 samples, current=%zu", samples_.size());
      return;
    }

    Eigen::Vector3d t = camera_to_gimbal_xyz_;
    double last_rmse = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < 50; ++iter) {
      const Eigen::VectorXd r = residualForTranslation(t);
      const double rmse = std::sqrt(r.squaredNorm() / static_cast<double>(r.size()));

      Eigen::MatrixXd J(r.size(), 3);
      const double eps = 1e-4;
      for (int j = 0; j < 3; ++j) {
        Eigen::Vector3d tp = t;
        tp(j) += eps;
        Eigen::Vector3d tm = t;
        tm(j) -= eps;
        J.col(j) = (residualForTranslation(tp) - residualForTranslation(tm)) / (2.0 * eps);
      }

      Eigen::Matrix3d H = J.transpose() * J;
      H += Eigen::Matrix3d::Identity() * 1e-6;
      const Eigen::Vector3d g = J.transpose() * r;
      const Eigen::Vector3d step = -H.ldlt().solve(g);
      if (!step.allFinite()) {
        RCLCPP_WARN(get_logger(), "optimization failed: non-finite step");
        return;
      }

      t += step;
      if (step.norm() < 1e-7 || std::abs(last_rmse - rmse) < 1e-7) {
        break;
      }
      last_rmse = rmse;
    }

    const Eigen::VectorXd final_r = residualForTranslation(t);
    const double final_rmse =
      std::sqrt(final_r.squaredNorm() / static_cast<double>(final_r.size()));
    const Eigen::VectorXd initial_r = residualForTranslation(camera_to_gimbal_xyz_);
    const double initial_rmse =
      std::sqrt(initial_r.squaredNorm() / static_cast<double>(initial_r.size()));

    RCLCPP_INFO(get_logger(), "========== camera_to_gimbal_xyz optimization ==========");
    RCLCPP_INFO(
      get_logger(), "samples=%zu fixed_camera_pitch_deg=%.6f depth=%.3f m",
      samples_.size(), camera_to_gimbal_pitch_deg_, default_depth_m_);
    RCLCPP_INFO(
      get_logger(), "initial xyz: [%.6f, %.6f, %.6f], angle_rmse=%.6f deg",
      camera_to_gimbal_xyz_.x(), camera_to_gimbal_xyz_.y(), camera_to_gimbal_xyz_.z(),
      initial_rmse);
    RCLCPP_INFO(
      get_logger(), "optimized xyz: [%.6f, %.6f, %.6f], angle_rmse=%.6f deg",
      t.x(), t.y(), t.z(), final_rmse);
    RCLCPP_INFO(
      get_logger(), "delta xyz: [%.6f, %.6f, %.6f]",
      t.x() - camera_to_gimbal_xyz_.x(),
      t.y() - camera_to_gimbal_xyz_.y(),
      t.z() - camera_to_gimbal_xyz_.z());
    RCLCPP_INFO(get_logger(), "=======================================================");
  }

  void handleKey(int key)
  {
    if (key < 0) {
      return;
    }
    const int k = key & 0xFF;
    if (k == 'q' || k == 'Q') {
      rclcpp::shutdown();
      return;
    }
    if (k == 'm' || k == 'M') {
      zoom_cam0_ = !zoom_cam0_;
      return;
    }
    if (k == 'r' || k == 'R' || k == ' ') {
      recordSample();
      return;
    }
    if (k == 'p' || k == 'P') {
      solveTranslation();
      return;
    }

    bool cmd_changed = false;
    if (k == 'a' || k == 'A') {
      manual_delta_yaw_deg_ += manual_step_deg_;
      cmd_changed = true;
    } else if (k == 'd' || k == 'D') {
      manual_delta_yaw_deg_ -= manual_step_deg_;
      cmd_changed = true;
    } else if (k == 'w' || k == 'W') {
      manual_delta_pitch_deg_ += manual_step_deg_;
      cmd_changed = true;
    } else if (k == 's' || k == 'S') {
      manual_delta_pitch_deg_ -= manual_step_deg_;
      cmd_changed = true;
    }
    if (cmd_changed) {
      if (!have_coarse_command_) {
        coarse_yaw_deg_ = command_yaw_deg_;
        coarse_pitch_deg_ = command_pitch_deg_;
        have_coarse_command_ = true;
      }
      updateCommandFromCoarseAndManual();
      have_command_ = true;
      publishCommand();
      return;
    }

    if (!selected_pixel_) {
      return;
    }
    bool pixel_changed = false;
    // OpenCV common special key values after waitKeyEx are backend-specific.
    if (key == 81 || key == 2424832) {           // left
      selected_pixel_->x -= 1.0;
      pixel_changed = true;
    } else if (key == 83 || key == 2555904) {    // right
      selected_pixel_->x += 1.0;
      pixel_changed = true;
    } else if (key == 82 || key == 2490368) {    // up
      selected_pixel_->y -= 1.0;
      pixel_changed = true;
    } else if (key == 84 || key == 2621440) {    // down
      selected_pixel_->y += 1.0;
      pixel_changed = true;
    }
    if (pixel_changed) {
      coarseAimSelectedPixel();
    }
  }

  cv::Mat drawCam0(const cv::Mat & src)
  {
    cv::Mat out = src.clone();
    if (selected_pixel_) {
      cv::drawMarker(out, *selected_pixel_, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 24, 2);
      cv::circle(out, *selected_pixel_, 8, cv::Scalar(0, 255, 255), 2);
    }
    if (!zoom_cam0_ || !selected_pixel_) {
      return out;
    }

    const int crop_w = std::min(320, out.cols);
    const int crop_h = std::min(240, out.rows);
    int x0 = static_cast<int>(selected_pixel_->x) - crop_w / 2;
    int y0 = static_cast<int>(selected_pixel_->y) - crop_h / 2;
    x0 = std::clamp(x0, 0, std::max(0, out.cols - crop_w));
    y0 = std::clamp(y0, 0, std::max(0, out.rows - crop_h));
    cv::Mat crop = out(cv::Rect(x0, y0, crop_w, crop_h)).clone();
    cv::resize(crop, crop, out.size(), 0.0, 0.0, cv::INTER_LINEAR);
    return crop;
  }

  cv::Mat panelImage()
  {
    cv::Mat cam0;
    cv::Mat cam2;
    {
      std::lock_guard<std::mutex> lk(image_mutex_);
      cam0 = cam0_image_.empty() ? cv::Mat(720, 1280, CV_8UC3, cv::Scalar(30, 30, 30)) :
        cam0_image_.clone();
      cam2 = cam2_image_.empty() ? cv::Mat(cam0.rows, cam0.cols, CV_8UC3, cv::Scalar(30, 30, 30)) :
        cam2_image_.clone();
    }
    if (cam2.size() != cam0.size()) {
      cv::resize(cam2, cam2, cam0.size());
    }
    cam0 = drawCam0(cam0);

    cv::putText(cam0, "cam0", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
    cv::putText(cam2, "cam2", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

    cv::Mat merged;
    cv::hconcat(cam0, cam2, merged);
    const std::string status =
      "samples=" + std::to_string(samples_.size()) +
      " coarse=(" + std::to_string(coarse_yaw_deg_).substr(0, 6) + "," +
      std::to_string(coarse_pitch_deg_).substr(0, 6) + ") " +
      "delta=(" + std::to_string(manual_delta_yaw_deg_).substr(0, 6) + "," +
      std::to_string(manual_delta_pitch_deg_).substr(0, 6) + ") " +
      "cmd=(" + std::to_string(command_yaw_deg_).substr(0, 6) + "," +
      std::to_string(command_pitch_deg_).substr(0, 6) + ")";
    cv::putText(merged, status, {20, merged.rows - 20}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      {0, 255, 255}, 2);

    const double sx = static_cast<double>(max_display_width_) / static_cast<double>(merged.cols);
    const double sy = static_cast<double>(max_display_height_) / static_cast<double>(merged.rows);
    const double scale = std::min(1.0, std::min(sx, sy));
    display_scale_ = scale;
    if (scale >= 0.999) {
      return merged;
    }

    cv::Mat display;
    cv::resize(merged, display, cv::Size(), scale, scale, cv::INTER_AREA);
    return display;
  }

  void onTimer()
  {
    if (have_command_) {
      publishCommand();
    }
    cv::imshow(window_name_, panelImage());
    const int key = cv::waitKeyEx(1);
    handleKey(key);
  }

private:
  std::string cam0_image_topic_;
  std::string cam2_image_topic_;
  std::string cam0_info_topic_;
  std::string cmd_topic_;
  std::string enable_topic_;
  std::string feedback_topic_;
  std::string gimbal_angle_order_;
  std::string window_name_{"gimbal_camera_translation_debug"};

  int yaw_index_{0};
  int pitch_index_{1};
  int feedback_yaw_index_{0};
  int feedback_pitch_index_{1};
  int publish_dim_{2};
  double default_depth_m_{20.0};
  double manual_step_deg_{0.05};
  int max_display_width_{1800};
  int max_display_height_{950};
  double display_scale_{1.0};
  bool publish_enable_{true};
  bool enable_value_{true};

  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  bool have_cam0_info_{false};

  Eigen::Vector3d camera_to_gimbal_xyz_{-0.05, -0.20, 0.0};
  Eigen::Matrix3d R_gimbal_from_camera_{Eigen::Matrix3d::Identity()};
  double camera_to_gimbal_pitch_deg_{0.0};
  Eigen::Vector3d laser_offset_xyz_{0.0, 0.0, 0.0};
  gimbal_aim_solver::GimbalGeometry gimbal_geometry_;
  gimbal_aim_solver::AngleSolver angle_solver_;

  std::mutex image_mutex_;
  cv::Mat cam0_image_;
  cv::Mat cam2_image_;
  std::optional<cv::Point2d> selected_pixel_;
  bool zoom_cam0_{false};

  double command_yaw_deg_{0.0};
  double command_pitch_deg_{0.0};
  bool have_command_{false};
  double coarse_yaw_deg_{0.0};
  double coarse_pitch_deg_{0.0};
  double manual_delta_yaw_deg_{0.0};
  double manual_delta_pitch_deg_{0.0};
  bool have_coarse_command_{false};
  double latest_feedback_yaw_deg_{0.0};
  double latest_feedback_pitch_deg_{0.0};
  bool have_feedback_{false};
  std::vector<Sample> samples_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enable_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam0_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam2_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam0_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr feedback_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GimbalCameraTranslationDebugNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
