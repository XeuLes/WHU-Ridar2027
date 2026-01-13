#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "camera_field_calib_tool/yaml_io.hpp"
#include "camera_field_calib_tool/camera_model.hpp"
#include "camera_field_calib_tool/interactive_crosshair.hpp"
#include "camera_field_calib_tool/pnp_calibrator.hpp"
#include "camera_field_calib_tool/reprojection_metrics.hpp"

namespace camera_field_calib_tool
{

// -----------------------------
// Simple "dialog" for XYZ input via OpenCV window
// -----------------------------
class XYZInputDialog
{
public:
  XYZInputDialog() = default;

  void Open(const std::string & title, const cv::Point2d & uv_orig)
  {
    title_ = title;
    uv_orig_ = uv_orig;
    buffer_.clear();
    active_ = true;
    confirmed_ = false;
    cancelled_ = false;

    cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
  }

  void Close()
  {
    if (!title_.empty()) {
      cv::destroyWindow(title_);
    }
    active_ = false;
  }

  bool IsActive() const { return active_; }
  bool IsConfirmed() const { return confirmed_; }
  bool IsCancelled() const { return cancelled_; }

  cv::Point3d GetXYZ() const { return xyz_; }
  cv::Point2d GetUvOrig() const { return uv_orig_; }

  void HandleKey(int key)
  {
    if (!active_) return;

    // ESC cancels
    if (key == 27) {
      cancelled_ = true;
      active_ = false;
      return;
    }

    // Enter confirms
    if (key == 10 || key == 13) {
      cv::Point3d xyz;
      if (TryParseXYZ(buffer_, xyz)) {
        xyz_ = xyz;
        confirmed_ = true;
        active_ = false;
      }
      return;
    }

    // Backspace/Delete
    if (key == 8 || key == 127) {
      if (!buffer_.empty()) buffer_.pop_back();
      return;
    }

    // Acceptable characters: digits, space, comma, dot, minus, plus
    const char c = static_cast<char>(key & 0xFF);
    if ((c >= '0' && c <= '9') || c == ' ' || c == ',' || c == '.' || c == '-' || c == '+') {
      buffer_.push_back(c);
      return;
    }
  }

  void Render()
  {
    if (!active_) return;

    cv::Mat img(220, 640, CV_8UC3, cv::Scalar(20, 20, 20));

    cv::putText(img, "Input XYZ (world coordinates):", cv::Point(20, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::putText(img, "Format: X Y Z  (space or comma separated)", cv::Point(20, 75),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

    {
      std::ostringstream oss;
      oss.setf(std::ios::fixed);
      oss.precision(2);
      oss << "Selected pixel (orig): (" << uv_orig_.x << ", " << uv_orig_.y << ")";
      cv::putText(img, oss.str(), cv::Point(20, 105),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 0), 1);
    }

    std::string show = buffer_;
    show.push_back('|');

    cv::rectangle(img, cv::Rect(18, 125, 604, 40), cv::Scalar(60, 60, 60), 1);
    cv::putText(img, show, cv::Point(25, 153),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);

    cv::putText(img, "Enter: confirm   Esc: cancel   Backspace: delete", cv::Point(20, 195),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

    cv::Point3d tmp;
    if (!buffer_.empty() && TryParseXYZ(buffer_, tmp)) {
      cv::putText(img, "Parsed OK", cv::Point(520, 105),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);
    } else if (!buffer_.empty()) {
      cv::putText(img, "Parse FAIL", cv::Point(520, 105),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow(title_, img);
  }

private:
  static bool TryParseXYZ(const std::string & s, cv::Point3d & xyz)
  {
    std::string t = s;
    for (char & ch : t) {
      if (ch == ',') ch = ' ';
    }
    std::istringstream iss(t);
    double x, y, z;
    if (!(iss >> x >> y >> z)) return false;
    xyz = cv::Point3d(x, y, z);
    return true;
  }

private:
  std::string title_;
  bool active_{false};
  bool confirmed_{false};
  bool cancelled_{false};

  cv::Point2d uv_orig_{0.0, 0.0};
  std::string buffer_;
  cv::Point3d xyz_{0.0, 0.0, 0.0};
};

// -----------------------------
// Node
// -----------------------------
class PnPCalibNode : public rclcpp::Node
{
public:
  PnPCalibNode() : Node("pnp_calib_node")
  {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    intrinsics_yaml_ = this->declare_parameter<std::string>("intrinsics_yaml", "config/camera_intrinsics.yaml");
    out_matrix_yaml_ = this->declare_parameter<std::string>("out_matrix_yaml", "out_matrix.yaml");
    use_refine_ = this->declare_parameter<bool>("use_refine", true);
    pnp_method_ = this->declare_parameter<int>("pnp_method", 0);  // 0 -> default EPnP
    min_points_ = this->declare_parameter<int>("min_points", 6);

    CameraIntrinsics intr;
    std::string err;
    if (!LoadCameraIntrinsicsYaml(intrinsics_yaml_, intr, &err)) {
      RCLCPP_ERROR(this->get_logger(), "Load intrinsics failed: %s", err.c_str());
      throw std::runtime_error("Failed to load intrinsics yaml");
    }

    cam_.SetIntrinsics(intr.K, intr.D, intr.width, intr.height);
    if (!cam_.IsValid()) {
      throw std::runtime_error("Invalid camera intrinsics");
    }

    calibrator_.SetCameraModel(cam_);

    window_name_ = "PnP Calib";
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, 1280, 720);

    // Mouse click to move crosshair
    cv::setMouseCallback(window_name_.c_str(), &PnPCalibNode::MouseCallback, this);

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PnPCalibNode::OnImage, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "pnp_calib_node started.");
    RCLCPP_INFO(this->get_logger(), "Image topic: %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Intrinsics: %s", intrinsics_yaml_.c_str());
    RCLCPP_INFO(this->get_logger(), "Output: %s", out_matrix_yaml_.c_str());

    PrintConsoleHelp();
  }

  ~PnPCalibNode() override
  {
    cv::destroyAllWindows();
  }

private:
  void PrintConsoleHelp()
  {
    std::cout << "\n=== PnP Calib Controls ===\n"
              << InteractiveCrosshair::HelpString() << "\n"
              << "Mouse Left Click: move crosshair to clicked pixel (display)\n"
              << "Enter: confirm 2D at crosshair -> pop XYZ dialog\n"
              << "Backspace/Delete: undo last point\n"
              << "P: solvePnP\n"
              << "S: save out_matrix.yaml (must solve first)\n"
              << "C: clear all points\n"
              << "Q/Esc: quit node\n"
              << "Tip: resize the window; the image will auto-fit.\n"
              << "=========================\n"
              << std::endl;
  }

  // Mouse callback trampoline
  static void MouseCallback(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    auto * self = reinterpret_cast<PnPCalibNode *>(userdata);
    if (self) self->OnMouse(event, x, y, flags);
  }

  void OnMouse(int event, int x, int y, int flags)
  {
    (void)flags;
    // Do not accept clicks while XYZ dialog is active
    if (xyz_dialog_.IsActive()) return;
    // Only act when UI initialized (we know image size)
    if (!ui_inited_) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
      // x,y are in DISPLAY coordinates (what user sees)
      crosshair_.SetPosition(cv::Point2d(static_cast<double>(x), static_cast<double>(y)));
    }
  }

  double ComputeAutoFitScale(int img_w, int img_h) const
  {
    int target_w = 1280;
    int target_h = 720;

#if CV_VERSION_MAJOR >= 4
    try {
      cv::Rect r = cv::getWindowImageRect(window_name_);
      if (r.width > 50 && r.height > 50) {
        target_w = r.width;
        target_h = r.height;
      }
    } catch (...) {
      // fallback
    }
#endif

    const double sx = static_cast<double>(target_w) / static_cast<double>(img_w);
    const double sy = static_cast<double>(target_h) / static_cast<double>(img_h);
    const double s = std::min(1.0, std::min(sx, sy));
    return (s > 0.0) ? s : 1.0;
  }

  void RescaleCrosshairIfNeeded(double new_scale, int disp_w, int disp_h)
  {
    if (!ui_inited_) {
      crosshair_.SetImageSize(disp_w, disp_h);
      ui_inited_ = true;
      last_scale_ = new_scale;
      return;
    }

    if (std::abs(new_scale - last_scale_) > 1e-6) {
      const cv::Point2d old_pos = crosshair_.GetPosition();
      const double k = (last_scale_ > 0.0) ? (new_scale / last_scale_) : 1.0;
      crosshair_.SetImageSize(disp_w, disp_h);
      crosshair_.SetPosition(cv::Point2d(old_pos.x * k, old_pos.y * k));
      last_scale_ = new_scale;
    } else {
      crosshair_.SetImageSize(disp_w, disp_h);
    }
  }

  void OnImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    const cv::Mat & img = cv_ptr->image;
    if (img.empty()) return;

    const int W = img.cols;
    const int H = img.rows;

    const double scale = ComputeAutoFitScale(W, H);
    last_frame_scale_ = scale;

    cv::Mat vis;
    if (scale < 1.0) {
      cv::resize(img, vis, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
      vis = img.clone();
    }

    RescaleCrosshairIfNeeded(scale, vis.cols, vis.rows);

    // draw existing points (stored in original pixel coords)
    const auto & img_pts = calibrator_.ImagePoints();
    for (size_t i = 0; i < img_pts.size(); ++i) {
      const double x_disp = img_pts[i].x * scale;
      const double y_disp = img_pts[i].y * scale;

      cv::circle(vis, cv::Point(static_cast<int>(x_disp), static_cast<int>(y_disp)),
                 3, cv::Scalar(255, 0, 0), -1);
      cv::putText(vis, std::to_string(i),
                  cv::Point(static_cast<int>(x_disp) + 5, static_cast<int>(y_disp) - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }

    if (has_last_result_) {
      std::string s = ReprojectionMetrics::ToString(last_result_.reproj);
      cv::putText(vis, s, cv::Point(10, 90),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    crosshair_.Draw(vis);
    crosshair_.DrawMagnifier(vis, vis);

    cv::imshow(window_name_, vis);

    if (xyz_dialog_.IsActive()) {
      xyz_dialog_.Render();
    }

    int key = cv::waitKey(1);
    if (key < 0) return;

    // If dialog active, route keys to dialog first
    if (xyz_dialog_.IsActive()) {
      xyz_dialog_.HandleKey(key);

      if (!xyz_dialog_.IsActive()) {
        if (xyz_dialog_.IsConfirmed()) {
          const cv::Point2d uv_orig = xyz_dialog_.GetUvOrig();
          const cv::Point3d xyz = xyz_dialog_.GetXYZ();
          calibrator_.AddPoint(xyz, uv_orig);

          std::cout << "[PnP] Added: 2D_orig=(" << uv_orig.x << "," << uv_orig.y << ") "
                    << "3D=(" << xyz.x << "," << xyz.y << "," << xyz.z << ")\n";
        } else if (xyz_dialog_.IsCancelled()) {
          std::cout << "[PnP] XYZ input cancelled.\n";
        }
        xyz_dialog_.Close();
      }
      return;
    }

    // Quit
    if (key == 27 || key == 'q' || key == 'Q') {
      rclcpp::shutdown();
      return;
    }

    // Crosshair keys
    crosshair_.HandleKey(key);

    // Undo
    if (key == 8 || key == 127) {
      calibrator_.Undo();
      return;
    }

    // Clear
    if (key == 'C') {
      calibrator_.Reset();
      has_last_result_ = false;
      return;
    }

    // Confirm 2D -> open dialog to input XYZ
    if (key == 10 || key == 13) {
      const cv::Point2d uv_disp = crosshair_.GetPosition();
      const double scale_now = (last_frame_scale_ > 0.0) ? last_frame_scale_ : 1.0;
      const cv::Point2d uv_orig(uv_disp.x / scale_now, uv_disp.y / scale_now);

      xyz_dialog_.Open("Input XYZ (press Enter to confirm)", uv_orig);
      return;
    }

    // Solve
    if (key == 'P') {
      if (static_cast<int>(calibrator_.Size()) < min_points_) {
        std::cout << "[PnP] Need at least " << min_points_
                  << " points. Current: " << calibrator_.Size() << "\n";
        return;
      }

      last_result_ = calibrator_.Solve(pnp_method_, use_refine_);
      has_last_result_ = last_result_.success;

      if (!last_result_.success) {
        std::cout << "[PnP] solvePnP failed.\n";
      } else {
        std::cout << "[PnP] solvePnP OK. " << ReprojectionMetrics::ToString(last_result_.reproj) << "\n";
        std::cout << "      rvec = [" << last_result_.rvec[0] << ", " << last_result_.rvec[1] << ", " << last_result_.rvec[2] << "]\n";
        std::cout << "      tvec = [" << last_result_.tvec[0] << ", " << last_result_.tvec[1] << ", " << last_result_.tvec[2] << "]\n";
      }
      return;
    }

    // Save
    if (key == 'S') {
      if (!has_last_result_ || !last_result_.success) {
        std::cout << "[PnP] No valid result to save. Press P to solve first.\n";
        return;
      }

      CameraExtrinsics ext;
      ext.rvec = last_result_.rvec;
      ext.tvec = last_result_.tvec;
      ext.num_points = static_cast<int>(calibrator_.Size());
      ext.reproj_rmse = last_result_.reproj.rmse;
      ext.reproj_max = last_result_.reproj.max_error;
      ext.image_topic = image_topic_;

      std::string err;
      if (!SaveExtrinsicsYaml(out_matrix_yaml_, ext, &err)) {
        std::cout << "[PnP] Save failed: " << err << "\n";
      } else {
        std::cout << "[PnP] Saved: " << out_matrix_yaml_ << "\n";
      }
      return;
    }
  }

private:
  std::string image_topic_;
  std::string intrinsics_yaml_;
  std::string out_matrix_yaml_;
  bool use_refine_{true};
  int pnp_method_{0};
  int min_points_{6};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  CameraModel cam_;
  PnPCalibrator calibrator_;

  std::string window_name_;
  bool ui_inited_{false};
  InteractiveCrosshair crosshair_;
  double last_scale_{1.0};

  // last frame scale (used to convert display->orig for click/enter)
  double last_frame_scale_{1.0};

  bool has_last_result_{false};
  PnPResult last_result_;

  XYZInputDialog xyz_dialog_;
};

}  // namespace camera_field_calib_tool

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<camera_field_calib_tool::PnPCalibNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
