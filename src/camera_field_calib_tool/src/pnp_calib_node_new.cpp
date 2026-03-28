#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "camera_field_calib_tool/yaml_io.hpp"
#include "camera_field_calib_tool/camera_model.hpp"
#include "camera_field_calib_tool/interactive_crosshair.hpp"
#include "camera_field_calib_tool/pnp_calibrator.hpp"
#include "camera_field_calib_tool/reprojection_metrics.hpp"

namespace camera_field_calib_tool
{

class PnPCalibNodeNew : public rclcpp::Node
{
public:
  PnPCalibNodeNew() : Node("pnp_calib_node_new")
  {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    intrinsics_yaml_ = this->declare_parameter<std::string>("intrinsics_yaml", "config/camera_intrinsics.yaml");
    out_matrix_yaml_ = this->declare_parameter<std::string>("out_matrix_yaml", "out_matrix_new.yaml");
    reference_points_yaml_ =
      this->declare_parameter<std::string>("reference_points_yaml", "config/reference_points.yaml");

    use_refine_ = this->declare_parameter<bool>("use_refine", true);
    pnp_method_ = this->declare_parameter<int>("pnp_method", 0);   // 0 -> default EPnP
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

    if (!LoadReferencePoints(reference_points_yaml_, &err)) {
      RCLCPP_ERROR(this->get_logger(), "Load reference points failed: %s", err.c_str());
      throw std::runtime_error("Failed to load reference points yaml");
    }

    window_name_ = "PnP Calib New";
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, 1280, 720);

    cv::setMouseCallback(window_name_.c_str(), &PnPCalibNodeNew::MouseCallback, this);

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PnPCalibNodeNew::OnImage, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "pnp_calib_node_new started.");
    RCLCPP_INFO(this->get_logger(), "Image topic: %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Intrinsics: %s", intrinsics_yaml_.c_str());
    RCLCPP_INFO(this->get_logger(), "Reference points: %s", reference_points_yaml_.c_str());
    RCLCPP_INFO(this->get_logger(), "Output: %s", out_matrix_yaml_.c_str());
    RCLCPP_INFO(this->get_logger(), "Loaded %zu reference points.", ref_world_points_.size());

    PrintConsoleHelp();
  }

  ~PnPCalibNodeNew() override
  {
    cv::destroyAllWindows();
  }

private:
  struct ReferencePoint
  {
    std::string name;
    cv::Point3d xyz;
  };

  void PrintConsoleHelp()
  {
    std::cout << "\n=== PnP Calib New Controls ===\n"
              << InteractiveCrosshair::HelpString() << "\n"
              << "Mouse Left Click: move crosshair to clicked pixel (display)\n"
              << "Enter: confirm 2D at crosshair -> auto bind next reference 3D point\n"
              << "Backspace/Delete: undo last point\n"
              << "P: solvePnP\n"
              << "S: save out_matrix.yaml (must solve first)\n"
              << "C: clear all points\n"
              << "Q/Esc: quit node\n"
              << "Tip: resize the window; the image will auto-fit.\n"
              << "=============================\n"
              << std::endl;

    if (!reference_points_.empty()) {
      std::cout << "Reference point order:\n";
      for (size_t i = 0; i < reference_points_.size(); ++i) {
        const auto & p = reference_points_[i];
        std::cout << "  [" << i << "] " << p.name
                  << " -> (" << p.xyz.x << ", " << p.xyz.y << ", " << p.xyz.z << ")\n";
      }
      std::cout << std::endl;
    }
  }

  bool LoadReferencePoints(const std::string & path, std::string * err)
  {
    try {
      cv::FileStorage fs(path, cv::FileStorage::READ);
      if (!fs.isOpened()) {
        if (err) {
          *err = "cannot open file: " + path;
        }
        return false;
      }

      cv::FileNode node = fs["reference_points"];
      if (node.empty() || !node.isSeq()) {
        if (err) {
          *err = "key 'reference_points' missing or not a sequence";
        }
        return false;
      }

      reference_points_.clear();
      ref_world_points_.clear();
      ref_point_names_.clear();

      size_t idx = 0;
      for (auto it = node.begin(); it != node.end(); ++it, ++idx) {
        cv::FileNode item = *it;
        if (!item.isMap()) {
          if (err) {
            *err = "reference_points[" + std::to_string(idx) + "] is not a map";
          }
          return false;
        }

        cv::FileNode name_node = item["name"];
        cv::FileNode pos_node = item["pos"];
        if (name_node.empty() || pos_node.empty()) {
          if (err) {
            *err = "reference_points[" + std::to_string(idx) + "] missing 'name' or 'pos'";
          }
          return false;
        }

        std::string name;
        name_node >> name;
        if (name.empty()) {
          if (err) {
            *err = "reference_points[" + std::to_string(idx) + "] has empty name";
          }
          return false;
        }

        std::vector<double> pos;
        pos_node >> pos;
        if (pos.size() != 3) {
          if (err) {
            *err = "reference_points[" + std::to_string(idx) + "].pos must have 3 values";
          }
          return false;
        }

        ReferencePoint rp;
        rp.name = name;
        rp.xyz = cv::Point3d(pos[0], pos[1], pos[2]);

        reference_points_.push_back(rp);
        ref_world_points_.push_back(rp.xyz);
        ref_point_names_.push_back(rp.name);
      }

      if (reference_points_.empty()) {
        if (err) {
          *err = "no reference points loaded";
        }
        return false;
      }

      return true;
    } catch (const std::exception & e) {
      if (err) {
        *err = e.what();
      }
      return false;
    } catch (...) {
      if (err) {
        *err = "unknown exception while loading reference points";
      }
      return false;
    }
  }

  static void MouseCallback(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    auto * self = reinterpret_cast<PnPCalibNodeNew *>(userdata);
    if (self) self->OnMouse(event, x, y, flags);
  }

  void OnMouse(int event, int x, int y, int flags)
  {
    (void)flags;
    if (!ui_inited_) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
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

  void DrawGuidance(cv::Mat & vis, double /*scale*/)
  {
    if (reference_points_.empty()) return;

    const size_t curr_idx = calibrator_.Size();

    auto shorten = [](const std::string & s, size_t max_len) -> std::string {
      if (s.size() <= max_len) return s;
      return s.substr(0, max_len - 3) + "...";
    };

    std::vector<std::string> lines;

    if (curr_idx < reference_points_.size()) {
      const auto & rp = reference_points_[curr_idx];

      {
        std::ostringstream oss;
        oss << "AUTO STEP " << (curr_idx + 1) << "/" << reference_points_.size();
        lines.push_back(oss.str());
      }

      lines.push_back("NEXT: " + shorten(rp.name, 42));

      {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "WORLD: (" << rp.xyz.x << ", " << rp.xyz.y << ", " << rp.xyz.z << ")";
        lines.push_back(oss.str());
      }
    } else {
      std::ostringstream oss;
      oss << "AUTO STEP DONE " << reference_points_.size() << "/" << reference_points_.size();
      lines.push_back(oss.str());
      lines.push_back("Press P to solve, S to save");
    }

    int baseline = 0;
    int max_text_w = 0;
    for (const auto & line : lines) {
      const cv::Size ts = cv::getTextSize(
        line, cv::FONT_HERSHEY_SIMPLEX, 0.62, 2, &baseline);
      max_text_w = std::max(max_text_w, ts.width);
    }

    const int pad_x = 12;
    const int pad_y = 10;
    const int line_h = 28;

    int box_w = max_text_w + pad_x * 2;
    int box_h = static_cast<int>(lines.size()) * line_h + pad_y * 2;

    int x0 = 16;
    int y0 = vis.rows - box_h - 16;

    if (x0 < 10) x0 = 10;
    if (y0 < 10) y0 = 10;
    if (x0 + box_w > vis.cols) box_w = vis.cols - x0 - 4;
    if (y0 + box_h > vis.rows) box_h = vis.rows - y0 - 4;

    if (box_w <= 0 || box_h <= 0) return;

    cv::Rect box(x0, y0, box_w, box_h);

    // 半透明深色背景
    cv::Mat roi = vis(box);
    cv::Mat overlay = roi.clone();
    overlay.setTo(cv::Scalar(24, 24, 24));
    cv::addWeighted(overlay, 0.65, roi, 0.35, 0.0, roi);

    // 边框
    cv::rectangle(vis, box, cv::Scalar(0, 255, 0), 1);

    // 文字
    int y = y0 + pad_y + 18;
    for (size_t i = 0; i < lines.size(); ++i) {
      const cv::Scalar color = (i == 0) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
      cv::putText(
        vis, lines[i], cv::Point(x0 + pad_x, y),
        cv::FONT_HERSHEY_SIMPLEX, 0.62, color, 2);
      y += line_h;
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

    DrawGuidance(vis, scale);

    crosshair_.Draw(vis);
    crosshair_.DrawMagnifier(vis, vis);

    cv::imshow(window_name_, vis);

    int key = cv::waitKey(1);
    if (key < 0) return;

    if (key == 27 || key == 'q' || key == 'Q') {
      rclcpp::shutdown();
      return;
    }

    crosshair_.HandleKey(key);

    if (key == 8 || key == 127) {
      calibrator_.Undo();
      has_last_result_ = false;
      std::cout << "[PnP-New] Undo last point. Current: " << calibrator_.Size() << "\n";
      return;
    }

    if (key == 'c' || key == 'C') {
      calibrator_.Reset();
      has_last_result_ = false;
      std::cout << "[PnP-New] Cleared all points.\n";
      return;
    }

    if (key == 10 || key == 13) {
      const size_t curr_idx = calibrator_.Size();
      if (curr_idx >= reference_points_.size()) {
        std::cout << "[PnP-New] No more reference points left to assign.\n";
        return;
      }

      const cv::Point2d uv_disp = crosshair_.GetPosition();
      const double scale_now = (last_frame_scale_ > 0.0) ? last_frame_scale_ : 1.0;
      const cv::Point2d uv_orig(uv_disp.x / scale_now, uv_disp.y / scale_now);

      const auto & rp = reference_points_[curr_idx];
      calibrator_.AddPoint(rp.xyz, uv_orig);
      has_last_result_ = false;

      std::cout << "[PnP-New] Added: idx=" << curr_idx
                << " name=" << rp.name
                << " 2D_orig=(" << uv_orig.x << "," << uv_orig.y << ")"
                << " 3D=(" << rp.xyz.x << "," << rp.xyz.y << "," << rp.xyz.z << ")\n";
      return;
    }

    if (key == 'p' || key == 'P') {
      if (static_cast<int>(calibrator_.Size()) < min_points_) {
        std::cout << "[PnP-New] Need at least " << min_points_
                  << " points. Current: " << calibrator_.Size() << "\n";
        return;
      }

      last_result_ = calibrator_.Solve(pnp_method_, use_refine_);
      has_last_result_ = last_result_.success;

      if (!last_result_.success) {
        std::cout << "[PnP-New] solvePnP failed.\n";
      } else {
        std::cout << "[PnP-New] solvePnP OK. "
                  << ReprojectionMetrics::ToString(last_result_.reproj) << "\n";
        std::cout << "          rvec = [" << last_result_.rvec[0] << ", "
                  << last_result_.rvec[1] << ", " << last_result_.rvec[2] << "]\n";
        std::cout << "          tvec = [" << last_result_.tvec[0] << ", "
                  << last_result_.tvec[1] << ", " << last_result_.tvec[2] << "]\n";
      }
      return;
    }

    if (key == 's' || key == 'S') {
      if (!has_last_result_ || !last_result_.success) {
        std::cout << "[PnP-New] No valid result to save. Press P to solve first.\n";
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
        std::cout << "[PnP-New] Save failed: " << err << "\n";
      } else {
        std::cout << "[PnP-New] Saved: " << out_matrix_yaml_ << "\n";
      }
      return;
    }
  }

private:
  std::string image_topic_;
  std::string intrinsics_yaml_;
  std::string out_matrix_yaml_;
  std::string reference_points_yaml_;

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
  double last_frame_scale_{1.0};

  bool has_last_result_{false};
  PnPResult last_result_;

  std::vector<ReferencePoint> reference_points_;
  std::vector<cv::Point3d> ref_world_points_;
  std::vector<std::string> ref_point_names_;
};

}  // namespace camera_field_calib_tool

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<camera_field_calib_tool::PnPCalibNodeNew>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}