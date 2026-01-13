#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "camera_field_calib_tool/yaml_io.hpp"
#include "camera_field_calib_tool/camera_model.hpp"
#include "camera_field_calib_tool/ray_plane_intersector.hpp"

namespace camera_field_calib_tool
{

// -----------------------------
// Dialog: input a double (plane Z)
// -----------------------------
class ZInputDialog
{
public:
  void Open(const std::string & title, double current_z)
  {
    title_ = title;
    buffer_.clear();
    active_ = true;
    confirmed_ = false;
    cancelled_ = false;
    z_ = current_z;
    cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
  }

  void Close()
  {
    if (!title_.empty()) cv::destroyWindow(title_);
    active_ = false;
  }

  bool IsActive() const { return active_; }
  bool IsConfirmed() const { return confirmed_; }
  bool IsCancelled() const { return cancelled_; }
  double GetZ() const { return z_; }

  void HandleKey(int key)
  {
    if (!active_) return;

    if (key == 27) { cancelled_ = true; active_ = false; return; }              // Esc
    if (key == 10 || key == 13) {                                               // Enter
      double v;
      if (TryParseDouble(buffer_, v)) { z_ = v; confirmed_ = true; active_ = false; }
      return;
    }
    if (key == 8 || key == 127) { if (!buffer_.empty()) buffer_.pop_back(); return; }  // Backspace/Delete

    const char c = static_cast<char>(key & 0xFF);
    if ((c >= '0' && c <= '9') || c == ' ' || c == ',' || c == '.' || c == '-' || c == '+') {
      buffer_.push_back(c);
    }
  }

  void Render()
  {
    if (!active_) return;

    cv::Mat img(200, 560, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::putText(img, "Input plane height Z:", cv::Point(20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "Current Z: " << z_;
    cv::putText(img, oss.str(), cv::Point(20, 85),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 0), 1);

    std::string show = buffer_;
    show.push_back('|');

    cv::rectangle(img, cv::Rect(18, 105, 524, 40), cv::Scalar(60, 60, 60), 1);
    cv::putText(img, show, cv::Point(25, 135),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::putText(img, "Enter: confirm   Esc: cancel   Backspace: delete", cv::Point(20, 175),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

    cv::imshow(title_, img);
  }

private:
  static bool TryParseDouble(const std::string & s, double & v)
  {
    std::string t = s;
    for (char & ch : t) if (ch == ',') ch = ' ';
    std::istringstream iss(t);
    return static_cast<bool>(iss >> v);
  }

private:
  std::string title_;
  bool active_{false};
  bool confirmed_{false};
  bool cancelled_{false};
  std::string buffer_;
  double z_{0.0};
};

// -----------------------------
// Dialog: rename selected region (string)  [triggered by N]
// -----------------------------
class NameInputDialog
{
public:
  void Open(const std::string & title, const std::string & current_name)
  {
    title_ = title;
    buffer_ = current_name;
    active_ = true;
    confirmed_ = false;
    cancelled_ = false;
    cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
  }

  void Close()
  {
    if (!title_.empty()) cv::destroyWindow(title_);
    active_ = false;
  }

  bool IsActive() const { return active_; }
  bool IsConfirmed() const { return confirmed_; }
  bool IsCancelled() const { return cancelled_; }
  std::string GetName() const { return buffer_; }

  void HandleKey(int key)
  {
    if (!active_) return;

    if (key == 27) { cancelled_ = true; active_ = false; return; }   // Esc
    if (key == 10 || key == 13) {                                     // Enter
      if (!buffer_.empty()) { confirmed_ = true; active_ = false; }
      return;
    }
    if (key == 8 || key == 127) {                                     // Backspace/Delete
      if (!buffer_.empty()) buffer_.pop_back();
      return;
    }

    const char c = static_cast<char>(key & 0xFF);
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-') {
      buffer_.push_back(c);
    }
  }

  void Render()
  {
    if (!active_) return;

    cv::Mat img(200, 640, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::putText(img, "Rename selected region:", cv::Point(20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    cv::putText(img, "Allowed: letters/digits/_/-", cv::Point(20, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

    std::string show = buffer_;
    show.push_back('|');

    cv::rectangle(img, cv::Rect(18, 105, 604, 40), cv::Scalar(60, 60, 60), 1);
    cv::putText(img, show, cv::Point(25, 135),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);

    cv::putText(img, "Enter: confirm   Esc: cancel   Backspace: delete", cv::Point(20, 175),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

    cv::imshow(title_, img);
  }

private:
  std::string title_;
  bool active_{false};
  bool confirmed_{false};
  bool cancelled_{false};
  std::string buffer_;
};

// -----------------------------
// YAML utilities: arbitrary region names
// -----------------------------
static bool LoadYamlIfExists(const std::string & path, YAML::Node & root, std::string & err)
{
  try {
    std::ifstream f(path);
    if (!f.good()) {
      root = YAML::Node(YAML::NodeType::Map);
      return true;
    }
    root = YAML::LoadFile(path);
    if (!root || !root.IsMap()) root = YAML::Node(YAML::NodeType::Map);
    return true;
  } catch (const std::exception & e) {
    err = std::string("Load YAML failed: ") + e.what();
    return false;
  }
}

static YAML::Node PointListToYaml(const std::vector<cv::Point3d> & pts)
{
  YAML::Node arr(YAML::NodeType::Sequence);
  for (const auto & p : pts) {
    YAML::Node m(YAML::NodeType::Map);
    m["x"] = p.x;
    m["y"] = p.y;
    m["z"] = p.z;
    arr.push_back(m);
  }
  return arr;
}

static bool SaveYamlWithHeader(const std::string & path, const YAML::Node & root, std::string & err)
{
  try {
    YAML::Emitter out;
    out.SetIndent(2);
    out << root;

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      err = "Open for write failed: " + path;
      return false;
    }
    ofs << "%YAML:1.0\n---\n";
    ofs << out.c_str() << "\n";
    return true;
  } catch (const std::exception & e) {
    err = std::string("Save YAML failed: ") + e.what();
    return false;
  }
}

static bool ParseWorldPointList(const YAML::Node & node, std::vector<cv::Point3d> & pts)
{
  if (!node || !node.IsSequence()) return false;
  pts.clear();
  pts.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const auto & p = node[i];
    if (!p || !p.IsMap() || !p["x"] || !p["y"] || !p["z"]) return false;
    cv::Point3d pt;
    pt.x = p["x"].as<double>();
    pt.y = p["y"].as<double>();
    pt.z = p["z"].as<double>();
    pts.push_back(pt);
  }
  return true;
}

// -----------------------------
// Node
// -----------------------------
class PlanePolygonNode : public rclcpp::Node
{
public:
  PlanePolygonNode() : Node("plane_polygon_node")
  {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    intrinsics_yaml_ = this->declare_parameter<std::string>("intrinsics_yaml", "config/camera_intrinsics.yaml");
    out_matrix_yaml_ = this->declare_parameter<std::string>("out_matrix_yaml", "out_matrix.yaml");
    field_points_yaml_ = this->declare_parameter<std::string>("field_points_yaml", "Field_Points.yaml");
    default_plane_z_ = this->declare_parameter<double>("default_plane_z", 0.0);

    // Load intrinsics
    CameraIntrinsics intr;
    std::string err;
    if (!LoadCameraIntrinsicsYaml(intrinsics_yaml_, intr, &err)) {
      RCLCPP_ERROR(this->get_logger(), "Load intrinsics failed: %s", err.c_str());
      throw std::runtime_error("Failed to load intrinsics yaml");
    }
    cam_.SetIntrinsics(intr.K, intr.D, intr.width, intr.height);
    if (!cam_.IsValid()) throw std::runtime_error("Invalid camera intrinsics");

    // Load extrinsics once
    if (!LoadExtrinsicsYaml(out_matrix_yaml_, ext_, &err)) {
      RCLCPP_ERROR(this->get_logger(), "Load extrinsics failed: %s", err.c_str());
      throw std::runtime_error("Failed to load out_matrix yaml");
    }
    rvec_ = ext_.rvec;
    tvec_ = ext_.tvec;
    intersector_.SetExtrinsics(rvec_, tvec_);

    // Editing defaults
    region_name_ = "Middle_Line";
    selected_region_name_.clear();
    plane_z_ = default_plane_z_;

    ReloadRegionsFromYaml();

    // UI
    window_name_ = "Plane Polygon";
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, 1280, 720);
    cv::setMouseCallback(window_name_.c_str(), &PlanePolygonNode::MouseCallback, this);

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PlanePolygonNode::OnImage, this, std::placeholders::_1));

    PrintHelp();
  }

  ~PlanePolygonNode() override { cv::destroyAllWindows(); }

private:
  void PrintHelp()
  {
    std::cout << "\n=== Plane Polygon Controls ===\n"
              << "Left Click        : add vertex (editing polygon)\n"
              << "Right Click       : select existing region ONLY (no undo)\n"
              << "Enter             : close editing polygon\n"
              << "z / Z             : undo last vertex\n"
              << "H                 : set plane height Z (dialog)\n"
              << "S                 : save editing region to YAML\n"
              << "N                 : rename SELECTED region (dialog)\n"
              << "R                 : reset editing polygon\n"
              << "Q/Esc             : quit\n"
              << "Tip: resize window; image auto-fit.\n"
              << "=============================\n"
              << std::endl;
  }

  bool DecodeToBGR(const sensor_msgs::msg::Image::ConstSharedPtr & msg, cv::Mat & bgr)
  {
    try {
      auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      bgr = cv_ptr->image;
      return !bgr.empty();
    } catch (...) {
      try {
        auto cv_ptr = cv_bridge::toCvShare(msg, "passthrough");
        if (cv_ptr->image.empty()) return false;

        const std::string enc = msg->encoding;
        if (enc.find("bayer") != std::string::npos || enc.find("Bayer") != std::string::npos) {
          cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_BayerRG2BGR);
          return !bgr.empty();
        }
        if (enc == "mono8" || enc == "8UC1") {
          cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_GRAY2BGR);
          return !bgr.empty();
        }
        bgr = cv_ptr->image;
        return !bgr.empty();
      } catch (...) {
        return false;
      }
    }
  }

  double ComputeAutoFitScale(int img_w, int img_h) const
  {
    int target_w = 1280, target_h = 720;
#if CV_VERSION_MAJOR >= 4
    try {
      cv::Rect r = cv::getWindowImageRect(window_name_);
      if (r.width > 50 && r.height > 50) { target_w = r.width; target_h = r.height; }
    } catch (...) {}
#endif
    const double sx = (double)target_w / (double)img_w;
    const double sy = (double)target_h / (double)img_h;
    return std::min(1.0, std::min(sx, sy));
  }

  void ReloadRegionsFromYaml()
  {
    regions_world_.clear();
    YAML::Node root;
    std::string err;
    if (!LoadYamlIfExists(field_points_yaml_, root, err)) {
      std::cout << "[Plane] " << err << "\n";
      return;
    }
    if (!root || !root.IsMap()) return;

    for (auto it = root.begin(); it != root.end(); ++it) {
      const std::string key = it->first.as<std::string>();
      std::vector<cv::Point3d> pts;
      if (ParseWorldPointList(it->second, pts) && pts.size() >= 3) {
        regions_world_[key] = std::move(pts);
      }
    }
  }

  bool ProjectWorldToImagePolygon(const std::vector<cv::Point3d> & world_pts,
                                  std::vector<cv::Point2f> & img_pts) const
  {
    if (world_pts.size() < 3) return false;
    std::vector<cv::Point3f> w3f;
    w3f.reserve(world_pts.size());
    for (const auto & p : world_pts) w3f.emplace_back((float)p.x, (float)p.y, (float)p.z);

    img_pts.clear();
    cv::projectPoints(w3f, rvec_, tvec_, cv::Mat(cam_.K()), cv::Mat(cam_.D()), img_pts);
    return img_pts.size() == world_pts.size();
  }

  bool SelectRegionAtPixel(const cv::Point2d & uv_orig)
  {
    for (const auto & kv : regions_world_) {
      const std::string & name = kv.first;
      std::vector<cv::Point2f> poly_uv;
      if (!ProjectWorldToImagePolygon(kv.second, poly_uv)) continue;

      std::vector<cv::Point> poly_i;
      poly_i.reserve(poly_uv.size());
      for (const auto & p : poly_uv) poly_i.emplace_back((int)std::lround(p.x), (int)std::lround(p.y));

      const double v = cv::pointPolygonTest(
        poly_i,
        cv::Point((int)std::lround(uv_orig.x), (int)std::lround(uv_orig.y)),
        false);

      if (v > 0) {
        selected_region_name_ = name;
        std::cout << "[Plane] Selected region: " << selected_region_name_ << "\n";
        return true;
      }
    }
    return false;
  }

  static void MouseCallback(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    auto * self = reinterpret_cast<PlanePolygonNode *>(userdata);
    if (self) self->OnMouse(event, x, y);
  }

  void OnMouse(int event, int x, int y)
  {
    if (!ui_inited_) return;
    if (z_dialog_.IsActive() || name_dialog_.IsActive()) return;

    const double s = (last_frame_scale_ > 0.0) ? last_frame_scale_ : 1.0;
    const cv::Point2d uv_orig((double)x / s, (double)y / s);

    if (event == cv::EVENT_LBUTTONDOWN) {
      if (closed_) return;
      pix_pts_orig_.push_back(uv_orig);
      return;
    }

    if (event == cv::EVENT_RBUTTONDOWN) {
      // right click only selects
      if (!SelectRegionAtPixel(uv_orig)) {
        std::cout << "[Plane] No region hit under right-click.\n";
      }
      return;
    }
  }

  void UndoLast()
  {
    if (!pix_pts_orig_.empty()) {
      pix_pts_orig_.pop_back();
      if (closed_ && pix_pts_orig_.size() < 3) closed_ = false;
    }
  }

  void ResetCurrent()
  {
    pix_pts_orig_.clear();
    closed_ = false;
  }

  bool ClosePolygon()
  {
    if (closed_) return true;
    if (pix_pts_orig_.size() < 3) return false;
    closed_ = true;
    return true;
  }

  bool SaveCurrentRegion()
  {
    if (!closed_) { std::cout << "[Plane] Polygon not closed. Press Enter first.\n"; return false; }
    if (pix_pts_orig_.size() < 3) { std::cout << "[Plane] Not enough points.\n"; return false; }
    if (region_name_.empty()) { std::cout << "[Plane] Editing region name empty.\n"; return false; }

    Plane plane = RayPlaneIntersector::MakeZPlane(plane_z_);
    std::vector<cv::Point3d> world_pts;
    world_pts.reserve(pix_pts_orig_.size());

    for (const auto & uv : pix_pts_orig_) {
      cv::Point3d Xw;
      if (!intersector_.PixelIntersectPlane(cam_, uv, plane, Xw)) {
        std::cout << "[Plane] Intersection failed at pixel (" << uv.x << "," << uv.y << ").\n";
        return false;
      }
      Xw.z = plane_z_;
      world_pts.push_back(Xw);
    }

    YAML::Node root;
    std::string err;
    if (!LoadYamlIfExists(field_points_yaml_, root, err)) { std::cout << "[Plane] " << err << "\n"; return false; }
    root[region_name_] = PointListToYaml(world_pts);

    if (!SaveYamlWithHeader(field_points_yaml_, root, err)) { std::cout << "[Plane] " << err << "\n"; return false; }

    std::cout << "[Plane] Saved '" << region_name_ << "' (" << world_pts.size() << " pts) Z=" << plane_z_
              << " to " << field_points_yaml_ << "\n";

    ReloadRegionsFromYaml();
    return true;
  }

  bool RenameSelectedRegion(const std::string & new_name)
  {
    if (selected_region_name_.empty()) { std::cout << "[Plane] No selected region.\n"; return false; }
    if (new_name.empty()) return false;
    if (new_name == selected_region_name_) return true;

    YAML::Node root;
    std::string err;
    if (!LoadYamlIfExists(field_points_yaml_, root, err)) { std::cout << "[Plane] " << err << "\n"; return false; }
    if (!root[selected_region_name_]) { std::cout << "[Plane] Selected key not found in YAML.\n"; return false; }

    root[new_name] = root[selected_region_name_];
    root.remove(selected_region_name_);

    if (!SaveYamlWithHeader(field_points_yaml_, root, err)) { std::cout << "[Plane] " << err << "\n"; return false; }

    std::cout << "[Plane] Renamed '" << selected_region_name_ << "' -> '" << new_name << "'\n";
    selected_region_name_ = new_name;

    ReloadRegionsFromYaml();
    return true;
  }

  void DrawOverlay(cv::Mat & vis, double scale)
  {
    // Draw saved regions
    for (const auto & kv : regions_world_) {
      const std::string & name = kv.first;
      std::vector<cv::Point2f> poly_uv;
      if (!ProjectWorldToImagePolygon(kv.second, poly_uv)) continue;

      std::vector<cv::Point> poly_disp;
      poly_disp.reserve(poly_uv.size());
      for (const auto & p : poly_uv) poly_disp.emplace_back((int)std::lround(p.x * scale), (int)std::lround(p.y * scale));

      const bool selected = (!selected_region_name_.empty() && name == selected_region_name_);
      const cv::Scalar col = selected ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
      const int thick = selected ? 2 : 1;

      cv::polylines(vis, poly_disp, true, col, thick);
      if (!poly_disp.empty()) {
        cv::putText(vis, name, poly_disp[0] + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
      }
    }

    // Draw editing polygon
    for (size_t i = 0; i < pix_pts_orig_.size(); ++i) {
      const double x = pix_pts_orig_[i].x * scale;
      const double y = pix_pts_orig_[i].y * scale;

      cv::circle(vis, cv::Point((int)x, (int)y), 3, cv::Scalar(0, 255, 0), -1);
      if (i >= 1) {
        const double x0 = pix_pts_orig_[i - 1].x * scale;
        const double y0 = pix_pts_orig_[i - 1].y * scale;
        cv::line(vis, cv::Point((int)x0, (int)y0), cv::Point((int)x, (int)y), cv::Scalar(0, 255, 0), 1);
      }
    }
    if (closed_ && pix_pts_orig_.size() >= 3) {
      const double x0 = pix_pts_orig_.front().x * scale;
      const double y0 = pix_pts_orig_.front().y * scale;
      const double x1 = pix_pts_orig_.back().x * scale;
      const double y1 = pix_pts_orig_.back().y * scale;
      cv::line(vis, cv::Point((int)x1, (int)y1), cv::Point((int)x0, (int)y0), cv::Scalar(0, 255, 0), 2);
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "Editing: " << region_name_
        << "   Z: " << plane_z_
        << "   Pts: " << pix_pts_orig_.size()
        << "   State: " << (closed_ ? "Closed" : "Collecting")
        << "   Selected: " << (selected_region_name_.empty() ? "<none>" : selected_region_name_);

    cv::putText(vis, oss.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

    cv::putText(vis,
                "LClick:add  RClick:select  Enter:close  z:undo  H:Z  S:save  N:renameSelected  R:reset  Q/Esc:quit",
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
  }

  void OnImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv::Mat bgr;
    if (!DecodeToBGR(msg, bgr) || bgr.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Failed to decode image to BGR.");
      return;
    }

    const double scale = ComputeAutoFitScale(bgr.cols, bgr.rows);
    last_frame_scale_ = scale;

    cv::Mat vis;
    if (scale < 1.0) cv::resize(bgr, vis, cv::Size(), scale, scale, cv::INTER_AREA);
    else vis = bgr.clone();

    ui_inited_ = true;

    DrawOverlay(vis, scale);
    cv::imshow(window_name_, vis);

    if (z_dialog_.IsActive()) z_dialog_.Render();
    if (name_dialog_.IsActive()) name_dialog_.Render();

    int key = cv::waitKey(1);
    if (key < 0) return;

    // Dialog routing
    if (z_dialog_.IsActive()) {
      z_dialog_.HandleKey(key);
      if (!z_dialog_.IsActive()) {
        if (z_dialog_.IsConfirmed()) { plane_z_ = z_dialog_.GetZ(); std::cout << "[Plane] Set Z = " << plane_z_ << "\n"; }
        z_dialog_.Close();
      }
      return;
    }
    if (name_dialog_.IsActive()) {
      name_dialog_.HandleKey(key);
      if (!name_dialog_.IsActive()) {
        if (name_dialog_.IsConfirmed()) RenameSelectedRegion(name_dialog_.GetName());
        name_dialog_.Close();
      }
      return;
    }

    // Quit
    if (key == 27 || key == 'q' || key == 'Q') { rclcpp::shutdown(); return; }

    // Undo: z / Z
    if (key == 'z' || key == 'Z') { UndoLast(); return; }

    // Close polygon
    if (key == 10 || key == 13) {
      if (!ClosePolygon()) std::cout << "[Plane] Need at least 3 points to close.\n";
      return;
    }

    // Commands (uppercase)
    if (key == 'H') { z_dialog_.Open("Input Z (Enter=OK, Esc=Cancel)", plane_z_); return; }
    if (key == 'R') { ResetCurrent(); return; }
    if (key == 'S') { SaveCurrentRegion(); return; }

    // Rename selected region via N
    if (key == 'N') {
      if (selected_region_name_.empty()) {
        std::cout << "[Plane] No selected region. Right-click a drawn region first.\n";
        return;
      }
      name_dialog_.Open("Rename (Enter=OK, Esc=Cancel)", selected_region_name_);
      return;
    }
  }

private:
  std::string image_topic_;
  std::string intrinsics_yaml_;
  std::string out_matrix_yaml_;
  std::string field_points_yaml_;
  double default_plane_z_{0.0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  CameraModel cam_;
  RayPlaneIntersector intersector_;

  std::string window_name_;
  bool ui_inited_{false};

  // Editing
  std::string region_name_;
  double plane_z_{0.0};
  std::vector<cv::Point2d> pix_pts_orig_;
  bool closed_{false};

  // Selection
  std::string selected_region_name_;

  // Loaded regions
  std::map<std::string, std::vector<cv::Point3d>> regions_world_;

  // Display scale
  double last_frame_scale_{1.0};

  // Extrinsics cache for projection
  CameraExtrinsics ext_;
  cv::Vec3d rvec_{0.0, 0.0, 0.0};
  cv::Vec3d tvec_{0.0, 0.0, 0.0};

  ZInputDialog z_dialog_;
  NameInputDialog name_dialog_;
};

}  // namespace camera_field_calib_tool

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<camera_field_calib_tool::PlanePolygonNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
