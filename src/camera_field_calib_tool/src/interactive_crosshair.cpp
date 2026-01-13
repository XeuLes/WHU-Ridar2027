#include "camera_field_calib_tool/interactive_crosshair.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>

namespace camera_field_calib_tool
{

InteractiveCrosshair::InteractiveCrosshair()
{
}

void InteractiveCrosshair::SetImageSize(int width, int height)
{
  width_ = width;
  height_ = height;

  if (pos_.x == 0.0 && pos_.y == 0.0 && width_ > 0 && height_ > 0) {
    pos_.x = width_ * 0.5;
    pos_.y = height_ * 0.5;
  }
  ClampPosition();
}

void InteractiveCrosshair::SetPosition(const cv::Point2d & pos)
{
  pos_ = pos;
  ClampPosition();
}

cv::Point2d InteractiveCrosshair::GetPosition() const
{
  return pos_;
}

void InteractiveCrosshair::SetStep(int step)
{
  step_ = std::max(1, step);
}

int InteractiveCrosshair::GetStep() const
{
  return step_;
}

bool InteractiveCrosshair::HandleKey(int key)
{
  bool changed = false;

  switch (key) {
    // Movement: lowercase only to avoid conflicts with command keys
    case 'a':
      pos_.x -= step_;
      changed = true;
      break;
    case 'd':
      pos_.x += step_;
      changed = true;
      break;
    case 'w':
      pos_.y -= step_;
      changed = true;
      break;
    case 's':
      pos_.y += step_;
      changed = true;
      break;

    // Step control
    case '+':
    case '=':
      step_ = std::min(step_ * 2, 50);
      changed = true;
      break;
    case '-':
    case '_':
      step_ = std::max(step_ / 2, 1);
      changed = true;
      break;

    // Toggle magnifier
    case 'm':
    case 'M':
      magnifier_enabled_ = !magnifier_enabled_;
      changed = true;
      break;

    // Reset
    case 'x':
    case 'X':
      Reset();
      changed = true;
      break;

    default:
      break;
  }

  ClampPosition();
  return changed;
}


void InteractiveCrosshair::Draw(cv::Mat & image) const
{
  const cv::Point center(static_cast<int>(pos_.x), static_cast<int>(pos_.y));

  // Crosshair
  cv::line(
    image,
    cv::Point(center.x - 20, center.y),
    cv::Point(center.x + 20, center.y),
    cv::Scalar(0, 255, 0),
    1);

  cv::line(
    image,
    cv::Point(center.x, center.y - 20),
    cv::Point(center.x, center.y + 20),
    cv::Scalar(0, 255, 0),
    1);

  // Center dot
  cv::circle(image, center, 2, cv::Scalar(0, 0, 255), -1);

  // Info text
  std::ostringstream oss;
  oss << "Pos: (" << static_cast<int>(pos_.x) << ", "
      << static_cast<int>(pos_.y) << ")  "
      << "Step: " << step_
      << "  [M]agnifier: " << (magnifier_enabled_ ? "ON" : "OFF")
      << "  Reset:[X]";

  cv::putText(
    image,
    oss.str(),
    cv::Point(10, 20),
    cv::FONT_HERSHEY_SIMPLEX,
    0.5,
    cv::Scalar(0, 255, 0),
    1);
}

void InteractiveCrosshair::EnableMagnifier(bool enable)
{
  magnifier_enabled_ = enable;
}

bool InteractiveCrosshair::IsMagnifierEnabled() const
{
  return magnifier_enabled_;
}

void InteractiveCrosshair::SetMagnifierRadius(int radius_px)
{
  magnifier_radius_ = std::max(5, radius_px);
}

void InteractiveCrosshair::SetMagnifierScale(int scale)
{
  magnifier_scale_ = std::max(1, scale);
}

void InteractiveCrosshair::DrawMagnifier(
  const cv::Mat & src,
  cv::Mat & canvas) const
{
  if (!magnifier_enabled_ || src.empty() || canvas.empty()) return;

  const int cx = static_cast<int>(pos_.x);
  const int cy = static_cast<int>(pos_.y);
  const int r = magnifier_radius_;

  // Build source ROI in a safe way (no cols-1 off-by-one)
  cv::Rect roi(cx - r, cy - r, 2 * r, 2 * r);
  const cv::Rect src_bounds(0, 0, src.cols, src.rows);
  roi = roi & src_bounds;  // intersection

  if (roi.width <= 1 || roi.height <= 1) return;

  cv::Mat patch = src(roi);
  cv::Mat zoom;
  cv::resize(patch, zoom, cv::Size(), magnifier_scale_, magnifier_scale_, cv::INTER_NEAREST);

  // Place magnifier at top-right
  int dst_x = canvas.cols - zoom.cols - 10;
  int dst_y = 10;

  // If zoom is larger than canvas, clamp by resizing zoom down to fit
  if (dst_x < 0 || dst_y < 0) {
    // shrink zoom to fit inside canvas with small margin
    const int max_w = std::max(1, canvas.cols - 20);
    const int max_h = std::max(1, canvas.rows - 20);
    const double sx = static_cast<double>(max_w) / static_cast<double>(zoom.cols);
    const double sy = static_cast<double>(max_h) / static_cast<double>(zoom.rows);
    const double s = std::min(1.0, std::min(sx, sy));

    if (s <= 0.0) return;

    cv::Mat zoom2;
    cv::resize(zoom, zoom2, cv::Size(), s, s, cv::INTER_AREA);
    zoom = zoom2;

    dst_x = canvas.cols - zoom.cols - 10;
    dst_y = 10;
    if (dst_x < 0 || dst_y < 0) return;
  }

  cv::Rect dst_roi(dst_x, dst_y, zoom.cols, zoom.rows);
  const cv::Rect dst_bounds(0, 0, canvas.cols, canvas.rows);
  dst_roi = dst_roi & dst_bounds;

  if (dst_roi.width <= 1 || dst_roi.height <= 1) return;

  // If dst_roi got clipped, clip zoom accordingly
  cv::Mat zoom_clipped = zoom(cv::Rect(0, 0, dst_roi.width, dst_roi.height));
  zoom_clipped.copyTo(canvas(dst_roi));
  cv::rectangle(canvas, dst_roi, cv::Scalar(0, 255, 0), 1);
}

void InteractiveCrosshair::Reset()
{
  if (width_ > 0 && height_ > 0) {
    pos_.x = width_ * 0.5;
    pos_.y = height_ * 0.5;
  } else {
    pos_ = cv::Point2d(0.0, 0.0);
  }
}

void InteractiveCrosshair::ClampPosition()
{
  if (width_ > 0) {
    pos_.x = std::max(0.0, std::min(pos_.x, static_cast<double>(width_ - 1)));
  }
  if (height_ > 0) {
    pos_.y = std::max(0.0, std::min(pos_.y, static_cast<double>(height_ - 1)));
  }
}

std::string InteractiveCrosshair::HelpString()
{
  return "w/a/s/d: move  |  +/-: step  |  M: magnifier  |  X: reset";
}


}  // namespace camera_field_calib_tool
