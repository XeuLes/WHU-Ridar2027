#include "camera_field_calib_tool/interactive_polygon.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>

namespace camera_field_calib_tool
{

InteractivePolygon::InteractivePolygon()
{
}

void InteractivePolygon::BindWindow(const std::string & window_name)
{
  window_name_ = window_name;
  cv::setMouseCallback(window_name_.c_str(), &InteractivePolygon::MouseCallback, this);
  window_bound_ = true;
}

void InteractivePolygon::SetPlaneIndex(int idx)
{
  plane_idx_ = std::max(0, idx);
}

int InteractivePolygon::GetPlaneIndex() const
{
  return plane_idx_;
}

std::string InteractivePolygon::CurrentPlaneName() const
{
  std::ostringstream oss;
  oss << "plane" << plane_idx_;
  return oss.str();
}

void InteractivePolygon::SetPlaneHeight(double z)
{
  plane_z_ = z;
}

double InteractivePolygon::GetPlaneHeight() const
{
  return plane_z_;
}

const std::vector<cv::Point2d> & InteractivePolygon::Points() const
{
  return pts_;
}

std::vector<cv::Point2d> & InteractivePolygon::PointsMutable()
{
  return pts_;
}

PolygonState InteractivePolygon::State() const
{
  return state_;
}

bool InteractivePolygon::IsClosed() const
{
  return state_ == PolygonState::Closed;
}

void InteractivePolygon::AddPoint(const cv::Point2d & p)
{
  if (state_ == PolygonState::Closed) {
    return;
  }
  pts_.push_back(p);
}

bool InteractivePolygon::UndoLastPoint()
{
  if (pts_.empty()) return false;

  if (state_ == PolygonState::Closed) {
    state_ = PolygonState::Collecting;
  }

  pts_.pop_back();
  return true;
}

bool InteractivePolygon::ClosePolygon()
{
  if (state_ == PolygonState::Closed) return true;
  if (pts_.size() < 3) return false;
  state_ = PolygonState::Closed;
  return true;
}

void InteractivePolygon::Reset()
{
  pts_.clear();
  state_ = PolygonState::Collecting;
}

void InteractivePolygon::NextPlane()
{
  plane_idx_ += 1;
  Reset();
}

bool InteractivePolygon::HandleKey(int key)
{
  bool changed = false;

  // Enter key in OpenCV may appear as 10 or 13 depending on backend.
  if (key == 10 || key == 13) {
    changed = ClosePolygon() || changed;
    return changed;
  }

  switch (key) {
    // Undo: Backspace(8) or Delete(127)
    case 8:
    case 127:
      changed = UndoLastPoint() || changed;
      break;

    // Reset current polygon
    case 'r':
    case 'R':
      Reset();
      changed = true;
      break;

    // Next plane
    case 'n':
    case 'N':
      NextPlane();
      changed = true;
      break;

    default:
      break;
  }

  return changed;
}

void InteractivePolygon::Draw(cv::Mat & image) const
{
  // Draw points
  for (size_t i = 0; i < pts_.size(); ++i) {
    cv::circle(image, cv::Point(static_cast<int>(pts_[i].x), static_cast<int>(pts_[i].y)),
               3, cv::Scalar(0, 0, 255), -1);

    // index label
    std::ostringstream oss;
    oss << i;
    cv::putText(image, oss.str(),
                cv::Point(static_cast<int>(pts_[i].x) + 5, static_cast<int>(pts_[i].y) - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
  }

  // Draw lines
  for (size_t i = 1; i < pts_.size(); ++i) {
    cv::line(image,
             cv::Point(static_cast<int>(pts_[i - 1].x), static_cast<int>(pts_[i - 1].y)),
             cv::Point(static_cast<int>(pts_[i].x), static_cast<int>(pts_[i].y)),
             cv::Scalar(0, 255, 0), 1);
  }

  // Close polygon line
  if (state_ == PolygonState::Closed && pts_.size() >= 3) {
    cv::line(image,
             cv::Point(static_cast<int>(pts_.back().x), static_cast<int>(pts_.back().y)),
             cv::Point(static_cast<int>(pts_.front().x), static_cast<int>(pts_.front().y)),
             cv::Scalar(0, 255, 0), 2);
  }

  DrawOverlayText(image);
}

std::string InteractivePolygon::HelpString()
{
  return "Mouse L: add point | Mouse R: undo | Enter: close | Backspace: undo | R: reset | N: next plane";
}

void InteractivePolygon::MouseCallback(int event, int x, int y, int flags, void * userdata)
{
  (void)flags;
  auto * self = reinterpret_cast<InteractivePolygon *>(userdata);
  if (self) {
    self->OnMouse(event, x, y, flags);
  }
}

void InteractivePolygon::OnMouse(int event, int x, int y, int flags)
{
  (void)flags;
  if (!window_bound_) return;

  if (event == cv::EVENT_LBUTTONDOWN) {
    AddPoint(cv::Point2d(static_cast<double>(x), static_cast<double>(y)));
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    UndoLastPoint();
  }
}

void InteractivePolygon::DrawOverlayText(cv::Mat & image) const
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);

  oss << "Region: " << CurrentPlaneName()
      << "  Z: " << plane_z_
      << "  Pts: " << pts_.size()
      << "  State: " << (state_ == PolygonState::Closed ? "Closed" : "Collecting");

  cv::putText(
    image,
    oss.str(),
    cv::Point(10, 40),
    cv::FONT_HERSHEY_SIMPLEX,
    0.5,
    cv::Scalar(0, 255, 0),
    1);

  cv::putText(
    image,
    HelpString(),
    cv::Point(10, 60),
    cv::FONT_HERSHEY_SIMPLEX,
    0.5,
    cv::Scalar(0, 255, 0),
    1);
}

}  // namespace camera_field_calib_tool
