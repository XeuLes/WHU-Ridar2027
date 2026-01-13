#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace camera_field_calib_tool
{

enum class PolygonState
{
  Collecting = 0,
  Closed = 1
};

class InteractivePolygon
{
public:
  InteractivePolygon();

  // Bind to a window name for mouse callback
  // Note: caller must ensure cv::namedWindow(window_name) has been called.
  void BindWindow(const std::string & window_name);

  // Plane naming: plane0..planeN
  void SetPlaneIndex(int idx);
  int GetPlaneIndex() const;
  std::string CurrentPlaneName() const;

  // Height (uniform Z) for the plane
  void SetPlaneHeight(double z);
  double GetPlaneHeight() const;

  // Points
  const std::vector<cv::Point2d> & Points() const;
  std::vector<cv::Point2d> & PointsMutable();

  // State
  PolygonState State() const;
  bool IsClosed() const;

  // Operations
  void AddPoint(const cv::Point2d & p);
  bool UndoLastPoint();
  bool ClosePolygon();   // requires >=3 points
  void Reset();          // clear points and set state to collecting
  void NextPlane();      // idx++ and Reset()

  // Draw on image
  void Draw(cv::Mat & image) const;

  // Handle key input (OpenCV waitKey code). Returns true if state changed.
  // Keys (suggested):
  //  - Enter: close polygon
  //  - Backspace/Delete: undo
  //  - N: next plane
  //  - R: reset current
  bool HandleKey(int key);

  // Help
  static std::string HelpString();

private:
  std::string window_name_{};
  bool window_bound_{false};

  int plane_idx_{0};
  double plane_z_{0.0};

  PolygonState state_{PolygonState::Collecting};
  std::vector<cv::Point2d> pts_{};

  // Mouse callback trampoline
  static void MouseCallback(int event, int x, int y, int flags, void * userdata);
  void OnMouse(int event, int x, int y, int flags);

  void DrawOverlayText(cv::Mat & image) const;
};

}  // namespace camera_field_calib_tool
