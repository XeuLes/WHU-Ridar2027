#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace camera_field_calib_tool
{

class InteractiveCrosshair
{
public:
  InteractiveCrosshair();

  // Initialize with image size
  void SetImageSize(int width, int height);

  // Set initial position (default: image center)
  void SetPosition(const cv::Point2d & pos);
  cv::Point2d GetPosition() const;

  // Step control (pixels per key press)
  void SetStep(int step);
  int GetStep() const;

  // Process keyboard input (OpenCV waitKey code)
  // Returns true if state changed (position / step)
  bool HandleKey(int key);

  // Draw crosshair and info overlay
  void Draw(cv::Mat & image) const;

  // Enable / disable magnifier
  void EnableMagnifier(bool enable);
  bool IsMagnifierEnabled() const;

  // Magnifier parameters
  void SetMagnifierRadius(int radius_px);   // source radius
  void SetMagnifierScale(int scale);         // zoom factor

  // Draw magnifier window (call after Draw if enabled)
  void DrawMagnifier(const cv::Mat & src, cv::Mat & canvas) const;

  // Reset to center
  void Reset();

  // Utility: help string (for UI overlay)
  static std::string HelpString();

private:
  cv::Point2d pos_{0.0, 0.0};
  int width_{0};
  int height_{0};

  int step_{1};

  bool magnifier_enabled_{true};
  int magnifier_radius_{25};
  int magnifier_scale_{6};

  void ClampPosition();
};

}  // namespace camera_field_calib_tool
