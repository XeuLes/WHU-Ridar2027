#pragma once

#include <opencv2/core.hpp>

#include "camera_field_calib_tool/camera_model.hpp"

namespace camera_field_calib_tool
{

struct Plane
{
  // Plane equation: n^T X + d = 0
  // For Z = z0 plane: n = (0,0,1), d = -z0
  cv::Vec3d normal{0.0, 0.0, 1.0};
  double d{0.0};
};

struct Ray
{
  cv::Point3d origin;   // world coordinate
  cv::Vec3d direction;  // normalized direction (world)
};

class RayPlaneIntersector
{
public:
  RayPlaneIntersector() = default;

  // Set camera extrinsics (world <- camera)
  // rvec, tvec follow OpenCV convention:
  //   X_cam = R * X_world + t
  bool SetExtrinsics(
    const cv::Vec3d & rvec,
    const cv::Vec3d & tvec);

  bool IsValid() const;

  // -----------------------------
  // Core operations
  // -----------------------------

  // Build a world-space ray from pixel
  Ray PixelToWorldRay(
    const CameraModel & cam,
    const cv::Point2d & pixel) const;

  // Ray-plane intersection
  // Returns false if ray is parallel to plane or intersection behind ray origin
  bool IntersectRayPlane(
    const Ray & ray,
    const Plane & plane,
    cv::Point3d & intersection) const;

  // Convenience: pixel -> intersection with plane
  bool PixelIntersectPlane(
    const CameraModel & cam,
    const cv::Point2d & pixel,
    const Plane & plane,
    cv::Point3d & intersection) const;

  // Utility: build Z = constant plane
  static Plane MakeZPlane(double z);

private:
  bool valid_{false};

  // Cached transforms
  cv::Matx33d R_cw_{cv::Matx33d::eye()};  // camera -> world rotation
  cv::Vec3d t_cw_{0.0, 0.0, 0.0};         // camera origin in world
};

}  // namespace camera_field_calib_tool
