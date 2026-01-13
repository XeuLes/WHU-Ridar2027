#include "camera_field_calib_tool/ray_plane_intersector.hpp"

#include <opencv2/calib3d.hpp>

namespace camera_field_calib_tool
{

bool RayPlaneIntersector::SetExtrinsics(
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec)
{
  // OpenCV convention:
  //   X_cam = R * X_world + t
  // So camera center in world:
  //   Cw = -R^T * t

  // Get R (world->camera) as Matx33d directly
  cv::Matx33d R_w2c;
  cv::Rodrigues(rvec, R_w2c);

  // camera->world rotation
  cv::Matx33d R_c2w = R_w2c.t();

  // camera center in world
  cv::Vec3d Cw = -(R_c2w * tvec);

  R_cw_ = R_c2w;
  t_cw_ = Cw;

  valid_ = true;
  return true;
}

bool RayPlaneIntersector::IsValid() const
{
  return valid_;
}

Ray RayPlaneIntersector::PixelToWorldRay(
  const CameraModel & cam,
  const cv::Point2d & pixel) const
{
  Ray ray;

  // Ray origin is camera center in world
  ray.origin = cv::Point3d(t_cw_);

  // Camera-space ray direction (not normalized)
  cv::Vec3d dir_cam = cam.PixelToNormalizedRay(pixel);

  // Transform to world space
  cv::Vec3d dir_world = R_cw_ * dir_cam;

  // Normalize
  const double norm = cv::norm(dir_world);
  if (norm > 0.0) {
    dir_world /= norm;
  }

  ray.direction = dir_world;
  return ray;
}

bool RayPlaneIntersector::IntersectRayPlane(
  const Ray & ray,
  const Plane & plane,
  cv::Point3d & intersection) const
{
  const double denom = plane.normal.dot(ray.direction);

  // Parallel or nearly parallel
  if (std::abs(denom) < 1e-9) {
    return false;
  }

  const double t =
    -(plane.normal.dot(cv::Vec3d(ray.origin.x, ray.origin.y, ray.origin.z)) + plane.d) / denom;

  // Intersection behind the camera
  if (t < 0.0) {
    return false;
  }

  intersection.x = ray.origin.x + t * ray.direction[0];
  intersection.y = ray.origin.y + t * ray.direction[1];
  intersection.z = ray.origin.z + t * ray.direction[2];

  return true;
}

bool RayPlaneIntersector::PixelIntersectPlane(
  const CameraModel & cam,
  const cv::Point2d & pixel,
  const Plane & plane,
  cv::Point3d & intersection) const
{
  if (!cam.IsValid() || !IsValid()) {
    return false;
  }

  Ray ray = PixelToWorldRay(cam, pixel);
  return IntersectRayPlane(ray, plane, intersection);
}

Plane RayPlaneIntersector::MakeZPlane(double z)
{
  Plane p;
  p.normal = cv::Vec3d(0.0, 0.0, 1.0);
  p.d = -z;
  return p;
}

}  // namespace camera_field_calib_tool
