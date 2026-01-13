#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace camera_field_calib_tool
{

// -----------------------------
// Data Models
// -----------------------------
struct CameraIntrinsics
{
  // 3x3 camera matrix
  cv::Matx33d K{};
  // distortion coefficients (k1,k2,p1,p2,k3,...) variable length
  std::vector<double> D{};
  // image size (optional; 0 if unknown)
  int width{0};
  int height{0};
};

struct CameraExtrinsics
{
  // Rodrigues rotation vector
  cv::Vec3d rvec{0.0, 0.0, 0.0};
  // translation vector
  cv::Vec3d tvec{0.0, 0.0, 0.0};

  // Optional metadata
  int num_points{0};
  double reproj_rmse{-1.0};
  double reproj_max{-1.0};
  std::string image_topic{};
  std::string timestamp{};
};

// Each plane region: name -> polygon vertices in world coordinates (Point3d)
using PlanePolygons = std::map<std::string, std::vector<cv::Point3d>>;

// -----------------------------
// YAML IO APIs
// -----------------------------
bool LoadCameraIntrinsicsYaml(
  const std::string & yaml_path,
  CameraIntrinsics & intr,
  std::string * err = nullptr);

bool SaveCameraIntrinsicsYaml(
  const std::string & yaml_path,
  const CameraIntrinsics & intr,
  std::string * err = nullptr);

// out_matrix.yaml
bool LoadExtrinsicsYaml(
  const std::string & yaml_path,
  CameraExtrinsics & ext,
  std::string * err = nullptr);

bool SaveExtrinsicsYaml(
  const std::string & yaml_path,
  const CameraExtrinsics & ext,
  std::string * err = nullptr);

// Field_Points.yaml: plane0..planeN
bool LoadPlanePolygonsYaml(
  const std::string & yaml_path,
  PlanePolygons & planes,
  std::string * err = nullptr);

// If overwrite=true, rewrite the whole file with provided planes.
// If overwrite=false, read existing file (if any), merge/replace keys in `planes`.
bool SavePlanePolygonsYaml(
  const std::string & yaml_path,
  const PlanePolygons & planes,
  bool overwrite = true,
  std::string * err = nullptr);

// Utility: validate expected plane name pattern "plane<idx>" (optional usage).
bool IsPlaneRegionNameValid(const std::string & name);

}  // namespace camera_field_calib_tool
