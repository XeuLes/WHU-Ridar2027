#include "camera_field_calib_tool/yaml_io.hpp"

#include <fstream>
#include <regex>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace camera_field_calib_tool
{
namespace
{

static void SetErr(std::string * err, const std::string & msg)
{
  if (err) {
    *err = msg;
  }
}

static bool FileExists(const std::string & path)
{
  std::ifstream f(path);
  return f.good();
}

static bool ParseVec3(const YAML::Node & node, cv::Vec3d & out)
{
  if (!node || !node.IsSequence() || node.size() != 3) return false;
  out[0] = node[0].as<double>();
  out[1] = node[1].as<double>();
  out[2] = node[2].as<double>();
  return true;
}

static bool ParseMat33(const YAML::Node & node, cv::Matx33d & out)
{
  // Accept either:
  // 1) flat list of 9 numbers
  // 2) list of 3 lists (3 numbers each)
  if (!node) return false;

  if (node.IsSequence() && node.size() == 9) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        out(r, c) = node[r * 3 + c].as<double>();
      }
    }
    return true;
  }

  if (node.IsSequence() && node.size() == 3) {
    for (int r = 0; r < 3; ++r) {
      if (!node[r].IsSequence() || node[r].size() != 3) return false;
      for (int c = 0; c < 3; ++c) {
        out(r, c) = node[r][c].as<double>();
      }
    }
    return true;
  }

  return false;
}

static YAML::Node ToYamlVec3(const cv::Vec3d & v)
{
  YAML::Node n(YAML::NodeType::Sequence);
  n.push_back(v[0]);
  n.push_back(v[1]);
  n.push_back(v[2]);
  return n;
}

static YAML::Node ToYamlMat33(const cv::Matx33d & m)
{
  // Write as 3x3 nested list for readability
  YAML::Node n(YAML::NodeType::Sequence);
  for (int r = 0; r < 3; ++r) {
    YAML::Node row(YAML::NodeType::Sequence);
    for (int c = 0; c < 3; ++c) row.push_back(m(r, c));
    n.push_back(row);
  }
  return n;
}

static bool ParsePoint3List(const YAML::Node & node, std::vector<cv::Point3d> & pts)
{
  // Expect: [ {x:..., y:..., z:...}, ... ]
  if (!node || !node.IsSequence()) return false;

  pts.clear();
  pts.reserve(node.size());

  for (std::size_t i = 0; i < node.size(); ++i) {
    const auto & p = node[i];
    if (!p || !p.IsMap()) return false;
    if (!p["x"] || !p["y"] || !p["z"]) return false;
    cv::Point3d pt;
    pt.x = p["x"].as<double>();
    pt.y = p["y"].as<double>();
    pt.z = p["z"].as<double>();
    pts.push_back(pt);
  }
  return true;
}

static YAML::Node ToYamlPoint3List(const std::vector<cv::Point3d> & pts)
{
  YAML::Node arr(YAML::NodeType::Sequence);
  for (const auto & pt : pts) {
    YAML::Node p(YAML::NodeType::Map);
    p["x"] = pt.x;
    p["y"] = pt.y;
    p["z"] = pt.z;
    arr.push_back(p);
  }
  return arr;
}

}  // namespace

bool IsPlaneRegionNameValid(const std::string & name)
{
  static const std::regex re("^plane[0-9]+$");
  return std::regex_match(name, re);
}

// -----------------------------
// Camera intrinsics
// -----------------------------
bool LoadCameraIntrinsicsYaml(
  const std::string & yaml_path,
  CameraIntrinsics & intr,
  std::string * err)
{
  try {
    if (!FileExists(yaml_path)) {
      SetErr(err, "Intrinsics YAML not found: " + yaml_path);
      return false;
    }

    YAML::Node root = YAML::LoadFile(yaml_path);

    // Supported layouts:
    // A) {camera_matrix: [[...],[...],[...]], dist_coeffs: [...], width:..., height:...}
    // B) ROS camera_info style:
    //    camera_matrix: {data:[...]} or K:[...], D:[...], width/height
    // We parse flexibly.

    // Width/height (optional)
    if (root["width"]) intr.width = root["width"].as<int>();
    if (root["height"]) intr.height = root["height"].as<int>();

    // K
    if (root["camera_matrix"]) {
      const auto & cm = root["camera_matrix"];
      if (cm.IsMap() && cm["data"]) {
        if (!ParseMat33(cm["data"], intr.K)) {
          SetErr(err, "Failed to parse camera_matrix.data as 3x3.");
          return false;
        }
      } else {
        if (!ParseMat33(cm, intr.K)) {
          SetErr(err, "Failed to parse camera_matrix as 3x3.");
          return false;
        }
      }
    } else if (root["K"]) {
      if (!ParseMat33(root["K"], intr.K)) {
        SetErr(err, "Failed to parse K as 3x3.");
        return false;
      }
    } else {
      SetErr(err, "Missing camera_matrix (or K) in intrinsics YAML.");
      return false;
    }

    // D
    intr.D.clear();
    if (root["dist_coeffs"]) {
      const auto & d = root["dist_coeffs"];
      if (!d.IsSequence()) {
        SetErr(err, "dist_coeffs must be a YAML sequence.");
        return false;
      }
      intr.D.reserve(d.size());
      for (std::size_t i = 0; i < d.size(); ++i) intr.D.push_back(d[i].as<double>());
    } else if (root["D"]) {
      const auto & d = root["D"];
      if (!d.IsSequence()) {
        SetErr(err, "D must be a YAML sequence.");
        return false;
      }
      intr.D.reserve(d.size());
      for (std::size_t i = 0; i < d.size(); ++i) intr.D.push_back(d[i].as<double>());
    } else {
      // Allow empty distortion (some setups)
      intr.D.clear();
    }

    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while loading intrinsics YAML: ") + e.what());
    return false;
  }
}

bool SaveCameraIntrinsicsYaml(
  const std::string & yaml_path,
  const CameraIntrinsics & intr,
  std::string * err)
{
  try {
    YAML::Node root(YAML::NodeType::Map);

    root["width"] = intr.width;
    root["height"] = intr.height;
    root["camera_matrix"] = ToYamlMat33(intr.K);

    YAML::Node d(YAML::NodeType::Sequence);
    for (double v : intr.D) d.push_back(v);
    root["dist_coeffs"] = d;

    std::ofstream ofs(yaml_path);
    if (!ofs.is_open()) {
      SetErr(err, "Failed to open for write: " + yaml_path);
      return false;
    }
    ofs << root;
    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while saving intrinsics YAML: ") + e.what());
    return false;
  }
}

// -----------------------------
// Extrinsics (out_matrix.yaml)
// -----------------------------
bool LoadExtrinsicsYaml(
  const std::string & yaml_path,
  CameraExtrinsics & ext,
  std::string * err)
{
  try {
    if (!FileExists(yaml_path)) {
      SetErr(err, "Extrinsics YAML not found: " + yaml_path);
      return false;
    }

    YAML::Node root = YAML::LoadFile(yaml_path);

    // Required fields
    if (!root["world_rvec"] || !root["world_tvec"]) {
      SetErr(err, "Missing world_rvec/world_tvec in extrinsics YAML.");
      return false;
    }
    if (!ParseVec3(root["world_rvec"], ext.rvec) || !ParseVec3(root["world_tvec"], ext.tvec)) {
      SetErr(err, "Failed to parse world_rvec/world_tvec as vec3.");
      return false;
    }

    // Optional metadata
    if (root["num_points"]) ext.num_points = root["num_points"].as<int>();
    if (root["reproj_rmse"]) ext.reproj_rmse = root["reproj_rmse"].as<double>();
    if (root["reproj_max"]) ext.reproj_max = root["reproj_max"].as<double>();
    if (root["image_topic"]) ext.image_topic = root["image_topic"].as<std::string>();
    if (root["timestamp"]) ext.timestamp = root["timestamp"].as<std::string>();

    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while loading extrinsics YAML: ") + e.what());
    return false;
  }
}

bool SaveExtrinsicsYaml(
  const std::string & yaml_path,
  const CameraExtrinsics & ext,
  std::string * err)
{
  try {
    YAML::Node root(YAML::NodeType::Map);

    root["world_rvec"] = ToYamlVec3(ext.rvec);
    root["world_tvec"] = ToYamlVec3(ext.tvec);

    // Optional metadata
    root["num_points"] = ext.num_points;
    root["reproj_rmse"] = ext.reproj_rmse;
    root["reproj_max"] = ext.reproj_max;
    if (!ext.image_topic.empty()) root["image_topic"] = ext.image_topic;
    if (!ext.timestamp.empty()) root["timestamp"] = ext.timestamp;

    std::ofstream ofs(yaml_path);
    if (!ofs.is_open()) {
      SetErr(err, "Failed to open for write: " + yaml_path);
      return false;
    }
    ofs << root;
    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while saving extrinsics YAML: ") + e.what());
    return false;
  }
}

// -----------------------------
// Plane polygons (Field_Points.yaml)
// -----------------------------
bool LoadPlanePolygonsYaml(
  const std::string & yaml_path,
  PlanePolygons & planes,
  std::string * err)
{
  try {
    if (!FileExists(yaml_path)) {
      SetErr(err, "Plane polygons YAML not found: " + yaml_path);
      return false;
    }

    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root || !root.IsMap()) {
      SetErr(err, "Plane polygons YAML root must be a map.");
      return false;
    }

    planes.clear();

    for (auto it = root.begin(); it != root.end(); ++it) {
      const std::string key = it->first.as<std::string>();
      const YAML::Node val = it->second;

      // Only load plane0..planeN (ignore meta keys if any)
      if (!IsPlaneRegionNameValid(key)) {
        continue;
      }

      std::vector<cv::Point3d> pts;
      if (!ParsePoint3List(val, pts)) {
        SetErr(err, "Failed to parse point list for key: " + key);
        return false;
      }
      planes[key] = std::move(pts);
    }

    if (planes.empty()) {
      SetErr(err, "No valid plane regions found (expected keys like plane0, plane1...).");
      return false;
    }

    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while loading plane polygons YAML: ") + e.what());
    return false;
  }
}

bool SavePlanePolygonsYaml(
  const std::string & yaml_path,
  const PlanePolygons & planes,
  bool overwrite,
  std::string * err)
{
  try {
    YAML::Node root(YAML::NodeType::Map);

    if (!overwrite && FileExists(yaml_path)) {
      // Load existing and merge
      YAML::Node existing = YAML::LoadFile(yaml_path);
      if (existing && existing.IsMap()) {
        root = existing;
      }
    }

    for (const auto & kv : planes) {
      const std::string & name = kv.first;
      const auto & pts = kv.second;

      if (!IsPlaneRegionNameValid(name)) {
        SetErr(err, "Invalid plane region name: " + name + " (expected plane0..planeN).");
        return false;
      }
      if (pts.size() < 3) {
        SetErr(err, "Plane polygon must have at least 3 points: " + name);
        return false;
      }

      root[name] = ToYamlPoint3List(pts);
    }

    std::ofstream ofs(yaml_path);
    if (!ofs.is_open()) {
      SetErr(err, "Failed to open for write: " + yaml_path);
      return false;
    }
    ofs << root;
    return true;
  } catch (const std::exception & e) {
    SetErr(err, std::string("Exception while saving plane polygons YAML: ") + e.what());
    return false;
  }
}

}  // namespace camera_field_calib_tool
