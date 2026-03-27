#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>

namespace hik_camera_ros2_driver
{

class ImageSaverNode : public rclcpp::Node
{
public:
  ImageSaverNode()
  : Node("image_saver_node"),
    frame_index_(0),
    saved_count_(0)
  {
    topic_ = this->declare_parameter<std::string>("topic", "/camera1/image");
    save_dir_ = this->declare_parameter<std::string>("save_dir", "/tmp/camera1_frames");
    image_format_ = this->declare_parameter<std::string>("image_format", "jpg");   // jpg/png
    save_every_n_ = this->declare_parameter<int>("save_every_n", 1);                // 每N帧存1张
    jpeg_quality_ = this->declare_parameter<int>("jpeg_quality", 95);               // jpg质量
    png_compression_ = this->declare_parameter<int>("png_compression", 3);          // png压缩级别 0~9
    use_timestamp_name_ = this->declare_parameter<bool>("use_timestamp_name", false);

    if (save_every_n_ <= 0) {
      RCLCPP_WARN(this->get_logger(), "save_every_n <= 0, force set to 1");
      save_every_n_ = 1;
    }

    if (!ensureDirectoryExists(save_dir_)) {
      throw std::runtime_error("failed to create save directory: " + save_dir_);
    }

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ImageSaverNode::imageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "image_saver_node started");
    RCLCPP_INFO(this->get_logger(), "subscribe topic: %s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "save_dir: %s", save_dir_.c_str());
    RCLCPP_INFO(this->get_logger(), "image_format: %s", image_format_.c_str());
    RCLCPP_INFO(this->get_logger(), "save_every_n: %d", save_every_n_);
    RCLCPP_INFO(this->get_logger(), "use_timestamp_name: %s", use_timestamp_name_ ? "true" : "false");
  }

private:
  static bool dirExists(const std::string & path)
  {
    struct stat info {};
    return (stat(path.c_str(), &info) == 0) && (info.st_mode & S_IFDIR);
  }

  static bool makeDir(const std::string & path)
  {
    if (path.empty()) {
      return false;
    }
    if (dirExists(path)) {
      return true;
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
  }

  static bool ensureDirectoryExists(const std::string & path)
  {
    if (path.empty()) {
      return false;
    }
    if (dirExists(path)) {
      return true;
    }

    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
      char c = path[i];
      current.push_back(c);

      if (c == '/' && !current.empty()) {
        if (current.size() == 1) {  // "/"
          continue;
        }
        if (!dirExists(current) && !makeDir(current)) {
          return false;
        }
      }
    }

    if (!dirExists(path) && !makeDir(path)) {
      return false;
    }
    return true;
  }

  std::string buildFilename(const sensor_msgs::msg::Image & msg) const
  {
    std::ostringstream oss;

    if (use_timestamp_name_) {
      const auto sec = static_cast<long long>(msg.header.stamp.sec);
      const auto nsec = static_cast<long long>(msg.header.stamp.nanosec);
      oss << save_dir_ << "/frame_" << sec << "_" << std::setw(9) << std::setfill('0') << nsec
          << "." << image_format_;
    } else {
      oss << save_dir_ << "/frame_" << std::setw(6) << std::setfill('0') << saved_count_
          << "." << image_format_;
    }

    return oss.str();
  }

  cv::Mat convertToBgr(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      if (msg->encoding == "bgr8") {
        return cv_bridge::toCvCopy(msg, "bgr8")->image;
      }
      if (msg->encoding == "rgb8") {
        cv::Mat rgb = cv_bridge::toCvCopy(msg, "rgb8")->image;
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
      }
      if (msg->encoding == "mono8") {
        cv::Mat mono = cv_bridge::toCvCopy(msg, "mono8")->image;
        cv::Mat bgr;
        cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
      }

      return cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed, encoding=%s, error=%s",
                   msg->encoding.c_str(), e.what());
      return cv::Mat();
    }
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    ++frame_index_;

    if ((frame_index_ % save_every_n_) != 0) {
      return;
    }

    cv::Mat image = convertToBgr(msg);
    if (image.empty()) {
      return;
    }

    const std::string filename = buildFilename(*msg);

    std::vector<int> params;
    if (image_format_ == "jpg" || image_format_ == "jpeg") {
      params.push_back(cv::IMWRITE_JPEG_QUALITY);
      params.push_back(jpeg_quality_);
    } else if (image_format_ == "png") {
      params.push_back(cv::IMWRITE_PNG_COMPRESSION);
      params.push_back(png_compression_);
    }

    bool ok = false;
    try {
      ok = cv::imwrite(filename, image, params);
    } catch (const cv::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv::imwrite exception: %s", e.what());
      return;
    }

    if (!ok) {
      RCLCPP_ERROR(this->get_logger(), "failed to save image: %s", filename.c_str());
      return;
    }

    ++saved_count_;

    if ((saved_count_ % 30) == 0) {
      RCLCPP_INFO(this->get_logger(), "saved %zu images", saved_count_);
    }
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  std::string topic_;
  std::string save_dir_;
  std::string image_format_;

  int save_every_n_;
  int jpeg_quality_;
  int png_compression_;
  bool use_timestamp_name_;

  size_t frame_index_;
  size_t saved_count_;
};

}  // namespace hik_camera_ros2_driver

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<hik_camera_ros2_driver::ImageSaverNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    fprintf(stderr, "image_saver_node fatal error: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
} 
// ros2 run hik_camera_ros2_driver image_saver_node --ros-args 
//   -p save_dir:=/home/user/DATA/jiguangmokuai/image1 
//   -p save_every_n:=1


