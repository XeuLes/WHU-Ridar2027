#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <cerrno>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace hik_camera_ros2_driver
{

class VideoSaverNode : public rclcpp::Node
{
public:
  VideoSaverNode()
  : Node("video_saver_node"),
    frame_index_(0),
    written_count_(0)
  {
    topic_ = this->declare_parameter<std::string>("topic", "/camera1/image");
    output_file_ = this->declare_parameter<std::string>("output_file", "/tmp/camera1_video.avi");
    fourcc_ = this->declare_parameter<std::string>("fourcc", "MJPG");
    fps_ = this->declare_parameter<double>("fps", 30.0);
    save_every_n_ = this->declare_parameter<int>("save_every_n", 1);
    max_frames_ = this->declare_parameter<int>("max_frames", 0);
    use_timestamp_name_ = this->declare_parameter<bool>("use_timestamp_name", false);
    resize_to_first_frame_ = this->declare_parameter<bool>("resize_to_first_frame", true);

    if (fps_ <= 0.0) {
      RCLCPP_WARN(this->get_logger(), "fps <= 0, force set to 30.0");
      fps_ = 30.0;
    }

    if (save_every_n_ <= 0) {
      RCLCPP_WARN(this->get_logger(), "save_every_n <= 0, force set to 1");
      save_every_n_ = 1;
    }

    output_file_ = buildOutputFile(output_file_);
    const std::string parent_dir = parentDirectory(output_file_);
    if (!parent_dir.empty() && !ensureDirectoryExists(parent_dir)) {
      throw std::runtime_error("failed to create output directory: " + parent_dir);
    }

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&VideoSaverNode::imageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "video_saver_node started");
    RCLCPP_INFO(this->get_logger(), "subscribe topic: %s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "output_file: %s", output_file_.c_str());
    RCLCPP_INFO(this->get_logger(), "fourcc: %s", fourcc_.c_str());
    RCLCPP_INFO(this->get_logger(), "fps: %.3f", fps_);
    RCLCPP_INFO(this->get_logger(), "save_every_n: %d", save_every_n_);
    RCLCPP_INFO(this->get_logger(), "max_frames: %d", max_frames_);
  }

  ~VideoSaverNode() override
  {
    if (writer_.isOpened()) {
      writer_.release();
      RCLCPP_INFO(this->get_logger(), "video saved: %s (%zu frames)",
                  output_file_.c_str(), written_count_);
    }
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
      return true;
    }
    if (dirExists(path)) {
      return true;
    }

    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
      const char c = path[i];
      current.push_back(c);

      if (c == '/' && !current.empty()) {
        if (current.size() == 1) {
          continue;
        }
        if (!dirExists(current) && !makeDir(current)) {
          return false;
        }
      }
    }

    return dirExists(path) || makeDir(path);
  }

  static std::string parentDirectory(const std::string & file)
  {
    const size_t pos = file.find_last_of('/');
    if (pos == std::string::npos) {
      return {};
    }
    if (pos == 0) {
      return "/";
    }
    return file.substr(0, pos);
  }

  std::string buildOutputFile(const std::string & requested) const
  {
    if (!use_timestamp_name_) {
      return requested;
    }

    const size_t slash = requested.find_last_of('/');
    const size_t dot = requested.find_last_of('.');
    const bool has_ext =
      dot != std::string::npos && (slash == std::string::npos || dot > slash);

    const std::string stem = has_ext ? requested.substr(0, dot) : requested;
    const std::string ext = has_ext ? requested.substr(dot) : ".avi";

    std::time_t now = std::time(nullptr);
    std::tm tm_now {};
    localtime_r(&now, &tm_now);

    std::ostringstream oss;
    oss << stem << "_" << std::put_time(&tm_now, "%Y%m%d_%H%M%S") << ext;
    return oss.str();
  }

  static int fourccCode(const std::string & fourcc)
  {
    std::string code = fourcc;
    if (code.size() < 4) {
      code = "MJPG";
    }
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
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

  bool openWriter(const cv::Mat & image)
  {
    frame_size_ = image.size();
    const int fourcc = fourccCode(fourcc_);

    try {
      writer_.open(output_file_, fourcc, fps_, frame_size_, true);
    } catch (const cv::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv::VideoWriter open exception: %s", e.what());
      return false;
    }

    if (!writer_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "failed to open video writer: %s", output_file_.c_str());
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "video writer opened: %s (%dx%d @ %.3f fps)",
                output_file_.c_str(), frame_size_.width, frame_size_.height, fps_);
    return true;
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (max_frames_ > 0 && static_cast<int>(written_count_) >= max_frames_) {
      return;
    }

    ++frame_index_;
    if ((frame_index_ % save_every_n_) != 0) {
      return;
    }

    cv::Mat image = convertToBgr(msg);
    if (image.empty()) {
      return;
    }

    if (!writer_.isOpened() && !openWriter(image)) {
      return;
    }

    if (image.size() != frame_size_) {
      if (!resize_to_first_frame_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "skip frame with changed size: got %dx%d, expected %dx%d",
          image.cols, image.rows, frame_size_.width, frame_size_.height);
        return;
      }
      cv::resize(image, image, frame_size_);
    }

    try {
      writer_.write(image);
    } catch (const cv::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv::VideoWriter write exception: %s", e.what());
      return;
    }

    ++written_count_;
    if ((written_count_ % 120) == 0) {
      RCLCPP_INFO(this->get_logger(), "written %zu video frames", written_count_);
    }
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  cv::VideoWriter writer_;
  cv::Size frame_size_;

  std::string topic_;
  std::string output_file_;
  std::string fourcc_;
  double fps_;
  int save_every_n_;
  int max_frames_;
  bool use_timestamp_name_;
  bool resize_to_first_frame_;

  size_t frame_index_;
  size_t written_count_;
};

}  // namespace hik_camera_ros2_driver

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<hik_camera_ros2_driver::VideoSaverNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    fprintf(stderr, "video_saver_node fatal error: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
