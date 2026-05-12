#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

namespace hik_camera_ros2_driver
{

namespace
{

constexpr const char* kRedBold = "\033[1;31m";
constexpr const char* kColorReset = "\033[0m";

}  // namespace

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
    save_every_n_ = this->declare_parameter<int>("save_every_n", 10);                // 每N帧存1张
    keyboard_save_enabled_ = this->declare_parameter<bool>("keyboard_save_enabled", false);
    jpeg_quality_ = this->declare_parameter<int>("jpeg_quality", 95);               // jpg质量
    png_compression_ = this->declare_parameter<int>("png_compression", 3);          // png压缩级别 0~9
    use_timestamp_name_ = this->declare_parameter<bool>("use_timestamp_name", false);

    if (save_every_n_ < 0) {
      RCLCPP_WARN(this->get_logger(), "save_every_n < 0, force set to 0");
      save_every_n_ = 0;
    }

    if (!ensureDirectoryExists(save_dir_)) {
      throw std::runtime_error("failed to create save directory: " + save_dir_);
    }

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ImageSaverNode::imageCallback, this, std::placeholders::_1));
    startKeyboardThread();

    RCLCPP_INFO(this->get_logger(), "image_saver_node started");
    RCLCPP_INFO(this->get_logger(), "subscribe topic: %s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "save_dir: %s", save_dir_.c_str());
    RCLCPP_INFO(this->get_logger(), "image_format: %s", image_format_.c_str());
    RCLCPP_INFO(
      this->get_logger(),
      "save_every_n: %d%s",
      save_every_n_,
      save_every_n_ == 0 ? " (periodic save disabled)" : "");
    RCLCPP_INFO(this->get_logger(), "keyboard_save_enabled: %s", keyboard_save_enabled_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "use_timestamp_name: %s", use_timestamp_name_ ? "true" : "false");
  }

  ~ImageSaverNode() override
  {
    stopKeyboardThread();
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
      RCLCPP_ERROR(
        this->get_logger(),
        "%scv_bridge conversion failed, encoding=%s, error=%s%s",
        kRedBold, msg->encoding.c_str(), e.what(), kColorReset);
      return cv::Mat();
    }
  }

  void startKeyboardThread()
  {
    if (!keyboard_save_enabled_) {
      return;
    }

    keyboard_fd_ = STDIN_FILENO;
    if (!isatty(keyboard_fd_)) {
      keyboard_fd_ = open("/dev/tty", O_RDONLY | O_NONBLOCK);
    }

    if (keyboard_fd_ < 0 || !isatty(keyboard_fd_)) {
      RCLCPP_WARN(this->get_logger(), "keyboard save disabled: no controlling TTY");
      keyboard_save_enabled_ = false;
      return;
    }

    if (tcgetattr(keyboard_fd_, &original_termios_) != 0) {
      RCLCPP_WARN(this->get_logger(), "%skeyboard save disabled: tcgetattr failed%s", kRedBold, kColorReset);
      closeKeyboardFd();
      keyboard_save_enabled_ = false;
      return;
    }

    has_original_termios_ = true;
    termios raw = original_termios_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(keyboard_fd_, TCSANOW, &raw) != 0) {
      RCLCPP_WARN(this->get_logger(), "%skeyboard save disabled: tcsetattr failed%s", kRedBold, kColorReset);
      has_original_termios_ = false;
      closeKeyboardFd();
      keyboard_save_enabled_ = false;
      return;
    }

    keyboard_thread_running_.store(true);
    keyboard_thread_ = std::thread([this]() {
      while (keyboard_thread_running_.load()) {
        unsigned char ch = 0;
        const ssize_t n = read(keyboard_fd_, &ch, 1);
        if (n > 0) {
          if (std::tolower(ch) == 's') {
            pending_keyboard_saves_.fetch_add(1);
            RCLCPP_INFO(this->get_logger(), "S pressed: save next image frame");
          }
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
    });
  }

  void stopKeyboardThread()
  {
    keyboard_thread_running_.store(false);
    if (keyboard_thread_.joinable()) {
      keyboard_thread_.join();
    }
    if (has_original_termios_) {
      tcsetattr(keyboard_fd_, TCSANOW, &original_termios_);
      has_original_termios_ = false;
    }
    closeKeyboardFd();
  }

  void closeKeyboardFd()
  {
    if (keyboard_fd_ >= 0 && keyboard_fd_ != STDIN_FILENO) {
      close(keyboard_fd_);
    }
    keyboard_fd_ = -1;
  }

  bool saveImage(const sensor_msgs::msg::Image::SharedPtr msg, const char * reason)
  {
    cv::Mat image = convertToBgr(msg);
    if (image.empty()) {
      return false;
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
      RCLCPP_ERROR(this->get_logger(), "%scv::imwrite exception: %s%s", kRedBold, e.what(), kColorReset);
      return false;
    }

    if (!ok) {
      RCLCPP_ERROR(this->get_logger(), "%sfailed to save image: %s%s", kRedBold, filename.c_str(), kColorReset);
      return false;
    }

    ++saved_count_;

    if (reason != nullptr || (saved_count_ % 30) == 0) {
      RCLCPP_INFO(this->get_logger(), "saved image: %s (%s, total=%zu)",
                  filename.c_str(), reason != nullptr ? reason : "periodic", saved_count_);
    }
    return true;
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    ++frame_index_;

    const bool keyboard_requested = pending_keyboard_saves_.load() > 0;
    const bool periodic_requested =
      save_every_n_ > 0 && (frame_index_ % static_cast<size_t>(save_every_n_)) == 0;

    if (!keyboard_requested && !periodic_requested) {
      return;
    }

    const char * reason = keyboard_requested ? "keyboard" : nullptr;
    if (saveImage(msg, reason) && keyboard_requested) {
      pending_keyboard_saves_.fetch_sub(1);
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
  bool keyboard_save_enabled_;

  size_t frame_index_;
  size_t saved_count_;

  std::atomic<bool> keyboard_thread_running_{false};
  std::atomic<int> pending_keyboard_saves_{0};
  std::thread keyboard_thread_;
  int keyboard_fd_{-1};
  termios original_termios_ {};
  bool has_original_termios_{false};
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
//   -p save_dir:=/home/user/DATA/jichuzhizao/image1 
//   -p save_every_n:=1
