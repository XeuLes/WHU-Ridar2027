#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_interface/msg/detect_result.hpp>

namespace debug_map
{

namespace
{

constexpr int kRobotSlots = 7;
constexpr int kTrackSlots = kRobotSlots * 2;

struct PointSample {
  float x{0.0f};
  float y{0.0f};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

struct TrackStatus {
  std::string state{"NO_STATUS"};
  std::string output_mode{"NO_OUTPUT"};
  float confidence{0.0f};
  int hit_count{0};
  int semantic_hit_count{0};
  int lidar_hit_count{0};
  int miss_count{0};
  bool current_matched{false};
  bool publishable{false};
  bool has_blind_guess{false};
  float blind_guess_score{0.0f};
  bool valid{false};
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};

int slotIndex(char color, int number_id)
{
  if (number_id < 0 || number_id >= kRobotSlots) {
    return -1;
  }
  return (color == 'R') ? kRobotSlots + number_id : number_id;
}

std::string robotLabel(int slot)
{
  const bool red = slot >= kRobotSlots;
  const int number = red ? slot - kRobotSlots + 1 : slot + 1;
  return std::string(red ? "R" : "B") + std::to_string(number);
}

bool validPoint(float x, float y)
{
  return std::isfinite(x) && std::isfinite(y) &&
         (x != 0.0f || y != 0.0f);
}

std::vector<std::string> split(const std::string& line, char delimiter)
{
  std::vector<std::string> tokens;
  std::stringstream stream(line);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int parseInt(const std::string& text, int fallback = 0)
{
  try {
    return std::stoi(text);
  } catch (...) {
    return fallback;
  }
}

float parseFloat(const std::string& text, float fallback = 0.0f)
{
  try {
    return std::stof(text);
  } catch (...) {
    return fallback;
  }
}

cv::Scalar stateColor(const std::string& state)
{
  if (state == "CONFIRMED") {
    return cv::Scalar(80, 210, 120);
  }
  if (state == "LOST") {
    return cv::Scalar(60, 190, 255);
  }
  if (state == "TENTATIVE") {
    return cv::Scalar(90, 210, 245);
  }
  if (state == "INACTIVE") {
    return cv::Scalar(150, 150, 150);
  }
  return cv::Scalar(120, 120, 120);
}

std::string formatPoint(const PointSample& sample,
                        const rclcpp::Time& now,
                        double stale_timeout_s)
{
  if (!sample.valid || sample.stamp.nanoseconds() == 0) {
    return "--";
  }
  if (std::fabs((now - sample.stamp).seconds()) > stale_timeout_s) {
    return "--";
  }

  char text[64];
  std::snprintf(text, sizeof(text), "%.2f, %.2f", sample.x, sample.y);
  return text;
}

std::string formatConfidence(const TrackStatus& status,
                             const rclcpp::Time& now,
                             double stale_timeout_s)
{
  if (!status.valid || status.stamp.nanoseconds() == 0 ||
      std::fabs((now - status.stamp).seconds()) > stale_timeout_s) {
    return "--";
  }

  char text[32];
  std::snprintf(text, sizeof(text), "%.2f", status.confidence);
  return text;
}

}  // namespace

class TrackStatusWindowNode : public rclcpp::Node
{
public:
  explicit TrackStatusWindowNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("track_status_window_node", options)
  {
    detect_topic_ = declare_parameter<std::string>("detect_topic", "/detect_result");
    resolve_topic_ = declare_parameter<std::string>("resolve_topic", "/resolve_result");
    kalman_topic_ = declare_parameter<std::string>("kalman_topic", "/kalman_detect");
    debug_map_topic_ = declare_parameter<std::string>("debug_map_topic", "/debug_map_result");
    status_topic_ = declare_parameter<std::string>("status_topic", "/kalman_track_status");
    window_name_ = declare_parameter<std::string>("window_name", "Track Status Monitor");
    stale_timeout_s_ = declare_parameter<double>("stale_timeout_s", 1.0);
    const double refresh_hz =
      std::max(1.0, declare_parameter<double>("refresh_hz", 10.0));

    auto sensor_qos = rclcpp::SensorDataQoS();
    detect_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      detect_topic_, sensor_qos,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, detect_samples_);
      });
    resolve_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      resolve_topic_, sensor_qos,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, resolve_samples_);
      });
    kalman_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      kalman_topic_, 10,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, kalman_samples_);
      });
    debug_map_sub_ = create_subscription<vision_interface::msg::DetectResult>(
      debug_map_topic_, 10,
      [this](vision_interface::msg::DetectResult::SharedPtr msg) {
        updateSamples(msg, debug_map_samples_);
      });
    status_sub_ = create_subscription<std_msgs::msg::String>(
      status_topic_, 10,
      std::bind(&TrackStatusWindowNode::statusCallback, this, std::placeholders::_1));

    refresh_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / refresh_hz)),
      std::bind(&TrackStatusWindowNode::drawWindow, this));

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, 1280, 760);

    RCLCPP_INFO(
      get_logger(),
      "Track status window started. detect=%s resolve=%s kalman=%s debug_map=%s status=%s",
      detect_topic_.c_str(), resolve_topic_.c_str(), kalman_topic_.c_str(),
      debug_map_topic_.c_str(), status_topic_.c_str());
  }

  ~TrackStatusWindowNode() override
  {
    cv::destroyWindow(window_name_);
  }

private:
  void updateSamples(const vision_interface::msg::DetectResult::SharedPtr& msg,
                     std::array<PointSample, kTrackSlots>& samples)
  {
    const rclcpp::Time stamp = now();

    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kRobotSlots; ++i) {
      updateOneSample(samples[slotIndex('B', i)], msg->blue_x[i], msg->blue_y[i], stamp);
      updateOneSample(samples[slotIndex('R', i)], msg->red_x[i], msg->red_y[i], stamp);
    }
  }

  void updateOneSample(PointSample& sample, float x, float y, const rclcpp::Time& stamp)
  {
    sample.x = x;
    sample.y = y;
    sample.valid = validPoint(x, y);
    sample.stamp = stamp;
  }

  void statusCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    const rclcpp::Time stamp = now();
    std::stringstream stream(msg->data);
    std::string line;

    std::lock_guard<std::mutex> lock(mutex_);
    while (std::getline(stream, line)) {
      if (line.empty() || line.rfind("stamp,", 0) == 0) {
        continue;
      }

      const auto fields = split(line, ',');
      if (fields.size() < 8 || fields[0].empty()) {
        continue;
      }

      const char color = fields[0][0];
      const int number_id = parseInt(fields[1], -1);
      const int slot = slotIndex(color, number_id);
      if (slot < 0) {
        continue;
      }

      TrackStatus next;
      next.state = fields[2];
      next.confidence = parseFloat(fields[3]);
      next.hit_count = parseInt(fields[4]);
      next.semantic_hit_count = parseInt(fields[5]);
      next.lidar_hit_count = parseInt(fields[6]);
      next.miss_count = parseInt(fields[7]);
      if (fields.size() >= 16) {
        next.current_matched = parseInt(fields[14]) != 0;
        next.publishable = parseInt(fields[15]) != 0;
      }
      if (fields.size() >= 19) {
        next.output_mode = fields[16];
        next.has_blind_guess = parseInt(fields[17]) != 0;
        next.blind_guess_score = parseFloat(fields[18]);
      }
      next.valid = true;
      next.stamp = stamp;

      const auto& old = statuses_[slot];
      if (old.valid && old.state != next.state) {
        pushLog(robotLabel(slot) + " " + old.state + " -> " + next.state);
      }
      statuses_[slot] = next;
    }
  }

  void pushLog(const std::string& line)
  {
    event_log_.push_back(line);
    while (event_log_.size() > 8) {
      event_log_.pop_front();
    }
  }

  void drawWindow()
  {
    std::array<PointSample, kTrackSlots> detect;
    std::array<PointSample, kTrackSlots> resolve;
    std::array<PointSample, kTrackSlots> kalman;
    std::array<PointSample, kTrackSlots> debug_map;
    std::array<TrackStatus, kTrackSlots> statuses;
    std::deque<std::string> event_log;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      detect = detect_samples_;
      resolve = resolve_samples_;
      kalman = kalman_samples_;
      debug_map = debug_map_samples_;
      statuses = statuses_;
      event_log = event_log_;
    }

    const rclcpp::Time t = now();
    cv::Mat canvas(760, 1280, CV_8UC3, cv::Scalar(25, 27, 30));
    drawTitle(canvas, t);
    drawTable(canvas, t, detect, resolve, kalman, debug_map, statuses);
    drawEventLog(canvas, event_log);

    cv::imshow(window_name_, canvas);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      rclcpp::shutdown();
    }
  }

  void drawTitle(cv::Mat& canvas, const rclcpp::Time& t)
  {
    cv::putText(canvas, "Track Status Monitor", cv::Point(24, 36),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(240, 240, 240), 2,
                cv::LINE_AA);

    std::ostringstream subtitle;
    subtitle << "stale timeout " << std::fixed << std::setprecision(1)
             << stale_timeout_s_ << "s, time " << std::setprecision(3)
             << t.seconds();
    cv::putText(canvas, subtitle.str(), cv::Point(24, 62),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(170, 175, 180), 1,
                cv::LINE_AA);
  }

  void drawTable(cv::Mat& canvas,
                 const rclcpp::Time& t,
                 const std::array<PointSample, kTrackSlots>& detect,
                 const std::array<PointSample, kTrackSlots>& resolve,
                 const std::array<PointSample, kTrackSlots>& kalman,
                 const std::array<PointSample, kTrackSlots>& debug_map,
                 const std::array<TrackStatus, kTrackSlots>& statuses)
  {
    const std::array<int, 7> cols{24, 92, 236, 310, 510, 710, 910};
    const int header_y = 100;
    const int row_h = 34;
    const int first_row_y = 132;

    cv::line(canvas, cv::Point(18, 76), cv::Point(1262, 76),
             cv::Scalar(60, 64, 70), 1, cv::LINE_AA);
    cv::rectangle(canvas, cv::Rect(18, 84, 1244, 30),
                  cv::Scalar(38, 42, 47), cv::FILLED);

    drawText(canvas, "ID", cols[0], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "State", cols[1], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Conf", cols[2], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Detect", cols[3], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Resolve", cols[4], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Kalman", cols[5], header_y, 0.48, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "DebugMap", cols[6], header_y, 0.48, cv::Scalar(190, 198, 205), 1);

    for (int slot = 0; slot < kTrackSlots; ++slot) {
      const int y = first_row_y + slot * row_h;
      if (slot % 2 == 0) {
        cv::rectangle(canvas, cv::Rect(18, y - 22, 1244, row_h),
                      cv::Scalar(30, 33, 37), cv::FILLED);
      }
      if (slot == kRobotSlots) {
        cv::line(canvas, cv::Point(18, y - 28), cv::Point(1262, y - 28),
                 cv::Scalar(70, 76, 84), 1, cv::LINE_AA);
      }

      const bool red = slot >= kRobotSlots;
      const cv::Scalar id_color = red ? cv::Scalar(100, 120, 245)
                                      : cv::Scalar(245, 150, 80);
      const auto& status = statuses[slot];
      const bool status_fresh =
        status.valid && std::fabs((t - status.stamp).seconds()) <= stale_timeout_s_;
      std::string state = status_fresh ? status.state : "NO_STATUS";
      if (status_fresh) {
        if (!status.output_mode.empty() && status.output_mode != "NO_OUTPUT") {
          state += "/" + status.output_mode;
        }
        state += status.current_matched ? " M" : " -";
        if (status.publishable) {
          state += " P";
        }
        if (status.has_blind_guess) {
          char score[32];
          std::snprintf(score, sizeof(score), " G%.2f", status.blind_guess_score);
          state += score;
        }
      }

      drawText(canvas, robotLabel(slot), cols[0], y, 0.55, id_color, 2);
      drawText(canvas, state, cols[1], y, 0.46, stateColor(status.state), 1);
      drawText(canvas, formatConfidence(status, t, stale_timeout_s_), cols[2], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(detect[slot], t, stale_timeout_s_), cols[3], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(resolve[slot], t, stale_timeout_s_), cols[4], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(kalman[slot], t, stale_timeout_s_), cols[5], y,
               0.46, cv::Scalar(218, 220, 224), 1);
      drawText(canvas, formatPoint(debug_map[slot], t, stale_timeout_s_), cols[6], y,
               0.46, cv::Scalar(218, 220, 224), 1);
    }
  }

  void drawEventLog(cv::Mat& canvas, const std::deque<std::string>& event_log)
  {
    const int top = 630;
    cv::rectangle(canvas, cv::Rect(18, top - 24, 1244, 136),
                  cv::Scalar(31, 34, 38), cv::FILLED);
    cv::putText(canvas, "State Changes", cv::Point(24, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(190, 198, 205), 1,
                cv::LINE_AA);

    int y = top + 24;
    for (auto it = event_log.rbegin(); it != event_log.rend(); ++it) {
      drawText(canvas, *it, 28, y, 0.44, cv::Scalar(218, 220, 224), 1);
      y += 20;
    }
  }

  void drawText(cv::Mat& canvas,
                const std::string& text,
                int x,
                int y,
                double scale,
                const cv::Scalar& color,
                int thickness)
  {
    cv::putText(canvas, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv::LINE_AA);
  }

  std::string detect_topic_;
  std::string resolve_topic_;
  std::string kalman_topic_;
  std::string debug_map_topic_;
  std::string status_topic_;
  std::string window_name_;
  double stale_timeout_s_{1.0};

  std::mutex mutex_;
  std::array<PointSample, kTrackSlots> detect_samples_{};
  std::array<PointSample, kTrackSlots> resolve_samples_{};
  std::array<PointSample, kTrackSlots> kalman_samples_{};
  std::array<PointSample, kTrackSlots> debug_map_samples_{};
  std::array<TrackStatus, kTrackSlots> statuses_{};
  std::deque<std::string> event_log_;

  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr detect_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr resolve_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr kalman_sub_;
  rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr debug_map_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
  rclcpp::TimerBase::SharedPtr refresh_timer_;
};

}  // namespace debug_map

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<debug_map::TrackStatusWindowNode>());
  rclcpp::shutdown();
  return 0;
}
