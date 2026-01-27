#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <algorithm>
#include <cmath>
#include <string>

class XXXTestNode final : public rclcpp::Node {
public:
  XXXTestNode() : rclcpp::Node("xxx_test_node")
  {
    // -----------------------------
    // Params
    // -----------------------------
    angles_topic_ = this->declare_parameter<std::string>("angles_topic", "/gimbal/angles");
    cmd_topic_    = this->declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
    enable_topic_ = this->declare_parameter<std::string>("enable_topic", "/gimbal/enable");
    debug_topic_  = this->declare_parameter<std::string>("debug_topic", "/gimbal/test_rx_angles");

    cmd_unit_     = this->declare_parameter<std::string>("cmd_unit", "deg"); // "deg" or "rad"

    yaw_min_deg_   = this->declare_parameter<double>("yaw_min_deg", -90.0);
    yaw_max_deg_   = this->declare_parameter<double>("yaw_max_deg",  90.0);
    yaw_speed_dps_ = this->declare_parameter<double>("yaw_speed_dps", 5.0);  // deg/s
    pitch_deg_     = this->declare_parameter<double>("pitch_deg", 0.0);

    publish_hz_ = this->declare_parameter<double>("publish_hz", 50.0); // cmd publish rate
    enable_hz_  = this->declare_parameter<double>("enable_hz", 5.0);   // keepalive enable publish

    // 参数健壮性处理
    if (publish_hz_ <= 0.0) publish_hz_ = 50.0;
    if (enable_hz_  <= 0.0) enable_hz_  = 5.0;

    if (yaw_max_deg_ < yaw_min_deg_) std::swap(yaw_max_deg_, yaw_min_deg_);

    const double span = yaw_max_deg_ - yaw_min_deg_;
    if (span <= 1e-6) {
      RCLCPP_WARN(this->get_logger(), "yaw range too small, forcing [-90, 90].");
      yaw_min_deg_ = -90.0;
      yaw_max_deg_ =  90.0;
    }

    if (cmd_unit_ != "deg" && cmd_unit_ != "rad") {
      RCLCPP_WARN(this->get_logger(), "cmd_unit invalid: '%s', fallback to 'deg'", cmd_unit_.c_str());
      cmd_unit_ = "deg";
    }

    // -----------------------------
    // Pub/Sub
    // -----------------------------
    pub_cmd_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic_, rclcpp::QoS(10));
    pub_enable_ = this->create_publisher<std_msgs::msg::Bool>(enable_topic_, rclcpp::QoS(10));

    sub_angles_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      angles_topic_, rclcpp::SensorDataQoS(),
      std::bind(&XXXTestNode::onAngles, this, std::placeholders::_1)
    );

    pub_debug_rx_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(debug_topic_, rclcpp::QoS(10));

    // ✅ 关键修复：start_time_ 放到构造函数中初始化（更稳）
    start_time_ = this->now();

    // -----------------------------
    // Timers
    // -----------------------------
    cmd_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_hz_)
      ),
      std::bind(&XXXTestNode::onCmdTimer, this)
    );

    enable_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / enable_hz_)
      ),
      std::bind(&XXXTestNode::onEnableTimer, this)
    );

    RCLCPP_INFO(
      this->get_logger(),
      "XXXTestNode started.\n"
      "  cmd_topic    = %s\n"
      "  enable_topic = %s\n"
      "  angles_topic = %s\n"
      "  debug_topic  = %s\n"
      "  cmd_unit     = %s\n"
      "  yaw_range    = [%.1f, %.1f] deg\n"
      "  yaw_speed    = %.2f deg/s\n"
      "  pitch_deg    = %.2f deg\n"
      "  publish_hz   = %.1f\n"
      "  enable_hz    = %.1f",
      cmd_topic_.c_str(),
      enable_topic_.c_str(),
      angles_topic_.c_str(),
      debug_topic_.c_str(),
      cmd_unit_.c_str(),
      yaw_min_deg_, yaw_max_deg_,
      yaw_speed_dps_,
      pitch_deg_,
      publish_hz_,
      enable_hz_
    );
  }

private:
  static constexpr double kPi = 3.14159265358979323846;
  static inline double deg2rad(double deg) { return deg * kPi / 180.0; }

  // 三角波：在 [min,max] 等速往复
  // 周期 T = 2*(max-min)/speed
  double triangleYawDeg(double t_sec) const
  {
    const double minv = yaw_min_deg_;
    const double maxv = yaw_max_deg_;
    const double span = maxv - minv;

    const double speed = std::max(1e-9, yaw_speed_dps_);
    const double T = 2.0 * span / speed;
    const double tm = std::fmod(t_sec, T);

    const double halfT = T / 2.0;
    if (tm <= halfT) {
      // min -> max
      double v = minv + speed * tm;
      if (v > maxv) v = maxv;
      return v;
    } else {
      // max -> min
      double v = maxv - speed * (tm - halfT);
      if (v < minv) v = minv;
      return v;
    }
  }

  void onEnableTimer()
  {
    std_msgs::msg::Bool en;
    en.data = true;
    pub_enable_->publish(en);
  }

  void onCmdTimer()
  {
    const rclcpp::Time now = this->now();
    const double t = (now - start_time_).seconds();

    const double yaw_deg   = triangleYawDeg(t);
    const double pitch_deg = pitch_deg_;

    double yaw_out   = yaw_deg;
    double pitch_out = pitch_deg;

    // 单位对齐：桥节点不会做单位转换，你这里要决定发 deg 还是 rad
    if (cmd_unit_ == "rad") {
      yaw_out   = deg2rad(yaw_deg);
      pitch_out = deg2rad(pitch_deg);
    }

    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.resize(2);
    cmd.data[0] = yaw_out;   // yaw_index=0
    cmd.data[1] = pitch_out; // pitch_index=1
    pub_cmd_->publish(cmd);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 500,
      "TX cmd: yaw=%.4f pitch=%.4f (%s)",
      cmd.data[0], cmd.data[1], cmd_unit_.c_str()
    );
  }

  void onAngles(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (!msg || msg->data.size() < 2) return;

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 500,
      "RX angles: yaw=%.4f pitch=%.4f (raw from bridge)",
      msg->data[0], msg->data[1]
    );

    // 转发到 debug 话题，方便 rqt_plot
    pub_debug_rx_->publish(*msg);
  }

private:
  // topics
  std::string angles_topic_;
  std::string cmd_topic_;
  std::string enable_topic_;
  std::string debug_topic_;

  // motion config
  std::string cmd_unit_; // "deg" or "rad"
  double yaw_min_deg_{-90.0};
  double yaw_max_deg_{90.0};
  double yaw_speed_dps_{5.0};
  double pitch_deg_{0.0};

  double publish_hz_{50.0};
  double enable_hz_{5.0};

  // ROS
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_enable_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_angles_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_debug_rx_;

  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr enable_timer_;

  // ✅ 初始化放构造函数
  rclcpp::Time start_time_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<XXXTestNode>());
  rclcpp::shutdown();
  return 0;
}
