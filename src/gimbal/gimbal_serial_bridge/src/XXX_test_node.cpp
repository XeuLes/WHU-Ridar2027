#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <cmath>
#include <string>

class XXXTestNode final : public rclcpp::Node {
public:
  XXXTestNode() : rclcpp::Node("xxx_test_node") {
    // Params
    angles_topic_ = this->declare_parameter<std::string>("angles_topic", "/gimbal/angles");
    cmd_topic_    = this->declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
    enable_topic_ = this->declare_parameter<std::string>("enable_topic", "/gimbal/enable");
    debug_topic_  = this->declare_parameter<std::string>("debug_topic", "/gimbal/test_rx_angles");

    cmd_unit_     = this->declare_parameter<std::string>("cmd_unit", "deg"); // "deg" or "rad"
    yaw_min_deg_  = this->declare_parameter<double>("yaw_min_deg", -90.0);
    yaw_max_deg_  = this->declare_parameter<double>("yaw_max_deg",  90.0);
    yaw_speed_dps_= this->declare_parameter<double>("yaw_speed_dps", 5.0);   // deg/s
    pitch_deg_    = this->declare_parameter<double>("pitch_deg", 0.0);

    publish_hz_   = this->declare_parameter<double>("publish_hz", 50.0);      // command publish rate
    enable_hz_    = this->declare_parameter<double>("enable_hz", 5.0);        // keepalive enable publish

    if (publish_hz_ <= 0.0) publish_hz_ = 50.0;
    if (enable_hz_  <= 0.0) enable_hz_  = 5.0;

    if (yaw_max_deg_ < yaw_min_deg_) std::swap(yaw_max_deg_, yaw_min_deg_);
    const double span = yaw_max_deg_ - yaw_min_deg_;
    if (span <= 1e-6) {
      RCLCPP_WARN(this->get_logger(), "yaw range too small, forcing [-90, 90].");
      yaw_min_deg_ = -90.0;
      yaw_max_deg_ =  90.0;
    }

    // Pub/Sub
    pub_cmd_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic_, rclcpp::QoS(10));
    pub_enable_ = this->create_publisher<std_msgs::msg::Bool>(enable_topic_, rclcpp::QoS(10));

    // 订阅桥节点输出的角度（下位机->ROS）
    sub_angles_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      angles_topic_, rclcpp::SensorDataQoS(),
      std::bind(&XXXTestNode::onAngles, this, std::placeholders::_1)
    );

    // 可选：把收到的角度转发到测试话题，便于你 rqt_plot/echo
    pub_debug_rx_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(debug_topic_, rclcpp::QoS(10));

    // Timers
    last_time_ = this->now();

    cmd_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / publish_hz_)),
      std::bind(&XXXTestNode::onCmdTimer, this)
    );

    enable_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / enable_hz_)),
      std::bind(&XXXTestNode::onEnableTimer, this)
    );

    RCLCPP_INFO(this->get_logger(),
                "XXXTestNode started. cmd_topic=%s enable_topic=%s angles_topic=%s cmd_unit=%s yaw=[%.1f, %.1f] deg speed=%.2f deg/s",
                cmd_topic_.c_str(), enable_topic_.c_str(), angles_topic_.c_str(),
                cmd_unit_.c_str(), yaw_min_deg_, yaw_max_deg_, yaw_speed_dps_);
  }

private:
  static inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

  // 三角波：在 [min,max] 等速往复
  // 设周期 T = 2*(max-min)/speed（从 min 到 max 再回到 min）
  double triangleYawDeg(double t_sec) const {
    const double minv = yaw_min_deg_;
    const double maxv = yaw_max_deg_;
    const double span = maxv - minv;
    const double T = 2.0 * span / std::max(1e-9, yaw_speed_dps_);
    const double tm = std::fmod(t_sec, T);

    // 0..span/speed: 上行；span/speed..T: 下行
    const double halfT = T / 2.0;
    if (tm <= halfT) {
      // min -> max
      return minv + yaw_speed_dps_ * tm;
    } else {
      // max -> min
      return maxv - yaw_speed_dps_ * (tm - halfT);
    }
  }

  void onEnableTimer() {
    std_msgs::msg::Bool en;
    en.data = true;
    pub_enable_->publish(en);
  }

  void onCmdTimer() {
    const rclcpp::Time now = this->now();
    const double t = (now - start_time_).seconds();

    const double yaw_deg = triangleYawDeg(t);
    const double pitch_deg = pitch_deg_;

    double yaw_out = yaw_deg;
    double pitch_out = pitch_deg;

    // 与下位机协议单位对齐：默认 deg；如果要 rad 则转换
    if (cmd_unit_ == "rad") {
      yaw_out = deg2rad(yaw_deg);
      pitch_out = deg2rad(pitch_deg);
    }

    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.resize(2);
    cmd.data[0] = yaw_out;   // yaw_index=0
    cmd.data[1] = pitch_out; // pitch_index=1
    pub_cmd_->publish(cmd);
  }

  void onAngles(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // 桥节点发布的是 [yaw,pitch]，这里做健壮性处理
    if (!msg || msg->data.size() < 2) return;

    // 打印收到的下位机角度
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                         "RX angles: yaw=%.4f pitch=%.4f (raw from bridge)",
                         msg->data[0], msg->data[1]);

    // 转发到 debug 话题（可用于 rqt_plot）
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

  // ros
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_enable_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_angles_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_debug_rx_;

  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr enable_timer_;

  rclcpp::Time start_time_{this->now()};
  rclcpp::Time last_time_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<XXXTestNode>());
  rclcpp::shutdown();
  return 0;
}
