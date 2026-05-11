#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/msg/map_publish_request.hpp"
#include "radar_decision_comm/msg/radar_decision_command.hpp"
#include "radar_decision_comm/msg/robot_interaction_request.hpp"

namespace radar_decision_comm
{

static std::chrono::milliseconds hz_to_period_ms(double hz)
{
  if (hz <= 0.0) {
    return std::chrono::milliseconds(1000);
  }
  auto ms = static_cast<int>(1000.0 / hz);
  if (ms < 1) {
    ms = 1;
  }
  return std::chrono::milliseconds(ms);
}

static int hz_to_min_interval_ms(double hz)
{
  if (hz <= 0.0) {
    return 0;
  }
  const int ms = static_cast<int>(std::lround(1000.0 / hz));
  return std::max(ms, 1);
}

template<typename MsgT>
struct StreamState
{
  MsgT pending_msg{};
  MsgT last_msg{};
  std::string pending_sig;
  std::string last_sig;
  bool has_pending{false};
  bool has_last{false};
  std::chrono::steady_clock::time_point last_pub_tp{};
};

class TxSchedulerComponent : public rclcpp::Node
{
public:
  explicit TxSchedulerComponent(const rclcpp::NodeOptions & options)
  : Node("tx_scheduler", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    scheduler_rate_hz_ = this->declare_parameter<double>("scheduler_rate_hz", 100.0);
    deduplicate_ = this->declare_parameter<bool>("deduplicate", true);
    prefer_latest_ = this->declare_parameter<bool>("prefer_latest", true);
    enable_rate_limit_ = this->declare_parameter<bool>("enable_rate_limit", true);

    map_max_hz_ = this->declare_parameter<double>("map_max_hz", 5.0);
    interaction_max_hz_ = this->declare_parameter<double>("interaction_max_hz", 30.0);
    radar_cmd_max_hz_ = this->declare_parameter<double>("radar_cmd_max_hz", 10.0);

    same_map_resend_ms_ = this->declare_parameter<int>("same_map_resend_ms", 1000);
    same_interaction_resend_ms_ = this->declare_parameter<int>("same_interaction_resend_ms", 1000);
    same_radar_cmd_resend_ms_ = this->declare_parameter<int>("same_radar_cmd_resend_ms", 1000);

    input_map_request_topic_ = this->declare_parameter<std::string>(
      "input_map_request_topic", "/radar/decision/map_publish_request");
    input_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "input_robot_interaction_topic", "/radar/decision/robot_interaction_request");
    input_radar_command_topic_ = this->declare_parameter<std::string>(
      "input_radar_command_topic", "/radar/decision/radar_command");

    output_map_request_topic_ = this->declare_parameter<std::string>(
      "output_map_request_topic", "/radar/tx/map_robot_data");
    output_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "output_robot_interaction_topic", "/radar/tx/robot_interaction");
    output_radar_command_topic_ = this->declare_parameter<std::string>(
      "output_radar_command_topic", "/radar/tx/radar_cmd");

    map_pub_ = this->create_publisher<radar_decision_comm::msg::MapPublishRequest>(
      output_map_request_topic_, 10);
    interaction_pub_ = this->create_publisher<radar_decision_comm::msg::RobotInteractionRequest>(
      output_robot_interaction_topic_, 10);
    radar_cmd_pub_ = this->create_publisher<radar_decision_comm::msg::RadarDecisionCommand>(
      output_radar_command_topic_, 10);

    map_sub_ = this->create_subscription<radar_decision_comm::msg::MapPublishRequest>(
      input_map_request_topic_, 10,
      [this](const radar_decision_comm::msg::MapPublishRequest::SharedPtr msg)
      {
        push_pending_map(*msg);
      });

    interaction_sub_ = this->create_subscription<radar_decision_comm::msg::RobotInteractionRequest>(
      input_robot_interaction_topic_, 20,
      [this](const radar_decision_comm::msg::RobotInteractionRequest::SharedPtr msg)
      {
        push_pending_interaction(*msg);
      });

    radar_cmd_sub_ = this->create_subscription<radar_decision_comm::msg::RadarDecisionCommand>(
      input_radar_command_topic_, 10,
      [this](const radar_decision_comm::msg::RadarDecisionCommand::SharedPtr msg)
      {
        push_pending_radar_cmd(*msg);
      });

    timer_ = this->create_wall_timer(
      hz_to_period_ms(scheduler_rate_hz_),
      [this]() { on_timer(); });

    RCLCPP_INFO(
      get_logger(),
      "TxSchedulerComponent started. enabled=%s rate_limit=%s dedup=%s prefer_latest=%s map_max_hz=%.1f interaction_max_hz=%.1f radar_cmd_max_hz=%.1f",
      enabled_ ? "true" : "false",
      enable_rate_limit_ ? "true" : "false",
      deduplicate_ ? "true" : "false",
      prefer_latest_ ? "true" : "false",
      map_max_hz_,
      interaction_max_hz_,
      radar_cmd_max_hz_);
  }

private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  static bool due(
    const TimePoint & now,
    const TimePoint & last,
    int period_ms)
  {
    return (now - last) >= std::chrono::milliseconds(period_ms);
  }

  std::string map_signature(const radar_decision_comm::msg::MapPublishRequest & msg) const
  {
    std::ostringstream oss;
    oss << "force=" << (msg.force ? 1 : 0) << ";n=" << msg.robots.size() << ";";
    for (const auto & robot : msg.robots) {
      const int x_cm = static_cast<int>(std::lround(robot.field_position.x * 100.0));
      const int y_cm = static_cast<int>(std::lround(robot.field_position.y * 100.0));
      oss << robot.robot_id << ":" << x_cm << "," << y_cm << ";";
    }
    return oss.str();
  }

  std::string interaction_bucket_key(
    const radar_decision_comm::msg::RobotInteractionRequest & msg) const
  {
    std::ostringstream oss;
    oss << msg.data_cmd_id << "|" << msg.receiver_id;
    return oss.str();
  }

  std::string interaction_signature(
    const radar_decision_comm::msg::RobotInteractionRequest & msg) const
  {
    std::ostringstream oss;
    oss << msg.data_cmd_id << "|" << msg.sender_id << "|" << msg.receiver_id << "|";
    for (auto b : msg.user_data) {
      oss << static_cast<int>(b) << ",";
    }
    return oss.str();
  }

  std::string radar_cmd_signature(
    const radar_decision_comm::msg::RadarDecisionCommand & msg) const
  {
    std::ostringstream oss;
    oss << static_cast<int>(msg.command_type) << "|"
        << static_cast<int>(msg.command_value) << "|"
        << static_cast<int>(msg.radar_cmd) << "|"
        << static_cast<int>(msg.password_cmd) << "|";
    for (auto b : msg.password) {
      oss << static_cast<int>(b) << ",";
    }
    oss << "|"
        << msg.note;
    return oss.str();
  }

  void push_pending_map(const radar_decision_comm::msg::MapPublishRequest & msg)
  {
    const auto sig = map_signature(msg);
    if (!map_state_.has_pending || prefer_latest_) {
      map_state_.pending_msg = msg;
      map_state_.pending_sig = sig;
      map_state_.has_pending = true;
    }
  }

  void push_pending_interaction(const radar_decision_comm::msg::RobotInteractionRequest & msg)
  {
    const auto bucket = interaction_bucket_key(msg);
    auto & state = interaction_states_[bucket];
    const auto sig = interaction_signature(msg);

    if (!state.has_pending || prefer_latest_) {
      state.pending_msg = msg;
      state.pending_sig = sig;
      state.has_pending = true;
    }
  }

  void push_pending_radar_cmd(const radar_decision_comm::msg::RadarDecisionCommand & msg)
  {
    const auto sig = radar_cmd_signature(msg);
    if (!radar_cmd_state_.has_pending || prefer_latest_) {
      radar_cmd_state_.pending_msg = msg;
      radar_cmd_state_.pending_sig = sig;
      radar_cmd_state_.has_pending = true;
    }
  }

  template<typename StateT>
  bool should_publish(
    const StateT & state,
    const TimePoint & now,
    double max_hz,
    int same_resend_ms) const
  {
    const bool changed = (!state.has_last || state.pending_sig != state.last_sig);

    if (enable_rate_limit_ && state.has_last) {
      const int min_interval_ms = hz_to_min_interval_ms(max_hz);
      if (!due(now, state.last_pub_tp, min_interval_ms)) {
        return false;
      }
    }

    if (deduplicate_ && !changed && state.has_last) {
      if (!due(now, state.last_pub_tp, same_resend_ms)) {
        return false;
      }
    }

    return true;
  }

  void on_timer()
  {
    if (!enabled_) {
      return;
    }

    const auto now = Clock::now();

    flush_map(now);
    flush_interactions(now);
    flush_radar_cmd(now);
  }

  void flush_map(const TimePoint & now)
  {
    if (!map_state_.has_pending) {
      return;
    }

    if (!should_publish(map_state_, now, map_max_hz_, same_map_resend_ms_)) {
      return;
    }

    map_pub_->publish(map_state_.pending_msg);
    map_state_.last_msg = map_state_.pending_msg;
    map_state_.last_sig = map_state_.pending_sig;
    map_state_.last_pub_tp = now;
    map_state_.has_last = true;
    map_state_.has_pending = false;

    static int log_count = 0;
    if ((log_count++ % 10) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "TxScheduler published MAP robots=%zu",
        map_state_.last_msg.robots.size());
    }
  }

  void flush_interactions(const TimePoint & now)
  {
    for (auto & kv : interaction_states_) {
      auto & state = kv.second;
      if (!state.has_pending) {
        continue;
      }

      if (!should_publish(state, now, interaction_max_hz_, same_interaction_resend_ms_)) {
        continue;
      }

      interaction_pub_->publish(state.pending_msg);
      state.last_msg = state.pending_msg;
      state.last_sig = state.pending_sig;
      state.last_pub_tp = now;
      state.has_last = true;
      state.has_pending = false;

      static int log_count = 0;
      if ((log_count++ % 10) == 0) {
        RCLCPP_INFO(
          this->get_logger(),
          "TxScheduler published INTERACTION bucket=%s data_cmd_id=0x%04x receiver=%u bytes=%zu",
          kv.first.c_str(),
          state.last_msg.data_cmd_id,
          state.last_msg.receiver_id,
          state.last_msg.user_data.size());
      }
    }
  }

  void flush_radar_cmd(const TimePoint & now)
  {
    if (!radar_cmd_state_.has_pending) {
      return;
    }

    if (!should_publish(radar_cmd_state_, now, radar_cmd_max_hz_, same_radar_cmd_resend_ms_)) {
      return;
    }

    radar_cmd_pub_->publish(radar_cmd_state_.pending_msg);
    radar_cmd_state_.last_msg = radar_cmd_state_.pending_msg;
    radar_cmd_state_.last_sig = radar_cmd_state_.pending_sig;
    radar_cmd_state_.last_pub_tp = now;
    radar_cmd_state_.has_last = true;
    radar_cmd_state_.has_pending = false;

    static int log_count = 0;
    if ((log_count++ % 10) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "TxScheduler published RADAR_CMD type=%u value=%u radar_cmd=%u password_cmd=%u",
        radar_cmd_state_.last_msg.command_type,
        radar_cmd_state_.last_msg.command_value,
        radar_cmd_state_.last_msg.radar_cmd,
        radar_cmd_state_.last_msg.password_cmd);
    }
  }

  bool enabled_{true};
  bool deduplicate_{true};
  bool prefer_latest_{true};
  bool enable_rate_limit_{true};

  double scheduler_rate_hz_{100.0};
  double map_max_hz_{5.0};
  double interaction_max_hz_{30.0};
  double radar_cmd_max_hz_{10.0};

  int same_map_resend_ms_{1000};
  int same_interaction_resend_ms_{1000};
  int same_radar_cmd_resend_ms_{1000};

  std::string input_map_request_topic_;
  std::string input_robot_interaction_topic_;
  std::string input_radar_command_topic_;
  std::string output_map_request_topic_;
  std::string output_robot_interaction_topic_;
  std::string output_radar_command_topic_;

  StreamState<radar_decision_comm::msg::MapPublishRequest> map_state_;
  std::unordered_map<std::string, StreamState<radar_decision_comm::msg::RobotInteractionRequest>> interaction_states_;
  StreamState<radar_decision_comm::msg::RadarDecisionCommand> radar_cmd_state_;

  rclcpp::Publisher<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr interaction_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_cmd_pub_;

  rclcpp::Subscription<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr interaction_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_cmd_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::TxSchedulerComponent)
