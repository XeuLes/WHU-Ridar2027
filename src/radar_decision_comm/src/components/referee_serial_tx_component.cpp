#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include "radar_decision_comm/codec/map_payload_builder.hpp"
#include "radar_decision_comm/codec/radar_command_payload_builder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"
#include "radar_decision_comm/common/interaction_cmd_ids.hpp"
#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/io/serial_port.hpp"
#include "radar_decision_comm/msg/map_publish_request.hpp"
#include "radar_decision_comm/msg/radar_decision_command.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
#include "radar_decision_comm/msg/robot_interaction_request.hpp"
#include "radar_decision_comm/srv/force_send.hpp"
#include "radar_decision_comm/srv/reopen_port.hpp"

namespace radar_decision_comm
{

class RefereeSerialTxComponent : public rclcpp::Node
{
public:
  explicit RefereeSerialTxComponent(const rclcpp::NodeOptions & options)
  : Node("referee_serial_tx", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    tx_mode_ = this->declare_parameter<std::string>("tx_mode", "topic");
    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    map_0305_layout_ = this->declare_parameter<std::string>("map_0305_layout", "legacy24");
    device_ = this->declare_parameter<std::string>("device", "/dev/ttyUSB0");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    timeout_ms_ = this->declare_parameter<int>("timeout_ms", 2);
    reconnect_interval_ms_ = this->declare_parameter<int>("reconnect_interval_ms", 1000);
    tx_thread_priority_ = this->declare_parameter<int>("tx_thread_priority", 80);
    enable_crc8_ = this->declare_parameter<bool>("enable_crc8", true);
    enable_crc16_ = this->declare_parameter<bool>("enable_crc16", true);
    send_queue_depth_ = this->declare_parameter<int>("send_queue_depth", 128);
    publish_debug_frames_ = this->declare_parameter<bool>("publish_debug_frames", true);
    debug_raw_frame_topic_ = this->declare_parameter<std::string>(
      "debug_raw_frame_topic", "/radar/debug/referee_tx_raw_bytes");
    use_referee_identity_color_ = this->declare_parameter<bool>("use_referee_identity_color", true);
    identity_topic_ = this->declare_parameter<std::string>(
      "identity_topic", "/radar/referee/robot_identity");

    input_map_request_topic_ = this->declare_parameter<std::string>(
      "input_map_request_topic", "/radar/tx/map_robot_data");
    input_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "input_robot_interaction_topic", "/radar/tx/robot_interaction");
    input_radar_command_topic_ = this->declare_parameter<std::string>(
      "input_radar_command_topic", "/radar/tx/radar_cmd");

    if (publish_debug_frames_) {
      raw_frame_pub_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>(
        debug_raw_frame_topic_, 20);
    }

    map_sub_ = this->create_subscription<radar_decision_comm::msg::MapPublishRequest>(
      input_map_request_topic_, 10,
      [this](const radar_decision_comm::msg::MapPublishRequest::SharedPtr msg)
      {
        if (!enabled_) return;
        handle_map_request(*msg);
      });

    interaction_sub_ = this->create_subscription<radar_decision_comm::msg::RobotInteractionRequest>(
      input_robot_interaction_topic_, 10,
      [this](const radar_decision_comm::msg::RobotInteractionRequest::SharedPtr msg)
      {
        if (!enabled_) return;
        handle_robot_interaction_request(*msg);
      });

    radar_cmd_sub_ = this->create_subscription<radar_decision_comm::msg::RadarDecisionCommand>(
      input_radar_command_topic_, 10,
      [this](const radar_decision_comm::msg::RadarDecisionCommand::SharedPtr msg)
      {
        if (!enabled_) return;
        handle_radar_command(*msg);
      });

    identity_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      identity_topic_, 10,
      [this](const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
      {
        if (!use_referee_identity_color_ || !msg || !msg->valid) {
          return;
        }
        const std::string next_color = team_identity::my_color_from_self_color(msg->self_color);
        if (next_color != my_color_) {
          my_color_ = next_color;
          RCLCPP_INFO(
            this->get_logger(),
            "RefereeSerialTxComponent switched my_color=%s from referee robot_id=%u",
            my_color_.c_str(),
            msg->robot_id);
        }
      });

    reopen_port_srv_ = this->create_service<radar_decision_comm::srv::ReopenPort>(
      "~/reopen_port",
      [this](
        const std::shared_ptr<radar_decision_comm::srv::ReopenPort::Request> request,
        std::shared_ptr<radar_decision_comm::srv::ReopenPort::Response> response)
      {
        if (tx_mode_ == "serial") {
          serial_.close();
          const bool ok = serial_.open(request->port_name, baudrate_, timeout_ms_);
          response->success = ok;
          response->message = ok ? "tx serial port reopened" : "failed to reopen tx serial port";
        } else {
          response->success = true;
          response->message = "tx mode is not serial; reopen skipped";
        }
      });

    force_send_srv_ = this->create_service<radar_decision_comm::srv::ForceSend>(
      "~/force_send",
      [this](
        const std::shared_ptr<radar_decision_comm::srv::ForceSend::Request> request,
        std::shared_ptr<radar_decision_comm::srv::ForceSend::Response> response)
      {
        (void)request;
        if (last_frame_.empty()) {
          response->success = false;
          response->message = "no cached frame to send";
          return;
        }
        const bool ok = emit_frame(last_frame_, last_cmd_id_, "force_send");
        response->success = ok;
        response->message = ok ? "cached frame sent" : "cached frame send failed";
      });

    RCLCPP_INFO(
      get_logger(),
      "RefereeSerialTxComponent started. enabled=%s mode=%s my_color=%s layout=%s",
      enabled_ ? "true" : "false",
      tx_mode_.c_str(),
      my_color_.c_str(),
      map_0305_layout_.c_str());
  }

private:
  uint16_t default_radar_sender_id() const
  {
    return (my_color_ == "Red") ? 9U : 109U;
  }

  static void append_u16_le(std::vector<uint8_t> & out, uint16_t value)
  {
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
  }

  std::vector<uint8_t> build_0301_payload(
    uint16_t data_cmd_id,
    uint16_t sender_id,
    uint16_t receiver_id,
    const std::vector<uint8_t> & user_data) const
  {
    std::vector<uint8_t> payload;
    payload.reserve(6 + user_data.size());

    append_u16_le(payload, data_cmd_id);
    append_u16_le(payload, sender_id);
    append_u16_le(payload, receiver_id);
    payload.insert(payload.end(), user_data.begin(), user_data.end());

    return payload;
  }

  std::vector<uint8_t> build_0301_payload(
    const radar_decision_comm::msg::RobotInteractionRequest & msg) const
  {
    const uint16_t sender_id = (msg.sender_id == 0) ? default_radar_sender_id() : msg.sender_id;
    return build_0301_payload(msg.data_cmd_id, sender_id, msg.receiver_id, msg.user_data);
  }

  bool emit_frame(const std::vector<uint8_t> & frame, uint16_t cmd_id, const char * reason)
  {
    bool ok = false;

    if (tx_mode_ == "topic") {
      if (raw_frame_pub_) {
        std_msgs::msg::UInt8MultiArray raw_msg;
        raw_msg.data = frame;
        raw_frame_pub_->publish(raw_msg);
      }
      ok = true;
    } else if (tx_mode_ == "serial") {
      if (!serial_.is_open()) {
        const bool opened = serial_.open(device_, baudrate_, timeout_ms_);
        if (!opened) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            3000,
            "Failed to open tx serial port %s @ %d",
            device_.c_str(),
            baudrate_);
          return false;
        }
      }

      const auto written = serial_.write(frame);
      ok = (written == frame.size());

      if (publish_debug_frames_ && raw_frame_pub_) {
        std_msgs::msg::UInt8MultiArray raw_msg;
        raw_msg.data = frame;
        raw_frame_pub_->publish(raw_msg);
      }
    }

    static int log_count = 0;
    if ((log_count++ % 10) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "TX %s cmd=0x%04x frame_bytes=%zu ok=%s",
        reason,
        cmd_id,
        frame.size(),
        ok ? "true" : "false");
    }

    last_cmd_id_ = cmd_id;
    return ok;
  }

  void handle_map_request(const radar_decision_comm::msg::MapPublishRequest & msg)
  {
    const auto build_result = MapPayloadBuilder::build_payload(msg, my_color_, map_0305_layout_);

    if (build_result.used_fallback) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Unsupported map_0305_layout=%s, fallback to %s",
        map_0305_layout_.c_str(),
        build_result.layout_used.c_str());
    }

    auto frame = encoder_.encode_frame(referee_cmd_ids::kMapRadarData, build_result.payload);
    last_frame_ = frame;

    const bool ok = emit_frame(frame, referee_cmd_ids::kMapRadarData, "map_0305");
    if (!ok) {
      RCLCPP_WARN(this->get_logger(), "Failed to transmit 0x0305 frame.");
    }
  }

  void handle_robot_interaction_request(
    const radar_decision_comm::msg::RobotInteractionRequest & msg)
  {
    if (msg.receiver_id == 0) {
      RCLCPP_WARN(this->get_logger(), "Drop 0x0301: receiver_id is 0.");
      return;
    }

    if (msg.user_data.size() > 112) {
      RCLCPP_WARN(
        this->get_logger(),
        "Drop 0x0301: user_data too large (%zu > 112).",
        msg.user_data.size());
      return;
    }

    const auto payload = build_0301_payload(msg);
    auto frame = encoder_.encode_frame(referee_cmd_ids::kRobotInteraction, payload);
    last_frame_ = frame;

    const bool ok = emit_frame(frame, referee_cmd_ids::kRobotInteraction, "robot_interaction_0301");
    if (!ok) {
      RCLCPP_WARN(this->get_logger(), "Failed to transmit 0x0301 frame.");
    }
  }

  void handle_radar_command(const radar_decision_comm::msg::RadarDecisionCommand & msg)
  {
    if (msg.command_type != 1) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Unsupported RadarDecisionCommand type=%u",
        msg.command_type);
      return;
    }

    const auto user_data =
      RadarCommandPayloadBuilder::build_radar_double_effect_times_payload(msg.command_value);

    const auto payload = build_0301_payload(
      interaction_cmd_ids::kRadarDecisionInfo,
      default_radar_sender_id(),
      interaction_cmd_ids::kRadarDecisionReceiver,
      user_data);

    auto frame = encoder_.encode_frame(referee_cmd_ids::kRobotInteraction, payload);
    last_frame_ = frame;

    const bool ok = emit_frame(frame, referee_cmd_ids::kRobotInteraction, "radar_decision_0121");
    if (!ok) {
      RCLCPP_WARN(this->get_logger(), "Failed to transmit radar decision 0x0121 frame.");
    }
  }

  bool enabled_{true};
  bool enable_crc8_{true};
  bool enable_crc16_{true};
  bool publish_debug_frames_{true};
  bool use_referee_identity_color_{true};

  int baudrate_{115200};
  int timeout_ms_{2};
  int reconnect_interval_ms_{1000};
  int tx_thread_priority_{80};
  int send_queue_depth_{128};

  uint16_t last_cmd_id_{0};
  std::vector<uint8_t> last_frame_;

  std::string tx_mode_{"topic"};
  std::string my_color_{"Red"};
  std::string map_0305_layout_{"legacy24"};
  std::string device_;
  std::string debug_raw_frame_topic_;
  std::string identity_topic_;
  std::string input_map_request_topic_;
  std::string input_robot_interaction_topic_;
  std::string input_radar_command_topic_;

  RefereeEncoder encoder_;
  SerialPort serial_;

  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_frame_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr interaction_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_cmd_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr identity_sub_;
  rclcpp::Service<radar_decision_comm::srv::ReopenPort>::SharedPtr reopen_port_srv_;
  rclcpp::Service<radar_decision_comm::srv::ForceSend>::SharedPtr force_send_srv_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::RefereeSerialTxComponent)
