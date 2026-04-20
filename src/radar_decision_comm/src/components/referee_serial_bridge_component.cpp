#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include "radar_decision_comm/codec/map_payload_builder.hpp"
#include "radar_decision_comm/codec/radar_command_payload_builder.hpp"
#include "radar_decision_comm/codec/referee_decoder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"
#include "radar_decision_comm/codec/referee_packet_parser.hpp"
#include "radar_decision_comm/common/interaction_cmd_ids.hpp"
#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/io/serial_port.hpp"
#include "radar_decision_comm/msg/dart_status.hpp"
#include "radar_decision_comm/msg/map_publish_request.hpp"
#include "radar_decision_comm/msg/radar_decision_command.hpp"
#include "radar_decision_comm/msg/radar_decision_sync.hpp"
#include "radar_decision_comm/msg/radar_mark_progress.hpp"
#include "radar_decision_comm/msg/referee_enemy_hp.hpp"
#include "radar_decision_comm/msg/referee_game_state.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
#include "radar_decision_comm/msg/robot_interaction_request.hpp"
#include "radar_decision_comm/srv/force_send.hpp"
#include "radar_decision_comm/srv/reopen_port.hpp"

namespace radar_decision_comm
{

namespace
{

std::chrono::milliseconds hz_to_period_ms(double hz)
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

}  // namespace

class RefereeSerialBridgeComponent : public rclcpp::Node
{
public:
  explicit RefereeSerialBridgeComponent(const rclcpp::NodeOptions & options)
  : Node("referee_serial_bridge", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    device_ = this->declare_parameter<std::string>("device", "/dev/ttyUSB0");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    timeout_ms_ = this->declare_parameter<int>("timeout_ms", 2);
    reconnect_interval_ms_ = this->declare_parameter<int>("reconnect_interval_ms", 1000);
    rx_thread_priority_ = this->declare_parameter<int>("rx_thread_priority", 80);
    tx_thread_priority_ = this->declare_parameter<int>("tx_thread_priority", 80);
    enable_crc8_check_ = this->declare_parameter<bool>("enable_crc8_check", true);
    enable_crc16_check_ = this->declare_parameter<bool>("enable_crc16_check", true);
    enable_crc8_ = this->declare_parameter<bool>("enable_crc8", true);
    enable_crc16_ = this->declare_parameter<bool>("enable_crc16", true);
    publish_raw_debug_ = this->declare_parameter<bool>("publish_raw_debug", false);
    publish_debug_frames_ = this->declare_parameter<bool>("publish_debug_frames", true);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 100.0);
    debug_mock_publish_ = this->declare_parameter<bool>("debug_mock_publish", true);
    debug_mock_rate_hz_ = this->declare_parameter<double>("debug_mock_rate_hz", 2.0);
    legacy_mark_progress_ = this->declare_parameter<bool>("legacy_mark_progress", true);
    legacy_dart_bits_ = this->declare_parameter<bool>("legacy_dart_bits", true);
    use_referee_identity_color_ = this->declare_parameter<bool>("use_referee_identity_color", true);
    send_queue_depth_ = this->declare_parameter<int>("send_queue_depth", 128);

    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    rx_mode_ = this->declare_parameter<std::string>("rx_mode", "serial");
    tx_mode_ = this->declare_parameter<std::string>("tx_mode", "serial");
    map_0305_layout_ = this->declare_parameter<std::string>("map_0305_layout", "legacy24");

    debug_raw_bytes_topic_ = this->declare_parameter<std::string>(
      "debug_raw_bytes_topic", "/radar/debug/referee_raw_bytes");
    debug_raw_frame_topic_ = this->declare_parameter<std::string>(
      "debug_raw_frame_topic", "/radar/debug/referee_tx_raw_bytes");
    output_identity_topic_ = this->declare_parameter<std::string>(
      "output_identity_topic", "/radar/referee/robot_identity");
    identity_topic_ = this->declare_parameter<std::string>(
      "identity_topic", "/radar/referee/robot_identity");

    input_map_request_topic_ = this->declare_parameter<std::string>(
      "input_map_request_topic", "/radar/tx/map_robot_data");
    input_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "input_robot_interaction_topic", "/radar/tx/robot_interaction");
    input_radar_command_topic_ = this->declare_parameter<std::string>(
      "input_radar_command_topic", "/radar/tx/radar_cmd");

    parser_.set_my_color(my_color_);
    parser_.set_legacy_mark_progress(legacy_mark_progress_);
    parser_.set_legacy_dart_bits(legacy_dart_bits_);
    decoder_.configure(enable_crc8_check_, enable_crc16_check_);

    game_status_pub_ = this->create_publisher<radar_decision_comm::msg::RefereeGameState>(
      "/radar/referee/game_status", 10);
    enemy_hp_pub_ = this->create_publisher<radar_decision_comm::msg::RefereeEnemyHp>(
      "/radar/referee/enemy_hp", 10);
    dart_status_pub_ = this->create_publisher<radar_decision_comm::msg::DartStatus>(
      "/radar/referee/dart_status", 10);
    mark_progress_pub_ = this->create_publisher<radar_decision_comm::msg::RadarMarkProgress>(
      "/radar/referee/mark_progress", 10);
    radar_sync_pub_ = this->create_publisher<radar_decision_comm::msg::RadarDecisionSync>(
      "/radar/referee/radar_decision_sync", 10);
    robot_identity_pub_ = this->create_publisher<radar_decision_comm::msg::RefereeRobotIdentity>(
      output_identity_topic_, 10);

    if (publish_raw_debug_) {
      raw_bytes_pub_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>(
        debug_raw_bytes_topic_, 20);
    }

    if (publish_debug_frames_ || tx_mode_ == "topic") {
      raw_frame_pub_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>(
        debug_raw_frame_topic_, 20);
    }

    raw_bytes_sub_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>(
      debug_raw_bytes_topic_, 20,
      [this](const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
      {
        if (!enabled_ || rx_mode_ != "topic" || !msg) {
          return;
        }
        ingest_bytes(msg->data, "topic");
      });

    map_sub_ = this->create_subscription<radar_decision_comm::msg::MapPublishRequest>(
      input_map_request_topic_, 10,
      [this](const radar_decision_comm::msg::MapPublishRequest::SharedPtr msg)
      {
        if (!enabled_ || !msg) {
          return;
        }
        handle_map_request(*msg);
      });

    interaction_sub_ = this->create_subscription<radar_decision_comm::msg::RobotInteractionRequest>(
      input_robot_interaction_topic_, 10,
      [this](const radar_decision_comm::msg::RobotInteractionRequest::SharedPtr msg)
      {
        if (!enabled_ || !msg) {
          return;
        }
        handle_robot_interaction_request(*msg);
      });

    radar_cmd_sub_ = this->create_subscription<radar_decision_comm::msg::RadarDecisionCommand>(
      input_radar_command_topic_, 10,
      [this](const radar_decision_comm::msg::RadarDecisionCommand::SharedPtr msg)
      {
        if (!enabled_ || !msg) {
          return;
        }
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
          parser_.set_my_color(my_color_);
          RCLCPP_INFO(
            this->get_logger(),
            "RefereeSerialBridgeComponent switched my_color=%s from referee robot_id=%u",
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
        if (!serial_transport_active()) {
          response->success = true;
          response->message = "serial transport inactive; reopen skipped";
          return;
        }

        stop_serial_thread();

        if (!request->port_name.empty()) {
          device_ = request->port_name;
        }

        bool ok = serial_.open(device_, baudrate_, timeout_ms_);
        if (!ok && rx_mode_ == "serial") {
          start_serial_thread();
          response->success = false;
          response->message = "failed to reopen serial port";
          return;
        }

        if (rx_mode_ == "serial") {
          start_serial_thread();
        }

        response->success = ok;
        response->message = ok ? "serial port reopened" : "failed to reopen serial port";
        RCLCPP_WARN(this->get_logger(), "ReopenPort called for %s", device_.c_str());
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

    if (rx_mode_ == "mock" && debug_mock_publish_) {
      mock_timer_ = this->create_wall_timer(
        hz_to_period_ms(debug_mock_rate_hz_),
        [this]() { publish_mock_referee_frames(); });
    }

    if (enabled_ && rx_mode_ == "serial") {
      start_serial_thread();
    }

    RCLCPP_INFO(
      this->get_logger(),
      "RefereeSerialBridgeComponent started. enabled=%s rx_mode=%s tx_mode=%s device=%s baudrate=%d timeout_ms=%d rx_crc8=%s rx_crc16=%s tx_crc8=%s tx_crc16=%s my_color=%s raw_rx_topic=%s raw_tx_topic=%s",
      enabled_ ? "true" : "false",
      rx_mode_.c_str(),
      tx_mode_.c_str(),
      device_.c_str(),
      baudrate_,
      timeout_ms_,
      enable_crc8_check_ ? "true" : "false",
      enable_crc16_check_ ? "true" : "false",
      enable_crc8_ ? "true" : "false",
      enable_crc16_ ? "true" : "false",
      my_color_.c_str(),
      debug_raw_bytes_topic_.c_str(),
      debug_raw_frame_topic_.c_str());
  }

  ~RefereeSerialBridgeComponent() override
  {
    stop_serial_thread();
  }

private:
  bool serial_transport_active() const
  {
    return enabled_ && (rx_mode_ == "serial" || tx_mode_ == "serial");
  }

  void start_serial_thread()
  {
    stop_rx_.store(false);

    if (serial_thread_.joinable()) {
      return;
    }

    serial_thread_ = std::thread([this]() { serial_loop(); });
  }

  void stop_serial_thread()
  {
    stop_rx_.store(true);
    if (serial_thread_.joinable()) {
      serial_thread_.join();
    }
    serial_.close();
  }

  void serial_loop()
  {
    while (!stop_rx_.load()) {
      if (!serial_.is_open()) {
        const bool ok = serial_.open(device_, baudrate_, timeout_ms_);
        if (ok) {
          RCLCPP_INFO(
            this->get_logger(),
            "Opened referee serial port %s @ %d",
            device_.c_str(),
            baudrate_);
        } else {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            3000,
            "Failed to open referee serial port %s @ %d",
            device_.c_str(),
            baudrate_);
          std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_interval_ms_));
          continue;
        }
      }

      auto bytes = serial_.read_some(512);
      if (!bytes.empty()) {
        publish_raw_bytes(bytes);
        ingest_bytes(bytes, "serial");
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void publish_raw_bytes(const std::vector<uint8_t> & bytes)
  {
    if (!publish_raw_debug_ || !raw_bytes_pub_ || bytes.empty()) {
      return;
    }
    std_msgs::msg::UInt8MultiArray raw_msg;
    raw_msg.data = bytes;
    raw_bytes_pub_->publish(raw_msg);
  }

  void publish_parsed(uint16_t cmd_id, const std::vector<uint8_t> & payload)
  {
    const auto result = parser_.parse(cmd_id, payload, this->now());

    if (!result.warning.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Parser warning for cmd=0x%04x: %s",
        cmd_id,
        result.warning.c_str());
    }

    if (result.has_game_state) {
      game_status_pub_->publish(result.game_state);
    }
    if (result.has_enemy_hp) {
      enemy_hp_pub_->publish(result.enemy_hp);
    }
    if (result.has_dart_status) {
      dart_status_pub_->publish(result.dart_status);
    }
    if (result.has_mark_progress) {
      mark_progress_pub_->publish(result.mark_progress);
    }
    if (result.has_radar_sync) {
      radar_sync_pub_->publish(result.radar_sync);
    }
    if (result.has_robot_identity) {
      robot_identity_pub_->publish(result.robot_identity);

      const std::string derived_color =
        team_identity::my_color_from_self_color(result.robot_identity.self_color);
      if (derived_color != my_color_) {
        my_color_ = derived_color;
        parser_.set_my_color(my_color_);
        RCLCPP_INFO(
          this->get_logger(),
          "Referee identity updated from 0x0201: robot_id=%u my_color=%s",
          result.robot_identity.robot_id,
          my_color_.c_str());
      }
    }
  }

  void ingest_bytes(const std::vector<uint8_t> & bytes, const std::string & source_tag)
  {
    decoder_.push_bytes(bytes);

    while (true) {
      auto decoded = decoder_.try_decode();
      if (!decoded.has_value()) {
        break;
      }

      publish_parsed(decoded->cmd_id, decoded->payload);

      static int log_count = 0;
      if ((log_count++ % 10) == 0) {
        RCLCPP_INFO(
          this->get_logger(),
          "Decoded frame from %s: seq=%u cmd=0x%04x payload=%zu raw=%zu",
          source_tag.c_str(),
          decoded->seq,
          decoded->cmd_id,
          decoded->payload.size(),
          decoded->raw_frame.size());
      }
    }
  }

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
    } else {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Unsupported tx_mode=%s",
        tx_mode_.c_str());
      return false;
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

  void emit_mock_packet(uint16_t cmd_id, const std::vector<uint8_t> & payload)
  {
    auto frame = encoder_.encode_frame(cmd_id, payload);
    ingest_bytes(frame, "mock");
  }

  void publish_mock_referee_frames()
  {
    const uint8_t game_type = 2;
    const uint8_t game_progress = 4;
    const uint16_t stage_remain = 320;
    const uint64_t unix_time = static_cast<uint64_t>(this->now().seconds());

    std::vector<uint8_t> game_payload(11, 0);
    game_payload[0] = static_cast<uint8_t>((game_progress << 4U) | game_type);
    game_payload[1] = static_cast<uint8_t>(stage_remain & 0xFFU);
    game_payload[2] = static_cast<uint8_t>((stage_remain >> 8U) & 0xFFU);
    for (int i = 0; i < 8; ++i) {
      game_payload[3 + i] = static_cast<uint8_t>((unix_time >> (8U * i)) & 0xFFU);
    }
    emit_mock_packet(0x0001, game_payload);

    std::vector<uint8_t> hp_payload(32, 0);
    auto put16 = [&hp_payload](std::size_t off, uint16_t v) {
      hp_payload[off] = static_cast<uint8_t>(v & 0xFFU);
      hp_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    };

    put16(0, 150);
    put16(2, 140);
    put16(4, 130);
    put16(6, 120);
    put16(8, 110);
    put16(10, 100);

    put16(16, 100);
    put16(18, 80);
    put16(20, 60);
    put16(22, 50);
    put16(24, 40);
    put16(26, 120);

    emit_mock_packet(0x0003, hp_payload);

    uint16_t dart_info = 0;
    if (legacy_dart_bits_) {
      dart_info = static_cast<uint16_t>((2U) | (1U << 2U) | (3U << 5U));
    } else {
      dart_info = static_cast<uint16_t>((2U) | (1U << 3U) | (3U << 6U));
    }

    std::vector<uint8_t> dart_payload(3, 0);
    dart_payload[0] = 12;
    dart_payload[1] = static_cast<uint8_t>(dart_info & 0xFFU);
    dart_payload[2] = static_cast<uint8_t>((dart_info >> 8U) & 0xFFU);
    emit_mock_packet(0x0105, dart_payload);

    std::vector<uint8_t> mark_payload{100, 45, 20, 0, 0, 80};
    emit_mock_packet(0x020C, mark_payload);

    std::vector<uint8_t> sync_payload{
      static_cast<uint8_t>((2U) | (0U << 2U) | (1U << 3U) | (1U << 5U))
    };
    emit_mock_packet(0x020E, sync_payload);

    static int log_count = 0;
    if ((log_count++ % 5) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Published mock referee frames through full byte-stream pipeline.");
    }
  }

  bool enabled_{true};
  bool enable_crc8_check_{true};
  bool enable_crc16_check_{true};
  bool enable_crc8_{true};
  bool enable_crc16_{true};
  bool publish_raw_debug_{false};
  bool publish_debug_frames_{true};
  bool debug_mock_publish_{true};
  bool legacy_mark_progress_{true};
  bool legacy_dart_bits_{true};
  bool use_referee_identity_color_{true};

  int baudrate_{115200};
  int timeout_ms_{2};
  int reconnect_interval_ms_{1000};
  int rx_thread_priority_{80};
  int tx_thread_priority_{80};
  int send_queue_depth_{128};

  double publish_rate_hz_{100.0};
  double debug_mock_rate_hz_{2.0};

  uint16_t last_cmd_id_{0};
  std::vector<uint8_t> last_frame_;

  std::string device_;
  std::string my_color_{"Red"};
  std::string rx_mode_{"serial"};
  std::string tx_mode_{"serial"};
  std::string map_0305_layout_{"legacy24"};
  std::string debug_raw_bytes_topic_{"/radar/debug/referee_raw_bytes"};
  std::string debug_raw_frame_topic_{"/radar/debug/referee_tx_raw_bytes"};
  std::string output_identity_topic_{"/radar/referee/robot_identity"};
  std::string identity_topic_{"/radar/referee/robot_identity"};
  std::string input_map_request_topic_;
  std::string input_robot_interaction_topic_;
  std::string input_radar_command_topic_;

  RefereePacketParser parser_;
  RefereeDecoder decoder_;
  RefereeEncoder encoder_;
  SerialPort serial_;

  std::atomic_bool stop_rx_{false};
  std::thread serial_thread_;

  rclcpp::Publisher<radar_decision_comm::msg::RefereeGameState>::SharedPtr game_status_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RefereeEnemyHp>::SharedPtr enemy_hp_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::DartStatus>::SharedPtr dart_status_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RadarMarkProgress>::SharedPtr mark_progress_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RadarDecisionSync>::SharedPtr radar_sync_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr robot_identity_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_bytes_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_frame_pub_;

  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_bytes_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr interaction_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_cmd_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr identity_sub_;

  rclcpp::Service<radar_decision_comm::srv::ReopenPort>::SharedPtr reopen_port_srv_;
  rclcpp::Service<radar_decision_comm::srv::ForceSend>::SharedPtr force_send_srv_;
  rclcpp::TimerBase::SharedPtr mock_timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::RefereeSerialBridgeComponent)
