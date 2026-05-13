#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include "radar_decision_comm/codec/referee_decoder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"
#include "radar_decision_comm/codec/referee_packet_parser.hpp"
#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/io/serial_port.hpp"
#include "radar_decision_comm/msg/dart_status.hpp"
#include "radar_decision_comm/msg/radar_decision_sync.hpp"
#include "radar_decision_comm/msg/radar_mark_progress.hpp"
#include "radar_decision_comm/msg/referee_enemy_hp.hpp"
#include "radar_decision_comm/msg/referee_game_state.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
#include "radar_decision_comm/srv/reopen_port.hpp"

namespace radar_decision_comm
{

namespace
{

constexpr const char * kRedBold = "\033[1;31m";
constexpr const char * kColorReset = "\033[0m";

}  // namespace

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

class RefereeSerialRxComponent : public rclcpp::Node
{
public:
  explicit RefereeSerialRxComponent(const rclcpp::NodeOptions & options)
  : Node("referee_serial_rx", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    device_ = this->declare_parameter<std::string>("device", "/dev/ttyUSB0");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    timeout_ms_ = this->declare_parameter<int>("timeout_ms", 2);
    reconnect_interval_ms_ = this->declare_parameter<int>("reconnect_interval_ms", 1000);
    rx_thread_priority_ = this->declare_parameter<int>("rx_thread_priority", 80);
    enable_crc8_check_ = this->declare_parameter<bool>("enable_crc8_check", true);
    enable_crc16_check_ = this->declare_parameter<bool>("enable_crc16_check", true);
    publish_raw_debug_ = this->declare_parameter<bool>("publish_raw_debug", false);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 100.0);
    debug_mock_publish_ = this->declare_parameter<bool>("debug_mock_publish", true);
    debug_mock_rate_hz_ = this->declare_parameter<double>("debug_mock_rate_hz", 2.0);
    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    legacy_mark_progress_ = this->declare_parameter<bool>("legacy_mark_progress", false);
    legacy_dart_bits_ = this->declare_parameter<bool>("legacy_dart_bits", false);
    rx_mode_ = this->declare_parameter<std::string>("rx_mode", "serial");
    debug_raw_bytes_topic_ = this->declare_parameter<std::string>(
      "debug_raw_bytes_topic", "/radar/debug/referee_raw_bytes");
    output_identity_topic_ = this->declare_parameter<std::string>(
      "output_identity_topic", "/radar/referee/robot_identity");

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

    reopen_port_srv_ = this->create_service<radar_decision_comm::srv::ReopenPort>(
      "~/reopen_port",
      [this](
        const std::shared_ptr<radar_decision_comm::srv::ReopenPort::Request> request,
        std::shared_ptr<radar_decision_comm::srv::ReopenPort::Response> response)
      {
        if (rx_mode_ == "serial") {
          serial_.close();
          const bool ok = serial_.open(request->port_name, baudrate_, timeout_ms_);
          response->success = ok;
          response->message = ok ? "serial port reopened" : "failed to reopen serial port";
        } else {
          response->success = true;
          response->message = "placeholder reopen success for port: " + request->port_name;
        }
        RCLCPP_WARN(this->get_logger(), "ReopenPort called for %s", request->port_name.c_str());
      });

    raw_bytes_sub_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>(
      debug_raw_bytes_topic_, 20,
      [this](const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
      {
        if (rx_mode_ != "topic") {
          return;
        }
        ingest_bytes(msg->data, "topic");
      });

    if (rx_mode_ == "mock" && debug_mock_publish_) {
      mock_timer_ = this->create_wall_timer(
        hz_to_period_ms(debug_mock_rate_hz_),
        [this]() { publish_mock_referee_frames(); });
    }

    if (rx_mode_ == "serial" && enabled_) {
      start_serial_thread();
    }

    RCLCPP_INFO(
      get_logger(),
      "RefereeSerialRxComponent started. enabled=%s mode=%s device=%s baudrate=%d timeout_ms=%d crc8=%s crc16=%s mock=%s my_color=%s legacy_mark=%s legacy_dart=%s raw_topic=%s",
      enabled_ ? "true" : "false",
      rx_mode_.c_str(),
      device_.c_str(),
      baudrate_,
      timeout_ms_,
      enable_crc8_check_ ? "true" : "false",
      enable_crc16_check_ ? "true" : "false",
      debug_mock_publish_ ? "true" : "false",
      my_color_.c_str(),
      legacy_mark_progress_ ? "true" : "false",
      legacy_dart_bits_ ? "true" : "false",
      debug_raw_bytes_topic_.c_str());
  }

  ~RefereeSerialRxComponent() override
  {
    stop_serial_thread();
  }

private:
  void start_serial_thread()
  {
    stop_rx_.store(false);
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
            "Opened serial port %s @ %d",
            device_.c_str(),
            baudrate_);
        } else {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            3000,
            "%sFailed to open serial port %s @ %d%s",
            kRedBold,
            device_.c_str(),
            baudrate_,
            kColorReset);
          std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_interval_ms_));
          continue;
        }
      }

      auto bytes = serial_.read_some(512);
      if (!bytes.empty()) {
        ingest_bytes(bytes, "serial");
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
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

    std::vector<uint8_t> hp_payload(16, 0);
    auto put16 = [&hp_payload](std::size_t off, uint16_t v) {
      hp_payload[off] = static_cast<uint8_t>(v & 0xFFU);
      hp_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    };

    put16(0, 150);
    put16(2, 140);
    put16(4, 130);
    put16(6, 120);
    put16(8, 0);
    put16(10, 100);
    put16(12, 1500);
    put16(14, 5000);

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

    const uint16_t mark_bits = static_cast<uint16_t>((1U << 0U) | (1U << 5U) | (1U << 6U));
    std::vector<uint8_t> mark_payload{
      static_cast<uint8_t>(mark_bits & 0xFFU),
      static_cast<uint8_t>((mark_bits >> 8U) & 0xFFU)
    };
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
  int baudrate_{115200};
  int timeout_ms_{2};
  int reconnect_interval_ms_{1000};
  int rx_thread_priority_{80};
  bool enable_crc8_check_{true};
  bool enable_crc16_check_{true};
  bool publish_raw_debug_{false};
  bool debug_mock_publish_{true};
  bool legacy_mark_progress_{false};
  bool legacy_dart_bits_{false};
  double publish_rate_hz_{100.0};
  double debug_mock_rate_hz_{2.0};
  std::string device_;
  std::string my_color_{"Red"};
  std::string rx_mode_{"serial"};
  std::string debug_raw_bytes_topic_{"/radar/debug/referee_raw_bytes"};
  std::string output_identity_topic_{"/radar/referee/robot_identity"};

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
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_bytes_sub_;
  rclcpp::Service<radar_decision_comm::srv::ReopenPort>::SharedPtr reopen_port_srv_;
  rclcpp::TimerBase::SharedPtr mock_timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::RefereeSerialRxComponent)
