#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include "radar_decision_comm/codec/referee_decoder.hpp"
#include "radar_decision_comm/codec/referee_encoder.hpp"
#include "radar_decision_comm/codec/wireless_packet_parser.hpp"
#include "radar_decision_comm/io/serial_port.hpp"
#include "radar_decision_comm/msg/enemy_world_state.hpp"
#include "radar_decision_comm/msg/wireless_enemy_ammo.hpp"
#include "radar_decision_comm/msg/wireless_enemy_buffs.hpp"
#include "radar_decision_comm/msg/wireless_enemy_hp.hpp"
#include "radar_decision_comm/msg/wireless_enemy_macro_state.hpp"
#include "radar_decision_comm/msg/wireless_interference_key.hpp"

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

class RadarWirelessRxComponent : public rclcpp::Node
{
public:
  explicit RadarWirelessRxComponent(const rclcpp::NodeOptions & options)
  : Node("radar_wireless_rx", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    backend_type_ = this->declare_parameter<std::string>("backend_type", "serial");
    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    rx_mode_ = this->declare_parameter<std::string>("rx_mode", "mock");
    debug_raw_bytes_topic_ = this->declare_parameter<std::string>(
      "debug_raw_bytes_topic", "/radar/debug/wireless_raw_bytes");
    debug_mock_publish_ = this->declare_parameter<bool>("debug_mock_publish", true);
    debug_mock_rate_hz_ = this->declare_parameter<double>("debug_mock_rate_hz", 2.0);

    device_ = this->declare_parameter<std::string>("device", "/dev/ttyUSB1");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    timeout_ms_ = this->declare_parameter<int>("timeout_ms", 2);
    reconnect_interval_ms_ = this->declare_parameter<int>("reconnect_interval_ms", 1000);
    rx_thread_priority_ = this->declare_parameter<int>("rx_thread_priority", 80);
    enable_crc8_check_ = this->declare_parameter<bool>("enable_crc8_check", true);
    enable_crc16_check_ = this->declare_parameter<bool>("enable_crc16_check", true);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 100.0);

    parser_.set_my_color(my_color_);
    decoder_.configure(enable_crc8_check_, enable_crc16_check_);

    enemy_world_pub_ = this->create_publisher<radar_decision_comm::msg::EnemyWorldState>(
      "/radar/wireless/enemy_world_state", 10);
    enemy_hp_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyHp>(
      "/radar/wireless/enemy_hp", 10);
    enemy_ammo_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyAmmo>(
      "/radar/wireless/enemy_ammo", 10);
    macro_state_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyMacroState>(
      "/radar/wireless/enemy_macro_state", 10);
    enemy_buffs_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessEnemyBuffs>(
      "/radar/wireless/enemy_buffs", 10);
    interference_key_pub_ = this->create_publisher<radar_decision_comm::msg::WirelessInterferenceKey>(
      "/radar/wireless/interference_key", 10);

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
        [this]() { publish_mock_wireless_frames(); });
    }

    if (rx_mode_ == "serial" && enabled_) {
      start_serial_thread();
    }

    RCLCPP_INFO(
      get_logger(),
      "RadarWirelessRxComponent started. enabled=%s backend=%s mode=%s color=%s device=%s baudrate=%d crc8=%s crc16=%s mock=%s raw_topic=%s",
      enabled_ ? "true" : "false",
      backend_type_.c_str(),
      rx_mode_.c_str(),
      my_color_.c_str(),
      device_.c_str(),
      baudrate_,
      enable_crc8_check_ ? "true" : "false",
      enable_crc16_check_ ? "true" : "false",
      debug_mock_publish_ ? "true" : "false",
      debug_raw_bytes_topic_.c_str());
  }

  ~RadarWirelessRxComponent() override
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
            "Opened wireless serial port %s @ %d",
            device_.c_str(),
            baudrate_);
        } else {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            3000,
            "Failed to open wireless serial port %s @ %d",
            device_.c_str(),
            baudrate_);
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
        "Wireless parser warning for cmd=0x%04x: %s",
        cmd_id,
        result.warning.c_str());
    }

    if (result.has_enemy_world_state) {
      enemy_world_pub_->publish(result.enemy_world_state);
    }
    if (result.has_enemy_hp) {
      enemy_hp_pub_->publish(result.enemy_hp);
    }
    if (result.has_enemy_ammo) {
      enemy_ammo_pub_->publish(result.enemy_ammo);
    }
    if (result.has_macro_state) {
      macro_state_pub_->publish(result.macro_state);
    }
    if (result.has_enemy_buffs) {
      enemy_buffs_pub_->publish(result.enemy_buffs);
    }
    if (result.has_interference_key) {
      interference_key_pub_->publish(result.interference_key);
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
          "Wireless decoded frame from %s: seq=%u cmd=0x%04x payload=%zu raw=%zu",
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

  void publish_mock_wireless_frames()
  {
    std::vector<uint8_t> pos_payload(24, 0);
    auto put16 = [&pos_payload](std::size_t off, uint16_t v) {
      pos_payload[off] = static_cast<uint8_t>(v & 0xFFU);
      pos_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    };

    put16(0, 1850);
    put16(2, 620);
    put16(20, 2000);
    put16(22, 800);
    emit_mock_packet(0x0A01, pos_payload);

    std::vector<uint8_t> hp_payload(12, 0);
    auto put16hp = [&hp_payload](std::size_t off, uint16_t v) {
      hp_payload[off] = static_cast<uint8_t>(v & 0xFFU);
      hp_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    };
    put16hp(0, 100);
    put16hp(2, 80);
    put16hp(4, 60);
    put16hp(6, 50);
    put16hp(8, 40);
    put16hp(10, 120);
    emit_mock_packet(0x0A02, hp_payload);

    std::vector<uint8_t> ammo_payload(12, 0);
    auto put16ammo = [&ammo_payload](std::size_t off, uint16_t v) {
      ammo_payload[off] = static_cast<uint8_t>(v & 0xFFU);
      ammo_payload[off + 1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    };
    put16ammo(0, 30);
    put16ammo(2, 20);
    put16ammo(4, 10);
    put16ammo(6, 8);
    put16ammo(8, 6);
    put16ammo(10, 50);
    emit_mock_packet(0x0A03, ammo_payload);

    emit_mock_packet(0x0A04, std::vector<uint8_t>{1, 2, 3, 4});
    emit_mock_packet(0x0A05, std::vector<uint8_t>{5, 6, 7, 8});
    emit_mock_packet(0x0A06, std::vector<uint8_t>{9, 10, 11, 12});

    static int log_count = 0;
    if ((log_count++ % 5) == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Published mock wireless frames through full byte-stream pipeline.");
    }
  }

  bool enabled_{true};
  bool debug_mock_publish_{true};
  bool enable_crc8_check_{true};
  bool enable_crc16_check_{true};

  int baudrate_{115200};
  int timeout_ms_{2};
  int reconnect_interval_ms_{1000};
  int rx_thread_priority_{80};

  double debug_mock_rate_hz_{2.0};
  double publish_rate_hz_{100.0};

  std::string backend_type_{"serial"};
  std::string my_color_{"Red"};
  std::string rx_mode_{"mock"};
  std::string debug_raw_bytes_topic_{"/radar/debug/wireless_raw_bytes"};
  std::string device_;

  WirelessPacketParser parser_;
  RefereeDecoder decoder_;
  RefereeEncoder encoder_;
  SerialPort serial_;

  std::atomic_bool stop_rx_{false};
  std::thread serial_thread_;

  rclcpp::Publisher<radar_decision_comm::msg::EnemyWorldState>::SharedPtr enemy_world_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyHp>::SharedPtr enemy_hp_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyAmmo>::SharedPtr enemy_ammo_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyMacroState>::SharedPtr macro_state_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessEnemyBuffs>::SharedPtr enemy_buffs_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::WirelessInterferenceKey>::SharedPtr interference_key_pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr raw_bytes_sub_;
  rclcpp::TimerBase::SharedPtr mock_timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::RadarWirelessRxComponent)
