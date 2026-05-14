#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <radar_decision_comm/msg/dart_status.hpp>
#include <radar_decision_comm/msg/map_publish_request.hpp>
#include <radar_decision_comm/msg/radar_decision_command.hpp>
#include <radar_decision_comm/msg/radar_decision_sync.hpp>
#include <radar_decision_comm/msg/radar_mark_progress.hpp>
#include <radar_decision_comm/msg/referee_enemy_hp.hpp>
#include <radar_decision_comm/msg/referee_game_state.hpp>
#include <radar_decision_comm/msg/referee_robot_identity.hpp>
#include <radar_decision_comm/msg/robot_interaction_request.hpp>
#include <radar_decision_comm/msg/wireless_enemy_ammo.hpp>
#include <radar_decision_comm/msg/wireless_enemy_buffs.hpp>
#include <radar_decision_comm/msg/wireless_enemy_hp.hpp>
#include <radar_decision_comm/msg/wireless_enemy_macro_state.hpp>
#include <radar_decision_comm/msg/wireless_interference_key.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

namespace debug_map
{

namespace
{

std::string boolText(bool value)
{
  return value ? "Y" : "N";
}

std::string colorText(int self_color)
{
  if (self_color == 0) {
    return "Blue";
  }
  if (self_color == 2) {
    return "Red";
  }
  return "Unknown";
}

std::string hexU16(uint16_t value)
{
  std::ostringstream oss;
  oss << "0x" << std::uppercase << std::hex << std::setw(4)
      << std::setfill('0') << static_cast<int>(value);
  return oss.str();
}

std::string keyString(const std::array<uint8_t, 6> & key)
{
  return std::string(key.begin(), key.end());
}

std::string truncateText(const std::string & text, std::size_t max_len)
{
  if (text.size() <= max_len) {
    return text;
  }
  if (max_len <= 3) {
    return text.substr(0, max_len);
  }
  return text.substr(0, max_len - 3) + "...";
}

std::string findJsonField(const std::string & json, const std::string & key)
{
  const std::string quoted_key = "\"" + key + "\"";
  const auto key_pos = json.find(quoted_key);
  if (key_pos == std::string::npos) {
    return "";
  }
  const auto colon_pos = json.find(':', key_pos + quoted_key.size());
  if (colon_pos == std::string::npos) {
    return "";
  }

  auto value_pos = colon_pos + 1;
  while (value_pos < json.size() && std::isspace(static_cast<unsigned char>(json[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= json.size()) {
    return "";
  }

  if (json[value_pos] == '"') {
    const auto end_pos = json.find('"', value_pos + 1);
    if (end_pos == std::string::npos) {
      return "";
    }
    return json.substr(value_pos + 1, end_pos - value_pos - 1);
  }

  auto end_pos = value_pos;
  while (end_pos < json.size() && json[end_pos] != ',' && json[end_pos] != '}') {
    ++end_pos;
  }
  return json.substr(value_pos, end_pos - value_pos);
}

std::string summarizePduJson(const std::string & json)
{
  const auto source = findJsonField(json, "source_type");
  const auto cmd_id = findJsonField(json, "cmd_id");
  const auto cmd_name = findJsonField(json, "cmd_name");
  const auto data_len = findJsonField(json, "data_len");

  std::ostringstream oss;
  if (!source.empty()) {
    oss << "src=" << source << " ";
  }
  if (!cmd_id.empty()) {
    try {
      oss << "cmd=" << hexU16(static_cast<uint16_t>(std::stoul(cmd_id))) << " ";
    } catch (...) {
      oss << "cmd=" << cmd_id << " ";
    }
  }
  if (!cmd_name.empty()) {
    oss << cmd_name << " ";
  }
  if (!data_len.empty()) {
    oss << "data=" << data_len << "B";
  }

  const auto text = oss.str();
  return text.empty() ? ("json bytes=" + std::to_string(json.size())) : text;
}

void drawText(cv::Mat & canvas,
              const std::string & text,
              int x,
              int y,
              double scale,
              const cv::Scalar & color,
              int thickness = 1)
{
  cv::putText(canvas, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
              scale, color, thickness, cv::LINE_AA);
}

}  // namespace

class RefereeStatusWindowNode : public rclcpp::Node
{
public:
  explicit RefereeStatusWindowNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("referee_status_window_node", options)
  {
    window_name_ = declare_parameter<std::string>("window_name", "Referee Status");
    refresh_hz_ = declare_parameter<double>("refresh_hz", 8.0);
    stale_timeout_s_ = declare_parameter<double>("stale_timeout_s", 2.0);
    rate_window_s_ = declare_parameter<double>("rate_window_s", 5.0);

    const auto game_status_topic =
      declare_parameter<std::string>("game_status_topic", "/radar/referee/game_status");
    const auto identity_topic =
      declare_parameter<std::string>("identity_topic", "/radar/referee/robot_identity");
    const auto enemy_hp_topic =
      declare_parameter<std::string>("enemy_hp_topic", "/radar/referee/enemy_hp");
    const auto dart_status_topic =
      declare_parameter<std::string>("dart_status_topic", "/radar/referee/dart_status");
    const auto mark_progress_topic =
      declare_parameter<std::string>("mark_progress_topic", "/radar/referee/mark_progress");
    const auto radar_sync_topic =
      declare_parameter<std::string>("radar_sync_topic", "/radar/referee/radar_decision_sync");
    const auto raw_rx_topic =
      declare_parameter<std::string>("raw_rx_topic", "/radar/debug/referee_raw_bytes");
    const auto raw_tx_topic =
      declare_parameter<std::string>("raw_tx_topic", "/radar/debug/referee_tx_raw_bytes");

    const auto pdu_all_topic =
      declare_parameter<std::string>("pdu_all_topic", "/radar/all_pdu");
    const auto pdu_info_topic =
      declare_parameter<std::string>("pdu_info_topic", "/radar/info_pdu");
    const auto pdu_jam_topic =
      declare_parameter<std::string>("pdu_jam_topic", "/radar/jam_pdu");
    const auto wireless_hp_topic =
      declare_parameter<std::string>("wireless_hp_topic", "/radar/wireless/enemy_hp");
    const auto wireless_ammo_topic =
      declare_parameter<std::string>("wireless_ammo_topic", "/radar/wireless/enemy_ammo");
    const auto wireless_macro_topic =
      declare_parameter<std::string>("wireless_macro_topic", "/radar/wireless/enemy_macro_state");
    const auto wireless_buffs_topic =
      declare_parameter<std::string>("wireless_buffs_topic", "/radar/wireless/enemy_buffs");
    const auto wireless_key_topic =
      declare_parameter<std::string>("wireless_key_topic", "/radar/wireless/interference_key");

    const auto tx_map_topic =
      declare_parameter<std::string>("tx_map_topic", "/radar/tx/map_robot_data");
    const auto tx_interaction_topic =
      declare_parameter<std::string>("tx_interaction_topic", "/radar/tx/robot_interaction");
    const auto tx_radar_cmd_topic =
      declare_parameter<std::string>("tx_radar_cmd_topic", "/radar/tx/radar_cmd");

    addTypedSubscription<radar_decision_comm::msg::RefereeGameState>(
      "RX", "0x0001 Game", game_status_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "type=" << static_cast<int>(msg.game_type)
            << " progress=" << static_cast<int>(msg.game_progress)
            << " remain=" << msg.stage_remain_time << "s";
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      "RX", "0x0201 Identity", identity_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "valid=" << boolText(msg.valid)
            << " id=" << msg.robot_id
            << " color=" << colorText(msg.self_color);
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RefereeEnemyHp>(
      "RX", "0x0003 HP", enemy_hp_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "H=" << msg.hero << " E=" << msg.engineer
            << " I3=" << msg.infantry_3 << " I4=" << msg.infantry_4
            << " S=" << msg.sentinel << " OP=" << msg.outpost
            << " Base=" << msg.base;
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::DartStatus>(
      "RX", "0x0105 Dart", dart_status_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "remain=" << static_cast<int>(msg.dart_remaining_time)
            << " hit_target=" << static_cast<int>(msg.dart_hit_target)
            << " hit_count=" << static_cast<int>(msg.dart_hit_count)
            << " selected=" << static_cast<int>(msg.selected_target);
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RadarMarkProgress>(
      "RX", "0x020C Mark", mark_progress_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "raw=" << hexU16(msg.raw_bits)
            << " enemy[H,E,3,4,6,S]="
            << static_cast<int>(msg.hero) << static_cast<int>(msg.engineer)
            << static_cast<int>(msg.infantry_3) << static_cast<int>(msg.infantry_4)
            << static_cast<int>(msg.aerial_6) << static_cast<int>(msg.sentinel);
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RadarDecisionSync>(
      "RX", "0x020E RadarSync", radar_sync_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "chance=" << static_cast<int>(msg.double_effect_chance)
            << " active=" << boolText(msg.is_double_effect_active)
            << " self_key_level=" << static_cast<int>(msg.self_key_level)
            << " can_change_key=" << boolText(msg.can_change_key);
        return oss.str();
      });

    addTypedSubscription<std_msgs::msg::UInt8MultiArray>(
      "RAW", "RX Bytes", raw_rx_topic,
      [](const auto & msg) {
        return "bytes=" + std::to_string(msg.data.size());
      });

    addTypedSubscription<std_msgs::msg::UInt8MultiArray>(
      "RAW", "TX Frame", raw_tx_topic,
      [](const auto & msg) {
        return "bytes=" + std::to_string(msg.data.size());
      });

    addPduSubscription("PDU", "All PDU", pdu_all_topic);
    addPduSubscription("PDU", "Info PDU", pdu_info_topic);
    addPduSubscription("PDU", "Jam PDU", pdu_jam_topic);

    addTypedSubscription<radar_decision_comm::msg::WirelessEnemyHp>(
      "WIRELESS", "0x0A02 HP", wireless_hp_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "H=" << msg.hero << " E=" << msg.engineer
            << " I3=" << msg.infantry_3 << " I4=" << msg.infantry_4
            << " S=" << msg.sentinel;
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::WirelessEnemyAmmo>(
      "WIRELESS", "0x0A03 Ammo", wireless_ammo_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "H=" << msg.hero << " I3=" << msg.infantry_3
            << " I4=" << msg.infantry_4 << " A6=" << msg.aerial_6
            << " S=" << msg.sentinel;
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::WirelessEnemyMacroState>(
      "WIRELESS", "0x0A04 Macro", wireless_macro_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "coins=" << msg.coins_remaining << "/" << msg.coins_total
            << " status=0x" << std::hex << std::uppercase << msg.status_bits;
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::WirelessEnemyBuffs>(
      "WIRELESS", "0x0A05 Buffs", wireless_buffs_topic,
      [](const auto & msg) {
        return "raw_bytes=" + std::to_string(msg.raw_bytes.size());
      });

    addTypedSubscription<radar_decision_comm::msg::WirelessInterferenceKey>(
      "WIRELESS", "0x0A06 Key", wireless_key_topic,
      [](const auto & msg) {
        return "key=" + keyString(msg.key) +
               " raw_bytes=" + std::to_string(msg.raw_bytes.size());
      });

    addTypedSubscription<radar_decision_comm::msg::MapPublishRequest>(
      "TX", "0x0305 Map", tx_map_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "robots=" << msg.robots.size() << " force=" << boolText(msg.force);
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RobotInteractionRequest>(
      "TX", "0x0301 Interaction", tx_interaction_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "data_cmd=" << hexU16(msg.data_cmd_id)
            << " sender=" << msg.sender_id
            << " receiver=" << msg.receiver_id
            << " bytes=" << msg.user_data.size();
        return oss.str();
      });

    addTypedSubscription<radar_decision_comm::msg::RadarDecisionCommand>(
      "TX", "0x0121 RadarCmd", tx_radar_cmd_topic,
      [](const auto & msg) {
        std::ostringstream oss;
        oss << "radar_cmd=" << static_cast<int>(msg.radar_cmd)
            << " password_cmd=" << static_cast<int>(msg.password_cmd);
        if (msg.password_cmd == 2U) {
          oss << " verify_key=" << keyString(msg.password);
        } else if (msg.password_cmd == 1U) {
          oss << " set_key=" << keyString(msg.password);
        }
        oss << " note=" << msg.note;
        return oss.str();
      });

    const auto period_ms = std::max(1, static_cast<int>(1000.0 / std::max(refresh_hz_, 1.0)));
    refresh_timer_ = create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&RefereeStatusWindowNode::drawWindow, this));

    RCLCPP_INFO(
      get_logger(),
      "Referee status window started. topics=%zu stale_timeout=%.2fs rate_window=%.2fs",
      topics_.size(), stale_timeout_s_, rate_window_s_);
  }

  ~RefereeStatusWindowNode() override
  {
    try {
      cv::destroyWindow(window_name_);
    } catch (...) {
    }
  }

private:
  struct TopicStats
  {
    std::string group;
    std::string label;
    std::string topic;
    uint64_t count{0};
    std::deque<rclcpp::Time> stamps;
    rclcpp::Time last_stamp{0, 0, RCL_ROS_TIME};
    std::string summary{"--"};
    bool seen{false};
  };

  template<typename MsgT, typename SummaryFn>
  void addTypedSubscription(
    const std::string & group,
    const std::string & label,
    const std::string & topic,
    SummaryFn summary_fn)
  {
    const auto index = addTopic(group, label, topic);
    subscriptions_.push_back(
      create_subscription<MsgT>(
        topic, 20,
        [this, index, summary_fn](const typename MsgT::SharedPtr msg) {
          updateTopic(index, summary_fn(*msg));
        }));
  }

  void addPduSubscription(
    const std::string & group,
    const std::string & label,
    const std::string & topic)
  {
    addTypedSubscription<std_msgs::msg::String>(
      group, label, topic,
      [](const auto & msg) {
        return summarizePduJson(msg.data);
      });
  }

  std::size_t addTopic(const std::string & group,
                       const std::string & label,
                       const std::string & topic)
  {
    topics_.push_back(TopicStats{group, label, topic});
    return topics_.size() - 1;
  }

  void updateTopic(std::size_t index, const std::string & summary)
  {
    const auto stamp = now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= topics_.size()) {
      return;
    }

    auto & topic = topics_[index];
    ++topic.count;
    topic.last_stamp = stamp;
    topic.stamps.push_back(stamp);
    topic.summary = truncateText(summary, 96);
    topic.seen = true;

    while (!topic.stamps.empty() &&
      (stamp - topic.stamps.front()).seconds() > std::max(rate_window_s_, 0.1))
    {
      topic.stamps.pop_front();
    }
  }

  double topicRateHz(const TopicStats & topic, const rclcpp::Time & t) const
  {
    if (topic.stamps.empty()) {
      return 0.0;
    }

    std::size_t recent = 0;
    for (const auto & stamp : topic.stamps) {
      if ((t - stamp).seconds() <= std::max(rate_window_s_, 0.1)) {
        ++recent;
      }
    }
    return static_cast<double>(recent) / std::max(rate_window_s_, 0.1);
  }

  double topicAgeS(const TopicStats & topic, const rclcpp::Time & t) const
  {
    if (!topic.seen || topic.last_stamp.nanoseconds() == 0) {
      return -1.0;
    }
    return std::max(0.0, (t - topic.last_stamp).seconds());
  }

  cv::Scalar groupColor(const std::string & group) const
  {
    if (group == "RX") {
      return cv::Scalar(80, 210, 120);
    }
    if (group == "WIRELESS") {
      return cv::Scalar(90, 210, 245);
    }
    if (group == "TX") {
      return cv::Scalar(245, 185, 85);
    }
    if (group == "PDU") {
      return cv::Scalar(205, 160, 245);
    }
    return cv::Scalar(180, 190, 200);
  }

  void drawWindow()
  {
    std::vector<TopicStats> topics;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      topics = topics_;
    }

    const auto t = now();
    const int width = 1320;
    const int row_h = 30;
    const int top = 88;
    const int height = std::max(720, top + static_cast<int>(topics.size()) * row_h + 46);
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(24, 27, 31));

    cv::rectangle(canvas, cv::Rect(0, 0, width, 62), cv::Scalar(32, 36, 42), cv::FILLED);
    drawText(canvas, "Referee System Message Statistics", 24, 38, 0.72,
             cv::Scalar(235, 238, 242), 2);

    uint64_t total_count = 0;
    int active_count = 0;
    for (const auto & topic : topics) {
      total_count += topic.count;
      const auto age = topicAgeS(topic, t);
      if (age >= 0.0 && age <= stale_timeout_s_) {
        ++active_count;
      }
    }

    char header[160];
    std::snprintf(
      header, sizeof(header), "topics=%zu active=%d total=%llu stale_timeout=%.1fs rate_window=%.1fs",
      topics.size(), active_count, static_cast<unsigned long long>(total_count),
      stale_timeout_s_, rate_window_s_);
    drawText(canvas, header, 670, 38, 0.48, cv::Scalar(190, 198, 205), 1);

    const int cols[] = {24, 98, 250, 338, 424, 504, 594};
    drawText(canvas, "Group", cols[0], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Message", cols[1], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Count", cols[2], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Hz", cols[3], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Age", cols[4], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "State", cols[5], 78, 0.45, cv::Scalar(190, 198, 205), 1);
    drawText(canvas, "Last", cols[6], 78, 0.45, cv::Scalar(190, 198, 205), 1);

    for (std::size_t i = 0; i < topics.size(); ++i) {
      const auto & topic = topics[i];
      const int y0 = top + static_cast<int>(i) * row_h - 21;
      const int y = top + static_cast<int>(i) * row_h;
      const cv::Scalar bg = (i % 2 == 0) ? cv::Scalar(29, 32, 37)
                                         : cv::Scalar(34, 37, 43);
      cv::rectangle(canvas, cv::Rect(18, y0, width - 36, row_h - 3), bg, cv::FILLED);

      const auto age = topicAgeS(topic, t);
      const bool fresh = age >= 0.0 && age <= stale_timeout_s_;
      const bool never = age < 0.0;
      const cv::Scalar state_color = never ? cv::Scalar(150, 150, 150)
                                           : (fresh ? cv::Scalar(80, 210, 120)
                                                    : cv::Scalar(70, 105, 245));
      const std::string age_text = never ? "--" : formatAge(age);
      const std::string state_text = never ? "NO DATA" : (fresh ? "OK" : "STALE");
      char rate_text[32];
      std::snprintf(rate_text, sizeof(rate_text), "%.2f", topicRateHz(topic, t));

      drawText(canvas, topic.group, cols[0], y, 0.43, groupColor(topic.group), 1);
      drawText(canvas, topic.label, cols[1], y, 0.43, cv::Scalar(224, 226, 230), 1);
      drawText(canvas, std::to_string(topic.count), cols[2], y, 0.43,
               cv::Scalar(224, 226, 230), 1);
      drawText(canvas, rate_text, cols[3], y, 0.43, cv::Scalar(224, 226, 230), 1);
      drawText(canvas, age_text, cols[4], y, 0.43, state_color, 1);
      drawText(canvas, state_text, cols[5], y, 0.43, state_color, 1);
      drawText(canvas, truncateText(topic.summary, 112), cols[6], y, 0.43,
               cv::Scalar(224, 226, 230), 1);
    }

    const int footer_y = height - 18;
    drawText(canvas, "RX: regular referee link   PDU/WIRELESS: GNU Radio decoded link   TX: frames queued to referee serial bridge",
             24, footer_y, 0.42, cv::Scalar(160, 168, 176), 1);

    cv::imshow(window_name_, canvas);
    cv::waitKey(1);
  }

  static std::string formatAge(double age)
  {
    char text[32];
    if (age >= 100.0) {
      std::snprintf(text, sizeof(text), "99+s");
    } else {
      std::snprintf(text, sizeof(text), "%.2fs", age);
    }
    return text;
  }

  std::string window_name_;
  double refresh_hz_{8.0};
  double stale_timeout_s_{2.0};
  double rate_window_s_{5.0};

  std::mutex mutex_;
  std::vector<TopicStats> topics_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
  rclcpp::TimerBase::SharedPtr refresh_timer_;
};

}  // namespace debug_map

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<debug_map::RefereeStatusWindowNode>());
  rclcpp::shutdown();
  return 0;
}
