#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/codec/interaction_payload_builder.hpp"
#include "radar_decision_comm/common/interaction_cmd_ids.hpp"
#include "radar_decision_comm/msg/battle_blackboard.hpp"
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

class DecisionEngineComponent : public rclcpp::Node
{
public:
  explicit DecisionEngineComponent(const rclcpp::NodeOptions & options)
  : Node("decision_engine", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);

    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    my_sentinel_id_ = this->declare_parameter<int>("my_sentinel_id", 7);
    infantry_3_id_ = this->declare_parameter<int>("infantry_3_id", 3);
    infantry_4_id_ = this->declare_parameter<int>("infantry_4_id", 4);
    enemy_hero_id_ = this->declare_parameter<int>("enemy_hero_id", 101);

    decision_rate_hz_ = this->declare_parameter<double>("decision_rate_hz", 50.0);
    map_publish_period_ms_ = this->declare_parameter<int>("map_publish_period_ms", 200);
    map_keepalive_ms_ = this->declare_parameter<int>("map_keepalive_ms", 1000);

    hero_alert_area_check_ = this->declare_parameter<bool>("hero_alert_area_check", true);
    hero_times_threshold_ = this->declare_parameter<int>("hero_times_threshold", 3);
    send_double_threshold_ = this->declare_parameter<int>("send_double_threshold", 3);
    map_send_stale_ms_ = this->declare_parameter<int>("map_send_stale_ms", 300);
    force_zero_when_lost_ms_ = this->declare_parameter<int>("force_zero_when_lost_ms", 800);

    low_hp_alert_threshold_ = this->declare_parameter<int>("low_hp_alert_threshold", 120);
    low_ammo_threshold_ = this->declare_parameter<int>("low_ammo_threshold", 10);
    buff_bonus_score_ = this->declare_parameter<double>("buff_bonus_score", 8.0);
    macro_bonus_score_ = this->declare_parameter<double>("macro_bonus_score", 3.0);

    enable_sentry_alert_angle_ = this->declare_parameter<bool>("enable_sentry_alert_angle", true);
    sentry_alert_period_ms_ = this->declare_parameter<int>("sentry_alert_period_ms", 200);
    sentry_alert_change_distance_m_ = this->declare_parameter<double>("sentry_alert_change_distance_m", 0.30);
    sentry_alert_keepalive_ms_ = this->declare_parameter<int>("sentry_alert_keepalive_ms", 1000);

    enable_sentry_field_state_ = this->declare_parameter<bool>("enable_sentry_field_state", true);
    sentry_field_period_ms_ = this->declare_parameter<int>("sentry_field_period_ms", 500);
    sentry_field_pos_epsilon_m_ = this->declare_parameter<double>("sentry_field_pos_epsilon_m", 0.10);
    sentry_field_keepalive_ms_ = this->declare_parameter<int>("sentry_field_keepalive_ms", 1000);

    enable_hero_alert_flag_ = this->declare_parameter<bool>("enable_hero_alert_flag", true);
    hero_alert_period_ms_ = this->declare_parameter<int>("hero_alert_period_ms", 500);
    hero_alert_keepalive_ms_ = this->declare_parameter<int>("hero_alert_keepalive_ms", 1000);

    enable_double_effect_forward_ = this->declare_parameter<bool>("enable_double_effect_forward", true);
    double_effect_period_ms_ = this->declare_parameter<int>("double_effect_period_ms", 1000);
    double_effect_keepalive_ms_ = this->declare_parameter<int>("double_effect_keepalive_ms", 1500);

    enable_radar_double_effect_command_ = this->declare_parameter<bool>("enable_radar_double_effect_command", true);
    radar_double_effect_period_ms_ = this->declare_parameter<int>("radar_double_effect_period_ms", 1000);
    radar_double_effect_keepalive_ms_ = this->declare_parameter<int>("radar_double_effect_keepalive_ms", 1500);

    output_map_request_topic_ = this->declare_parameter<std::string>(
      "output_map_request_topic", "/radar/decision/map_publish_request");
    output_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "output_robot_interaction_topic", "/radar/decision/robot_interaction_request");
    output_radar_command_topic_ = this->declare_parameter<std::string>(
      "output_radar_command_topic", "/radar/decision/radar_command");

    map_request_pub_ = this->create_publisher<radar_decision_comm::msg::MapPublishRequest>(
      output_map_request_topic_, 10);
    robot_interaction_pub_ = this->create_publisher<radar_decision_comm::msg::RobotInteractionRequest>(
      output_robot_interaction_topic_, 10);
    radar_command_pub_ = this->create_publisher<radar_decision_comm::msg::RadarDecisionCommand>(
      output_radar_command_topic_, 10);

    blackboard_sub_ = this->create_subscription<radar_decision_comm::msg::BattleBlackboard>(
      "/radar/fusion/battle_blackboard", 10,
      [this](const radar_decision_comm::msg::BattleBlackboard::SharedPtr msg)
      {
        last_blackboard_ = *msg;
        have_blackboard_ = true;
      });

    timer_ = this->create_wall_timer(
      hz_to_period_ms(decision_rate_hz_),
      [this]() { on_timer(); });

    RCLCPP_INFO(
      get_logger(),
      "DecisionEngineComponent started. enabled=%s color=%s sentinel=%d inf3=%d inf4=%d enemy_hero=%d",
      enabled_ ? "true" : "false",
      my_color_.c_str(),
      my_sentinel_id_,
      infantry_3_id_,
      infantry_4_id_,
      enemy_hero_id_);
  }

private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  struct InteractionState
  {
    std::vector<uint8_t> user_data;
    uint16_t data_cmd_id{0};
    uint16_t receiver_id{0};
    TimePoint last_sent_tp{};
    bool ever_sent{false};
  };

  struct RadarCmdState
  {
    uint8_t command_type{0};
    uint8_t command_value{0};
    std::string note;
    TimePoint last_sent_tp{};
    bool ever_sent{false};
  };

  static bool due(
    const TimePoint & now,
    const TimePoint & last,
    int period_ms)
  {
    return (now - last) >= std::chrono::milliseconds(period_ms);
  }

  int enemy_slot_from_robot_id(uint16_t robot_id) const
  {
    if (my_color_ == "Red") {
      switch (robot_id) {
        case 101: return 0;
        case 102: return 1;
        case 103: return 2;
        case 104: return 3;
        case 105: return 4;
        case 107: return 5;
        default: return -1;
      }
    } else {
      switch (robot_id) {
        case 1: return 0;
        case 2: return 1;
        case 3: return 2;
        case 4: return 3;
        case 5: return 4;
        case 7: return 5;
        default: return -1;
      }
    }
  }

  uint16_t quadrant_from_xy(double x, double y) const
  {
    const bool right = x >= 14.0;
    const bool upper = y >= 7.5;
    if (!right && upper) return 1;
    if (right && upper) return 2;
    if (!right && !upper) return 3;
    return 4;
  }

  uint16_t wireless_hp_for_robot(uint16_t robot_id) const
  {
    switch (enemy_slot_from_robot_id(robot_id)) {
      case 0: return last_blackboard_.wireless_enemy_hp.hero;
      case 1: return last_blackboard_.wireless_enemy_hp.engineer;
      case 2: return last_blackboard_.wireless_enemy_hp.infantry_3;
      case 3: return last_blackboard_.wireless_enemy_hp.infantry_4;
      case 4: return last_blackboard_.wireless_enemy_hp.infantry_5;
      case 5: return last_blackboard_.wireless_enemy_hp.sentinel;
      default: return 0;
    }
  }

  uint16_t wireless_ammo_for_robot(uint16_t robot_id) const
  {
    switch (enemy_slot_from_robot_id(robot_id)) {
      case 0: return last_blackboard_.wireless_enemy_ammo.hero;
      case 1: return last_blackboard_.wireless_enemy_ammo.engineer;
      case 2: return last_blackboard_.wireless_enemy_ammo.infantry_3;
      case 3: return last_blackboard_.wireless_enemy_ammo.infantry_4;
      case 4: return last_blackboard_.wireless_enemy_ammo.infantry_5;
      case 5: return last_blackboard_.wireless_enemy_ammo.sentinel;
      default: return 0;
    }
  }

  double robot_base_threat(uint16_t robot_id) const
  {
    const int slot = enemy_slot_from_robot_id(robot_id);
    switch (slot) {
      case 0: return 100.0; // hero
      case 5: return 80.0;  // sentinel
      case 2: return 65.0;
      case 3: return 60.0;
      case 4: return 55.0;
      case 1: return 45.0;  // engineer
      default: return 30.0;
    }
  }

  const radar_decision_comm::msg::TrackedRobot * choose_primary_target() const
  {
    if (last_blackboard_.enemy_state.robots.empty()) {
      return nullptr;
    }

    const bool wireless_valid = !last_blackboard_.wireless_stale;
    const bool buffs_valid =
      wireless_valid && !last_blackboard_.wireless_enemy_buffs.raw_bytes.empty();
    const bool macro_valid =
      wireless_valid && !last_blackboard_.wireless_macro_state.raw_bytes.empty();

    const radar_decision_comm::msg::TrackedRobot * best = nullptr;
    double best_score = -1e9;

    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      double score = robot_base_threat(robot.robot_id);

      if (wireless_valid) {
        const auto hp = wireless_hp_for_robot(robot.robot_id);
        if (hp > 0) {
          score += (300.0 - std::min<double>(hp, 300.0)) * 0.2;
        }

        const auto ammo = wireless_ammo_for_robot(robot.robot_id);
        if (ammo > 0 && ammo <= static_cast<uint16_t>(low_ammo_threshold_)) {
          score += 6.0;
        }
      }

      if (buffs_valid) {
        score += buff_bonus_score_;
      }
      if (macro_valid) {
        score += macro_bonus_score_;
      }

      if (robot.source_mask & 0x01U) {
        score += 4.0;
      }
      if (robot.source_mask & 0x04U) {
        score += 2.0;
      }

      if (!best || score > best_score) {
        best = &robot;
        best_score = score;
      }
    }

    return best;
  }

  radar_decision_comm::msg::RobotInteractionRequest make_request(
    uint16_t data_cmd_id,
    uint16_t receiver_id,
    const std::vector<uint8_t> & user_data) const
  {
    radar_decision_comm::msg::RobotInteractionRequest msg;
    msg.header.stamp = this->now();
    msg.data_cmd_id = data_cmd_id;
    msg.sender_id = 0;
    msg.receiver_id = receiver_id;
    msg.user_data = user_data;
    return msg;
  }

  std::string build_map_signature() const
  {
    std::ostringstream oss;
    oss << "n=" << last_blackboard_.enemy_state.robots.size() << ";";
    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      const int x_cm = static_cast<int>(std::lround(robot.field_position.x * 100.0));
      const int y_cm = static_cast<int>(std::lround(robot.field_position.y * 100.0));
      oss << robot.robot_id << ":" << x_cm << "," << y_cm << ";";
    }
    return oss.str();
  }

  bool publish_map_request_if_needed(const TimePoint & now)
  {
    const auto signature = build_map_signature();
    const bool changed = (!map_state_.ever_sent || signature != map_state_.signature);
    const bool keepalive = (!map_state_.ever_sent || due(now, map_state_.last_sent_tp, map_keepalive_ms_));

    if (!(changed || keepalive)) {
      return false;
    }

    radar_decision_comm::msg::MapPublishRequest map_msg;
    map_msg.header.stamp = this->now();
    map_msg.force = false;
    map_msg.robots = last_blackboard_.enemy_state.robots;
    map_request_pub_->publish(map_msg);

    map_state_.signature = signature;
    map_state_.last_sent_tp = now;
    map_state_.ever_sent = true;

    RCLCPP_INFO(
      this->get_logger(),
      "Decision map publish: robots=%zu changed=%s keepalive=%s",
      map_msg.robots.size(),
      changed ? "true" : "false",
      keepalive ? "true" : "false");
    return true;
  }

  bool maybe_publish_interaction(
    const std::string & key,
    const radar_decision_comm::msg::RobotInteractionRequest & req,
    int keepalive_ms,
    const TimePoint & now,
    const char * log_tag)
  {
    auto & state = interaction_states_[key];

    const bool changed =
      (!state.ever_sent) ||
      (state.data_cmd_id != req.data_cmd_id) ||
      (state.receiver_id != req.receiver_id) ||
      (state.user_data != req.user_data);

    const bool keepalive =
      (!state.ever_sent) ||
      due(now, state.last_sent_tp, keepalive_ms);

    if (!(changed || keepalive)) {
      return false;
    }

    robot_interaction_pub_->publish(req);

    state.data_cmd_id = req.data_cmd_id;
    state.receiver_id = req.receiver_id;
    state.user_data = req.user_data;
    state.last_sent_tp = now;
    state.ever_sent = true;

    RCLCPP_INFO(
      this->get_logger(),
      "Decision %s publish: data_cmd_id=0x%04x receiver=%u bytes=%zu changed=%s keepalive=%s",
      log_tag,
      req.data_cmd_id,
      req.receiver_id,
      req.user_data.size(),
      changed ? "true" : "false",
      keepalive ? "true" : "false");
    return true;
  }

  bool maybe_publish_radar_command(
    const radar_decision_comm::msg::RadarDecisionCommand & cmd,
    int keepalive_ms,
    const TimePoint & now,
    const char * log_tag)
  {
    const bool changed =
      (!radar_cmd_state_.ever_sent) ||
      (radar_cmd_state_.command_type != cmd.command_type) ||
      (radar_cmd_state_.command_value != cmd.command_value) ||
      (radar_cmd_state_.note != cmd.note);

    const bool keepalive =
      (!radar_cmd_state_.ever_sent) ||
      due(now, radar_cmd_state_.last_sent_tp, keepalive_ms);

    if (!(changed || keepalive)) {
      return false;
    }

    radar_command_pub_->publish(cmd);

    radar_cmd_state_.command_type = cmd.command_type;
    radar_cmd_state_.command_value = cmd.command_value;
    radar_cmd_state_.note = cmd.note;
    radar_cmd_state_.last_sent_tp = now;
    radar_cmd_state_.ever_sent = true;

    RCLCPP_INFO(
      this->get_logger(),
      "Decision %s publish: type=%u value=%u note=%s changed=%s keepalive=%s",
      log_tag,
      cmd.command_type,
      cmd.command_value,
      cmd.note.c_str(),
      changed ? "true" : "false",
      keepalive ? "true" : "false");
    return true;
  }

  std::optional<radar_decision_comm::msg::RobotInteractionRequest> build_sentry_alert_angle_req() const
  {
    const auto * robot = choose_primary_target();
    if (!robot) {
      return std::nullopt;
    }

    const float dx = static_cast<float>(robot->field_position.x);
    const float dy = static_cast<float>(robot->field_position.y);
    const float distance = std::sqrt(dx * dx + dy * dy);
    const uint16_t quadrant = quadrant_from_xy(robot->field_position.x, robot->field_position.y);

    const auto payload = InteractionPayloadBuilder::build_sentry_alert_angle_payload(
      robot->robot_id, distance, quadrant);

    return make_request(
      interaction_cmd_ids::kSentryAlertAngle,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);
  }

  radar_decision_comm::msg::RobotInteractionRequest build_sentry_field_state_req() const
  {
    std::array<EnemyFieldStateSlot, 6> slots{};

    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      const int slot = enemy_slot_from_robot_id(robot.robot_id);
      if (slot < 0 || slot >= 6) {
        continue;
      }
      slots[static_cast<std::size_t>(slot)].valid = 1;
      slots[static_cast<std::size_t>(slot)].x = static_cast<float>(robot.field_position.x);
      slots[static_cast<std::size_t>(slot)].y = static_cast<float>(robot.field_position.y);
    }

    const auto payload = InteractionPayloadBuilder::build_sentry_field_state_payload(slots);
    return make_request(
      interaction_cmd_ids::kSentryFieldState,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);
  }

  radar_decision_comm::msg::RobotInteractionRequest build_hero_alert_flag_req() const
  {
    bool hero_alert = false;

    if (last_blackboard_.mark_progress.hero >= 100) {
      hero_alert = true;
    }

    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      if (robot.robot_id == static_cast<uint16_t>(enemy_hero_id_)) {
        hero_alert = true;
        break;
      }
    }

    if (!last_blackboard_.wireless_stale) {
      const auto hero_hp = wireless_hp_for_robot(static_cast<uint16_t>(enemy_hero_id_));
      if (hero_hp > 0 && hero_hp <= static_cast<uint16_t>(low_hp_alert_threshold_)) {
        hero_alert = true;
      }
    }

    const auto payload = InteractionPayloadBuilder::build_hero_alert_flag_payload(hero_alert);
    return make_request(
      interaction_cmd_ids::kHeroAlertFlag,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);
  }

  std::vector<radar_decision_comm::msg::RobotInteractionRequest> build_double_effect_forward_reqs() const
  {
    std::vector<radar_decision_comm::msg::RobotInteractionRequest> out;

    if (last_blackboard_.referee_stale) {
      return out;
    }

    const uint8_t times = last_blackboard_.radar_sync.double_effect_chance;
    if (times == 0) {
      return out;
    }

    const auto payload = InteractionPayloadBuilder::build_double_effect_forward_payload(times);
    out.push_back(make_request(
      interaction_cmd_ids::kDoubleEffectForward,
      static_cast<uint16_t>(infantry_3_id_),
      payload));
    out.push_back(make_request(
      interaction_cmd_ids::kDoubleEffectForward,
      static_cast<uint16_t>(infantry_4_id_),
      payload));
    return out;
  }

  std::optional<radar_decision_comm::msg::RadarDecisionCommand> build_radar_double_effect_command() const
  {
    if (last_blackboard_.referee_stale) {
      return std::nullopt;
    }

    const auto times = last_blackboard_.radar_sync.double_effect_chance;
    const bool is_active = last_blackboard_.radar_sync.is_double_effect_active;

    if (times == 0 || is_active) {
      return std::nullopt;
    }

    std::ostringstream note;
    note << "radar_double_effect";
    if (!last_blackboard_.wireless_stale) {
      note << "|hero_hp=" << wireless_hp_for_robot(static_cast<uint16_t>(enemy_hero_id_));
      note << "|hero_ammo=" << wireless_ammo_for_robot(static_cast<uint16_t>(enemy_hero_id_));
      note << "|buff_bytes=" << last_blackboard_.wireless_enemy_buffs.raw_bytes.size();
      note << "|macro_bytes=" << last_blackboard_.wireless_macro_state.raw_bytes.size();
      note << "|key_bytes=" << last_blackboard_.wireless_interference_key.raw_bytes.size();
    }

    radar_decision_comm::msg::RadarDecisionCommand cmd;
    cmd.header.stamp = this->now();
    cmd.command_type = 1;
    cmd.command_value = times;
    cmd.note = note.str();
    return cmd;
  }

  void on_timer()
  {
    if (!enabled_ || !have_blackboard_) {
      return;
    }

    const auto now = Clock::now();

    if (due(now, last_map_eval_tp_, map_publish_period_ms_)) {
      (void)publish_map_request_if_needed(now);
      last_map_eval_tp_ = now;
    }

    if (enable_sentry_alert_angle_ &&
        due(now, last_sentry_alert_eval_tp_, sentry_alert_period_ms_)) {
      auto req = build_sentry_alert_angle_req();
      if (req.has_value()) {
        (void)maybe_publish_interaction(
          "sentry_alert_angle",
          *req,
          sentry_alert_keepalive_ms_,
          now,
          "0x0201");
      }
      last_sentry_alert_eval_tp_ = now;
    }

    if (enable_sentry_field_state_ &&
        due(now, last_sentry_field_eval_tp_, sentry_field_period_ms_)) {
      auto req = build_sentry_field_state_req();
      (void)maybe_publish_interaction(
        "sentry_field_state",
        req,
        sentry_field_keepalive_ms_,
        now,
        "0x0202");
      last_sentry_field_eval_tp_ = now;
    }

    if (enable_hero_alert_flag_ &&
        due(now, last_hero_alert_eval_tp_, hero_alert_period_ms_)) {
      auto req = build_hero_alert_flag_req();
      (void)maybe_publish_interaction(
        "hero_alert_flag",
        req,
        hero_alert_keepalive_ms_,
        now,
        "0x0203");
      last_hero_alert_eval_tp_ = now;
    }

    if (enable_double_effect_forward_ &&
        due(now, last_double_effect_eval_tp_, double_effect_period_ms_)) {
      auto reqs = build_double_effect_forward_reqs();
      for (const auto & req : reqs) {
        const std::string key =
          std::string("double_effect_") + std::to_string(req.receiver_id);
        (void)maybe_publish_interaction(
          key,
          req,
          double_effect_keepalive_ms_,
          now,
          "0x0204");
      }
      last_double_effect_eval_tp_ = now;
    }

    if (enable_radar_double_effect_command_ &&
        due(now, last_radar_cmd_eval_tp_, radar_double_effect_period_ms_)) {
      auto cmd = build_radar_double_effect_command();
      if (cmd.has_value()) {
        (void)maybe_publish_radar_command(
          *cmd,
          radar_double_effect_keepalive_ms_,
          now,
          "0x0121");
      }
      last_radar_cmd_eval_tp_ = now;
    }
  }

  struct MapState
  {
    std::string signature;
    TimePoint last_sent_tp{};
    bool ever_sent{false};
  };

  bool enabled_{true};
  bool hero_alert_area_check_{true};
  bool have_blackboard_{false};

  bool enable_sentry_alert_angle_{true};
  bool enable_sentry_field_state_{true};
  bool enable_hero_alert_flag_{true};
  bool enable_double_effect_forward_{true};
  bool enable_radar_double_effect_command_{true};

  int my_sentinel_id_{7};
  int infantry_3_id_{3};
  int infantry_4_id_{4};
  int enemy_hero_id_{101};

  int hero_times_threshold_{3};
  int send_double_threshold_{3};
  int map_send_stale_ms_{300};
  int force_zero_when_lost_ms_{800};
  int low_hp_alert_threshold_{120};
  int low_ammo_threshold_{10};

  int map_publish_period_ms_{200};
  int map_keepalive_ms_{1000};

  int sentry_alert_period_ms_{200};
  int sentry_alert_keepalive_ms_{1000};

  int sentry_field_period_ms_{500};
  int sentry_field_keepalive_ms_{1000};

  int hero_alert_period_ms_{500};
  int hero_alert_keepalive_ms_{1000};

  int double_effect_period_ms_{1000};
  int double_effect_keepalive_ms_{1500};

  int radar_double_effect_period_ms_{1000};
  int radar_double_effect_keepalive_ms_{1500};

  double decision_rate_hz_{50.0};
  double sentry_alert_change_distance_m_{0.30};
  double sentry_field_pos_epsilon_m_{0.10};
  double buff_bonus_score_{8.0};
  double macro_bonus_score_{3.0};

  std::string my_color_{"Red"};
  std::string output_map_request_topic_;
  std::string output_robot_interaction_topic_;
  std::string output_radar_command_topic_;

  radar_decision_comm::msg::BattleBlackboard last_blackboard_;

  TimePoint last_map_eval_tp_{};
  TimePoint last_sentry_alert_eval_tp_{};
  TimePoint last_sentry_field_eval_tp_{};
  TimePoint last_hero_alert_eval_tp_{};
  TimePoint last_double_effect_eval_tp_{};
  TimePoint last_radar_cmd_eval_tp_{};

  MapState map_state_;
  std::unordered_map<std::string, InteractionState> interaction_states_;
  RadarCmdState radar_cmd_state_{};

  rclcpp::Publisher<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_request_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr robot_interaction_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_command_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::BattleBlackboard>::SharedPtr blackboard_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::DecisionEngineComponent)
