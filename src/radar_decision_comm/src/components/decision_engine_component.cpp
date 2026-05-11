#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/loggers/bt_zmq_publisher.h>
#include <behaviortree_cpp_v3/xml_parsing.h>

#include "radar_decision_comm/codec/interaction_payload_builder.hpp"
#include "radar_decision_comm/common/interaction_cmd_ids.hpp"
#include "radar_decision_comm/common/team_identity.hpp"
#include "radar_decision_comm/decision/double_effect_fsm.hpp"
#include "radar_decision_comm/decision/hero_lob_ballistics.hpp"
#include "radar_decision_comm/decision/hero_lob_fsm.hpp"
#include "radar_decision_comm/decision/hero_lob_window.hpp"
#include "radar_decision_comm/decision/target_selector.hpp"
#include "radar_decision_comm/msg/battle_blackboard.hpp"
#include "radar_decision_comm/msg/hero_lob_assist_debug.hpp"
#include "radar_decision_comm/msg/hero_lob_assist_status.hpp"
#include "radar_decision_comm/msg/map_publish_request.hpp"
#include "radar_decision_comm/msg/radar_decision_command.hpp"
#include "radar_decision_comm/msg/referee_robot_identity.hpp"
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

struct DerivedTeamIds
{
  int my_hero_id{1};
  int my_sentinel_id{7};
  int infantry_3_id{3};
  int infantry_4_id{4};
  int enemy_hero_id{101};
};

static DerivedTeamIds derive_team_ids_from_color(const std::string & my_color)
{
  if (my_color == "Blue" || my_color == "blue") {
    return DerivedTeamIds{101, 107, 103, 104, 1};
  }
  return DerivedTeamIds{};
}

class DecisionEngineComponent : public rclcpp::Node
{
public:
  explicit DecisionEngineComponent(const rclcpp::NodeOptions & options)
  : Node("decision_engine", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", true);
    decision_mode_ = this->declare_parameter<std::string>("decision_mode", "btcpp");   // rule / btcpp

    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    derive_ids_from_my_color_ = this->declare_parameter<bool>("derive_ids_from_my_color", true);
    use_referee_identity_color_ = this->declare_parameter<bool>("use_referee_identity_color", true);
    identity_topic_ = this->declare_parameter<std::string>(
      "identity_topic", "/radar/referee/robot_identity");
    my_hero_id_ = this->declare_parameter<int>("my_hero_id", 1);
    my_sentinel_id_ = this->declare_parameter<int>("my_sentinel_id", 7);
    infantry_3_id_ = this->declare_parameter<int>("infantry_3_id", 3);
    infantry_4_id_ = this->declare_parameter<int>("infantry_4_id", 4);
    enemy_hero_id_ = this->declare_parameter<int>("enemy_hero_id", 101);

    apply_my_color(my_color_);

    decision_rate_hz_ = this->declare_parameter<double>("decision_rate_hz", 50.0);
    map_publish_period_ms_ = this->declare_parameter<int>("map_publish_period_ms", 200);
    map_keepalive_ms_ = this->declare_parameter<int>("map_keepalive_ms", 1000);

    sentry_alert_keepalive_ms_ = this->declare_parameter<int>("sentry_alert_keepalive_ms", 1000);
    sentry_field_keepalive_ms_ = this->declare_parameter<int>("sentry_field_keepalive_ms", 1000);
    hero_alert_keepalive_ms_ = this->declare_parameter<int>("hero_alert_keepalive_ms", 1000);
    double_effect_keepalive_ms_ = this->declare_parameter<int>("double_effect_keepalive_ms", 1500);
    radar_double_effect_keepalive_ms_ = this->declare_parameter<int>("radar_double_effect_keepalive_ms", 1500);

    low_hp_alert_threshold_ = this->declare_parameter<int>("low_hp_alert_threshold", 120);
    low_ammo_threshold_ = this->declare_parameter<int>("low_ammo_threshold", 10);
    buff_bonus_score_ = this->declare_parameter<double>("buff_bonus_score", 8.0);
    macro_bonus_score_ = this->declare_parameter<double>("macro_bonus_score", 3.0);

    enable_sentry_alert_angle_ = this->declare_parameter<bool>("enable_sentry_alert_angle", true);
    enable_sentry_field_state_ = this->declare_parameter<bool>("enable_sentry_field_state", true);
    enable_hero_alert_flag_ = this->declare_parameter<bool>("enable_hero_alert_flag", true);
    enable_hero_lob_assist_ = this->declare_parameter<bool>("enable_hero_lob_assist", false);
    enable_double_effect_forward_ = this->declare_parameter<bool>("enable_double_effect_forward", true);
    enable_radar_double_effect_command_ = this->declare_parameter<bool>("enable_radar_double_effect_command", true);
    enable_interference_key_command_ =
      this->declare_parameter<bool>("enable_interference_key_command", true);
    interference_key_vote_window_ms_ =
      this->declare_parameter<int>("interference_key_vote_window_ms", 10000);
    interference_key_update_period_ms_ =
      this->declare_parameter<int>("interference_key_update_period_ms", 1000);
    radar_cmd_counter_state_path_ =
      this->declare_parameter<std::string>(
        "radar_cmd_counter_state_path", "/tmp/radar_decision_0121_counter.txt");
    radar_cmd_counter_save_period_ms_ =
      this->declare_parameter<int>("radar_cmd_counter_save_period_ms", 1000);

    hero_lob_receiver_id_ = this->declare_parameter<int>("hero_lob_receiver_id", my_hero_id_);
    if (derive_ids_from_my_color_) {
      hero_lob_receiver_id_ = my_hero_id_;
    }
    hero_lob_data_cmd_id_ = this->declare_parameter<int>(
      "hero_lob_data_cmd_id", static_cast<int>(interaction_cmd_ids::kHeroLobAssist));
    hero_lob_shooter_height_m_ = this->declare_parameter<double>("hero_lob_shooter_height_m", 0.30);
    hero_lob_window_s_ = this->declare_parameter<double>("hero_lob_window_s", 2.0);
    hero_lob_min_samples_ = this->declare_parameter<int>("hero_lob_min_samples", 10);
    hero_lob_max_avg_displacement_m_ = this->declare_parameter<double>(
      "hero_lob_max_avg_displacement_m", 0.60);
    hero_lob_enemy_block_radius_m_ = this->declare_parameter<double>(
      "hero_lob_enemy_block_radius_m", 3.0);
    hero_lob_min_enemy_confidence_ = this->declare_parameter<double>(
      "hero_lob_min_enemy_confidence", 0.50);
    hero_lob_require_enemy_valid_ = this->declare_parameter<bool>(
      "hero_lob_require_enemy_valid", true);
    hero_lob_require_enemy_direct_source_ = this->declare_parameter<bool>(
      "hero_lob_require_enemy_direct_source", false);
    hero_lob_enter_quiet_ms_ = this->declare_parameter<int>("hero_lob_enter_quiet_ms", 500);
    hero_lob_keepalive_ms_ = this->declare_parameter<int>("hero_lob_keepalive_ms", 500);
    hero_lob_clear_repeat_ = this->declare_parameter<int>("hero_lob_clear_repeat", 3);
    hero_lob_solve_period_ms_ = this->declare_parameter<int>("hero_lob_solve_period_ms", 200);
    hero_lob_muzzle_speed_mps_ = this->declare_parameter<double>("hero_lob_muzzle_speed_mps", 0.0);
    hero_lob_drag_k_ = this->declare_parameter<double>("hero_lob_drag_k", 0.0);
    hero_lob_yaw_sigma_deg_ = this->declare_parameter<double>("hero_lob_yaw_sigma_deg", 0.0);
    hero_lob_pitch_sigma_deg_ = this->declare_parameter<double>("hero_lob_pitch_sigma_deg", 0.0);
    hero_lob_yaw_bias_deg_ = this->declare_parameter<double>("hero_lob_yaw_bias_deg", 0.0);
    hero_lob_pitch_bias_deg_ = this->declare_parameter<double>("hero_lob_pitch_bias_deg", 0.0);
    hero_lob_integrate_dt_s_ = this->declare_parameter<double>("hero_lob_integrate_dt_s", 0.005);
    hero_lob_max_flight_time_s_ = this->declare_parameter<double>("hero_lob_max_flight_time_s", 5.0);
    hero_lob_yaw_refine_range_deg_ = this->declare_parameter<double>("hero_lob_yaw_refine_range_deg", 1.0);
    hero_lob_yaw_refine_step_deg_ = this->declare_parameter<double>("hero_lob_yaw_refine_step_deg", 0.1);
    hero_lob_pitch_search_min_deg_ = this->declare_parameter<double>(
      "hero_lob_pitch_search_min_deg", 15.0);
    hero_lob_pitch_search_max_deg_ = this->declare_parameter<double>(
      "hero_lob_pitch_search_max_deg", 75.0);
    hero_lob_pitch_search_step_deg_ = this->declare_parameter<double>(
      "hero_lob_pitch_search_step_deg", 0.2);
    hero_lob_target_xyz_when_red_ = this->declare_parameter<std::vector<double>>(
      "hero_lob_target_xyz_when_red", std::vector<double>{0.0, 0.0, 0.0});
    hero_lob_target_xyz_when_blue_ = this->declare_parameter<std::vector<double>>(
      "hero_lob_target_xyz_when_blue", std::vector<double>{0.0, 0.0, 0.0});

    bt_xml_path_ = this->declare_parameter<std::string>("bt_xml_path", "");
    bt_models_path_ = this->declare_parameter<std::string>("bt_models_path", "/tmp/radar_decision_nodes.xml");
    enable_groot_monitoring_ = this->declare_parameter<bool>("enable_groot_monitoring", true);
    groot_publisher_port_ = this->declare_parameter<int>("groot_publisher_port", 1666);
    groot_server_port_ = this->declare_parameter<int>("groot_server_port", 1667);

    output_map_request_topic_ = this->declare_parameter<std::string>(
      "output_map_request_topic", "/radar/decision/map_publish_request");
    output_robot_interaction_topic_ = this->declare_parameter<std::string>(
      "output_robot_interaction_topic", "/radar/decision/robot_interaction_request");
    output_radar_command_topic_ = this->declare_parameter<std::string>(
      "output_radar_command_topic", "/radar/decision/radar_command");
    output_hero_lob_status_topic_ = this->declare_parameter<std::string>(
      "output_hero_lob_status_topic", "/radar/debug/hero_lob_status");
    output_hero_lob_debug_topic_ = this->declare_parameter<std::string>(
      "output_hero_lob_debug_topic", "/radar/debug/hero_lob_debug");

    selector_cfg_.my_color = my_color_;
    selector_cfg_.enemy_hero_id = enemy_hero_id_;
    selector_cfg_.low_hp_alert_threshold = low_hp_alert_threshold_;
    selector_cfg_.low_ammo_threshold = low_ammo_threshold_;
    selector_cfg_.buff_bonus_score = buff_bonus_score_;
    selector_cfg_.macro_bonus_score = macro_bonus_score_;

    map_request_pub_ = this->create_publisher<radar_decision_comm::msg::MapPublishRequest>(
      output_map_request_topic_, 10);
    robot_interaction_pub_ = this->create_publisher<radar_decision_comm::msg::RobotInteractionRequest>(
      output_robot_interaction_topic_, 10);
    radar_command_pub_ = this->create_publisher<radar_decision_comm::msg::RadarDecisionCommand>(
      output_radar_command_topic_, 10);
    hero_lob_status_pub_ = this->create_publisher<radar_decision_comm::msg::HeroLobAssistStatus>(
      output_hero_lob_status_topic_, 10);
    hero_lob_debug_pub_ = this->create_publisher<radar_decision_comm::msg::HeroLobAssistDebug>(
      output_hero_lob_debug_topic_, 10);

    blackboard_sub_ = this->create_subscription<radar_decision_comm::msg::BattleBlackboard>(
      "/radar/fusion/battle_blackboard", 10,
      [this](const radar_decision_comm::msg::BattleBlackboard::SharedPtr msg)
      {
        last_blackboard_ = *msg;
        have_blackboard_ = true;
      });

    identity_sub_ = this->create_subscription<radar_decision_comm::msg::RefereeRobotIdentity>(
      identity_topic_, 10,
      [this](const radar_decision_comm::msg::RefereeRobotIdentity::SharedPtr msg)
      {
        if (!use_referee_identity_color_ || !msg || !msg->valid) {
          return;
        }
        const auto next_color = team_identity::my_color_from_self_color(msg->self_color);
        if (next_color != my_color_) {
          apply_my_color(next_color);
          RCLCPP_INFO(
            this->get_logger(),
            "DecisionEngineComponent switched my_color=%s from referee robot_id=%u",
            my_color_.c_str(),
            msg->robot_id);
        }
      });

    hero_lob_window_.set_window_seconds(hero_lob_window_s_);
    hero_lob_window_.set_min_samples(static_cast<std::size_t>(std::max(hero_lob_min_samples_, 1)));
    hero_lob_window_.set_max_avg_displacement_m(hero_lob_max_avg_displacement_m_);
    load_radar_cmd_counter_state();

    HeroLobBallisticsConfig hero_lob_ballistics_cfg;
    hero_lob_ballistics_cfg.muzzle_speed_mps = hero_lob_muzzle_speed_mps_;
    hero_lob_ballistics_cfg.drag_k = hero_lob_drag_k_;
    hero_lob_ballistics_cfg.yaw_sigma_deg = hero_lob_yaw_sigma_deg_;
    hero_lob_ballistics_cfg.pitch_sigma_deg = hero_lob_pitch_sigma_deg_;
    hero_lob_ballistics_cfg.yaw_bias_deg = hero_lob_yaw_bias_deg_;
    hero_lob_ballistics_cfg.pitch_bias_deg = hero_lob_pitch_bias_deg_;
    hero_lob_ballistics_cfg.integrate_dt_s = hero_lob_integrate_dt_s_;
    hero_lob_ballistics_cfg.max_flight_time_s = hero_lob_max_flight_time_s_;
    hero_lob_ballistics_cfg.yaw_refine_range_deg = hero_lob_yaw_refine_range_deg_;
    hero_lob_ballistics_cfg.yaw_refine_step_deg = hero_lob_yaw_refine_step_deg_;
    hero_lob_ballistics_cfg.pitch_search_min_deg = hero_lob_pitch_search_min_deg_;
    hero_lob_ballistics_cfg.pitch_search_max_deg = hero_lob_pitch_search_max_deg_;
    hero_lob_ballistics_cfg.pitch_search_step_deg = hero_lob_pitch_search_step_deg_;
    hero_lob_ballistics_.set_config(hero_lob_ballistics_cfg);
    hero_lob_fsm_.set_clear_repeat(hero_lob_clear_repeat_);

    init_bt();

    timer_ = this->create_wall_timer(
      hz_to_period_ms(decision_rate_hz_),
      [this]() { on_timer(); });

    RCLCPP_INFO(
      get_logger(),
      "DecisionEngineComponent started. enabled=%s mode=%s xml=%s models=%s groot=%s",
      enabled_ ? "true" : "false",
      decision_mode_.c_str(),
      resolved_bt_xml_path_.c_str(),
      bt_models_path_.c_str(),
      enable_groot_monitoring_ ? "true" : "false");
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
    uint8_t radar_cmd{0};
    uint8_t password_cmd{0};
    std::array<uint8_t, 6> password{};
    std::string note;
    TimePoint last_sent_tp{};
    bool ever_sent{false};
  };

  struct InterferenceKeySample
  {
    std::array<uint8_t, 6> key{};
    TimePoint tp{};
  };

  struct MapState
  {
    std::string signature;
    TimePoint last_sent_tp{};
    bool ever_sent{false};
  };

  static bool due(const TimePoint & now, const TimePoint & last, int period_ms)
  {
    return (now - last) >= std::chrono::milliseconds(period_ms);
  }

  static bool is_alnum_ascii_key(const std::array<uint8_t, 6> & key)
  {
    return std::all_of(
      key.begin(), key.end(),
      [](uint8_t value) {
        return std::isalnum(static_cast<unsigned char>(value)) != 0;
      });
  }

  static std::string key_to_string(const std::array<uint8_t, 6> & key)
  {
    return std::string(key.begin(), key.end());
  }

  std::array<uint8_t, 6> random_ascii_key()
  {
    static constexpr char kAlphabet[] =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::array<uint8_t, 6> key{};
    std::uniform_int_distribution<std::size_t> dist(0, sizeof(kAlphabet) - 2);
    for (auto & byte : key) {
      byte = static_cast<uint8_t>(kAlphabet[dist(rng_)]);
    }
    return key;
  }

  void load_radar_cmd_counter_state()
  {
    if (radar_cmd_counter_state_path_.empty()) {
      return;
    }

    std::ifstream in(radar_cmd_counter_state_path_);
    int value = 0;
    if (!(in >> value)) {
      return;
    }

    value = std::clamp(value, 0, 255);
    radar_cmd_counter_ = static_cast<uint8_t>(value);
    RCLCPP_INFO(
      this->get_logger(),
      "Loaded 0x0121 radar_cmd counter=%d from %s",
      value,
      radar_cmd_counter_state_path_.c_str());
  }

  void save_radar_cmd_counter_state(bool force = false)
  {
    if (radar_cmd_counter_state_path_.empty()) {
      return;
    }
    if (
      !force && have_last_radar_cmd_counter_save_tp_ &&
      !due(current_tick_now_, last_radar_cmd_counter_save_tp_, radar_cmd_counter_save_period_ms_))
    {
      return;
    }

    std::ofstream out(radar_cmd_counter_state_path_, std::ios::trunc);
    if (!out) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Failed to open 0x0121 counter state file: %s",
        radar_cmd_counter_state_path_.c_str());
      return;
    }

    out << static_cast<int>(radar_cmd_counter_) << '\n';
    last_radar_cmd_counter_save_tp_ = current_tick_now_;
    have_last_radar_cmd_counter_save_tp_ = true;
  }

  void update_match_start_counter_reset()
  {
    const int progress = static_cast<int>(last_blackboard_.game_state.game_progress);
    if (last_game_progress_ < 0) {
      last_game_progress_ = progress;
      return;
    }

    if (last_game_progress_ != 4 && progress == 4) {
      radar_cmd_counter_ = 0U;
      double_effect_request_latched_ = false;
      save_radar_cmd_counter_state(true);
      RCLCPP_INFO(
        this->get_logger(),
        "Referee game_progress entered 4; reset 0x0121 radar_cmd counter to 0.");
    }

    last_game_progress_ = progress;
  }

  int enemy_slot_from_robot_id(uint16_t robot_id) const
  {
    if (my_color_ == "Red") {
      switch (robot_id) {
        case 101: return 0;
        case 102: return 1;
        case 103: return 2;
        case 104: return 3;
        case 106: return 4;
        case 107: return 5;
        default: return -1;
      }
    }
    switch (robot_id) {
      case 1: return 0;
      case 2: return 1;
      case 3: return 2;
      case 4: return 3;
      case 6: return 4;
      case 7: return 5;
      default: return -1;
    }
  }

  uint16_t wireless_hp_for_robot(uint16_t robot_id) const
  {
    switch (enemy_slot_from_robot_id(robot_id)) {
      case 0: return last_blackboard_.wireless_enemy_hp.hero;
      case 1: return last_blackboard_.wireless_enemy_hp.engineer;
      case 2: return last_blackboard_.wireless_enemy_hp.infantry_3;
      case 3: return last_blackboard_.wireless_enemy_hp.infantry_4;
      case 4: return 0;
      case 5: return last_blackboard_.wireless_enemy_hp.sentinel;
      default: return 0;
    }
  }

  uint16_t wireless_ammo_for_robot(uint16_t robot_id) const
  {
    switch (enemy_slot_from_robot_id(robot_id)) {
      case 0: return last_blackboard_.wireless_enemy_ammo.hero;
      case 1: return 0;
      case 2: return last_blackboard_.wireless_enemy_ammo.infantry_3;
      case 3: return last_blackboard_.wireless_enemy_ammo.infantry_4;
      case 4: return last_blackboard_.wireless_enemy_ammo.aerial_6;
      case 5: return last_blackboard_.wireless_enemy_ammo.sentinel;
      default: return 0;
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

  void apply_my_color(const std::string & my_color)
  {
    my_color_ = my_color;
    if (derive_ids_from_my_color_) {
      const auto ids = derive_team_ids_from_color(my_color_);
      my_hero_id_ = ids.my_hero_id;
      my_sentinel_id_ = ids.my_sentinel_id;
      infantry_3_id_ = ids.infantry_3_id;
      infantry_4_id_ = ids.infantry_4_id;
      enemy_hero_id_ = ids.enemy_hero_id;
      hero_lob_receiver_id_ = my_hero_id_;
    }
    selector_cfg_.my_color = my_color_;
    selector_cfg_.enemy_hero_id = enemy_hero_id_;
  }

  radar_decision_comm::msg::RobotInteractionRequest make_request(
    uint16_t data_cmd_id, uint16_t receiver_id, const std::vector<uint8_t> & user_data) const
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
    oss << "enemy_n=" << last_blackboard_.enemy_state.robots.size() << ";";
    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      const int x_cm = static_cast<int>(std::lround(robot.field_position.x * 100.0));
      const int y_cm = static_cast<int>(std::lround(robot.field_position.y * 100.0));
      oss << "e" << robot.robot_id << ":" << x_cm << "," << y_cm << ";";
    }
    oss << "our_stale=" << (last_blackboard_.our_stale ? 1 : 0)
        << ";our_n=" << (last_blackboard_.our_stale ? 0 : last_blackboard_.our_state.robots.size())
        << ";";
    if (!last_blackboard_.our_stale) {
      for (const auto & robot : last_blackboard_.our_state.robots) {
        const int x_cm = static_cast<int>(std::lround(robot.field_position.x * 100.0));
        const int y_cm = static_cast<int>(std::lround(robot.field_position.y * 100.0));
        oss << "a" << robot.robot_id << ":" << x_cm << "," << y_cm << ";";
      }
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
    if (!last_blackboard_.our_stale) {
      map_msg.robots.insert(
        map_msg.robots.end(),
        last_blackboard_.our_state.robots.begin(),
        last_blackboard_.our_state.robots.end());
    }
    map_request_pub_->publish(map_msg);

    map_state_.signature = signature;
    map_state_.last_sent_tp = now;
    map_state_.ever_sent = true;
    return true;
  }

  bool maybe_publish_interaction(
    const std::string & key,
    const radar_decision_comm::msg::RobotInteractionRequest & req,
    int keepalive_ms,
    const TimePoint & now)
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
    return true;
  }

  bool maybe_publish_radar_command(
    const radar_decision_comm::msg::RadarDecisionCommand & cmd,
    int keepalive_ms,
    const TimePoint & now)
  {
    const bool changed =
      (!radar_cmd_state_.ever_sent) ||
      (radar_cmd_state_.command_type != cmd.command_type) ||
      (radar_cmd_state_.command_value != cmd.command_value) ||
      (radar_cmd_state_.radar_cmd != cmd.radar_cmd) ||
      (radar_cmd_state_.password_cmd != cmd.password_cmd) ||
      (radar_cmd_state_.password != cmd.password) ||
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
    radar_cmd_state_.radar_cmd = cmd.radar_cmd;
    radar_cmd_state_.password_cmd = cmd.password_cmd;
    radar_cmd_state_.password = cmd.password;
    radar_cmd_state_.note = cmd.note;
    radar_cmd_state_.last_sent_tp = now;
    radar_cmd_state_.ever_sent = true;
    return true;
  }

  bool evaluate_hero_alert() const
  {
    if (!last_blackboard_.referee_stale && last_blackboard_.mark_progress.hero != 0) {
      return true;
    }
    if (current_selected_target_.has_target &&
        current_selected_target_.primary_target.robot_id == static_cast<uint16_t>(enemy_hero_id_)) {
      return true;
    }
    if (!last_blackboard_.wireless_stale) {
      const auto hero_hp = wireless_hp_for_robot(static_cast<uint16_t>(enemy_hero_id_));
      if (hero_hp > 0 && hero_hp <= static_cast<uint16_t>(low_hp_alert_threshold_)) {
        return true;
      }
    }
    return false;
  }

  DoubleEffectFSMInput build_double_effect_input() const
  {
    DoubleEffectFSMInput in;
    in.referee_fresh = !last_blackboard_.referee_stale;
    in.wireless_fresh = !last_blackboard_.wireless_stale;
    in.double_effect_available =
      (!last_blackboard_.referee_stale && last_blackboard_.radar_sync.double_effect_chance > 0);
    in.double_effect_active =
      (!last_blackboard_.referee_stale && last_blackboard_.radar_sync.is_double_effect_active);
    in.hero_marked =
      (!last_blackboard_.referee_stale && last_blackboard_.mark_progress.hero != 0);
    in.hero_hp = wireless_hp_for_robot(static_cast<uint16_t>(enemy_hero_id_));
    in.hero_ammo = wireless_ammo_for_robot(static_cast<uint16_t>(enemy_hero_id_));
    in.has_buffs = (!last_blackboard_.wireless_stale &&
                    !last_blackboard_.wireless_enemy_buffs.raw_bytes.empty());
    in.has_macro = (!last_blackboard_.wireless_stale &&
                    !last_blackboard_.wireless_macro_state.raw_bytes.empty());
    in.available_times = last_blackboard_.referee_stale ? 0 : last_blackboard_.radar_sync.double_effect_chance;
    in.low_hp_threshold = low_hp_alert_threshold_;
    in.low_ammo_threshold = low_ammo_threshold_;
    return in;
  }

  void update_interference_key_votes(const TimePoint & now)
  {
    const auto & msg = last_blackboard_.wireless_interference_key;
    const int64_t stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
    if (stamp_ns > 0 && stamp_ns != last_interference_key_stamp_ns_ && is_alnum_ascii_key(msg.key)) {
      interference_key_samples_.push_back(InterferenceKeySample{msg.key, now});
      last_interference_key_stamp_ns_ = stamp_ns;
      if (!interference_key_window_active_) {
        interference_key_window_start_ = now;
        interference_key_window_active_ = true;
      }
    }

    const auto window = std::chrono::milliseconds(std::max(interference_key_vote_window_ms_, 1));
    while (!interference_key_samples_.empty() && (now - interference_key_samples_.front().tp) > window) {
      interference_key_samples_.pop_front();
    }
    if (interference_key_samples_.empty()) {
      interference_key_window_active_ = false;
    }
  }

  bool select_voted_interference_key(std::array<uint8_t, 6> * out) const
  {
    if (out == nullptr || interference_key_samples_.empty()) {
      return false;
    }

    std::unordered_map<std::string, int> counts;
    std::unordered_map<std::string, std::array<uint8_t, 6>> keys;
    for (const auto & sample : interference_key_samples_) {
      const auto key_text = key_to_string(sample.key);
      ++counts[key_text];
      keys[key_text] = sample.key;
    }

    std::string best_key_text;
    int best_count = -1;
    for (const auto & kv : counts) {
      if (kv.second > best_count || (kv.second == best_count && kv.first < best_key_text)) {
        best_key_text = kv.first;
        best_count = kv.second;
      }
    }

    if (best_count <= 0) {
      return false;
    }

    *out = keys.at(best_key_text);
    return true;
  }

  bool take_password_action(uint8_t * password_cmd, std::array<uint8_t, 6> * password)
  {
    if (!enable_interference_key_command_ || password_cmd == nullptr || password == nullptr) {
      return false;
    }

    if (
      interference_key_window_active_ &&
      due(current_tick_now_, interference_key_window_start_, interference_key_vote_window_ms_))
    {
      std::array<uint8_t, 6> voted_key{};
      if (select_voted_interference_key(&voted_key)) {
        *password_cmd = 2U;
        *password = voted_key;
        last_interference_verify_tp_ = current_tick_now_;
        have_last_interference_verify_tp_ = true;
        interference_key_samples_.clear();
        interference_key_window_active_ = false;
        return true;
      }
      interference_key_window_active_ = false;
    }

    if (
      !have_last_interference_update_tp_ ||
      due(current_tick_now_, last_interference_update_tp_, interference_key_update_period_ms_))
    {
      *password_cmd = 1U;
      *password = random_ascii_key();
      last_interference_update_tp_ = current_tick_now_;
      have_last_interference_update_tp_ = true;
      return true;
    }

    return false;
  }

  static double xy_distance(
    const geometry_msgs::msg::Point & lhs,
    const geometry_msgs::msg::Point & rhs)
  {
    return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
  }

  bool find_our_hero_track(radar_decision_comm::msg::TrackedRobot * out) const
  {
    if (out == nullptr) {
      return false;
    }

    bool found = false;
    float best_confidence = -1.0F;
    for (const auto & robot : last_blackboard_.our_state.robots) {
      if (robot.robot_id != static_cast<uint16_t>(my_hero_id_)) {
        continue;
      }

      const bool prefer_this =
        (!found) ||
        (robot.is_valid && !out->is_valid) ||
        (robot.confidence > best_confidence);
      if (prefer_this) {
        *out = robot;
        best_confidence = robot.confidence;
        found = true;
      }
    }
    return found;
  }

  geometry_msgs::msg::Point resolve_hero_lob_target_point(bool * valid) const
  {
    geometry_msgs::msg::Point point;
    const auto & target_xyz = (my_color_ == "Red") ? hero_lob_target_xyz_when_red_ : hero_lob_target_xyz_when_blue_;
    const bool ok = target_xyz.size() == 3;
    if (ok) {
      point.x = target_xyz[0];
      point.y = target_xyz[1];
      point.z = target_xyz[2];
    }
    if (valid != nullptr) {
      *valid = ok;
    }
    return point;
  }

  uint16_t quantized_hero_lob_age_ms() const
  {
    if (!have_hero_lob_solve_tp_) {
      return 0;
    }

    const auto age_ms = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(current_tick_now_ - last_hero_lob_solve_tp_).count());
    const int quantum_ms = std::max(hero_lob_keepalive_ms_, 1);
    const int quantized_ms = std::max(age_ms, 0) / quantum_ms * quantum_ms;
    return static_cast<uint16_t>(std::min(quantized_ms, 65535));
  }

  void publish_hero_lob_topics()
  {
    radar_decision_comm::msg::HeroLobAssistStatus status_msg;
    status_msg.header.stamp = this->now();
    status_msg.state = static_cast<uint8_t>(current_hero_lob_fsm_output_.state);
    status_msg.hero_robot_id = static_cast<uint16_t>(my_hero_id_);
    status_msg.nearest_enemy_robot_id = current_nearest_enemy_id_;
    status_msg.hero_found = current_hero_found_;
    status_msg.our_track_fresh = current_hero_track_fresh_;
    status_msg.pose_stable = current_hero_lob_window_output_.stable;
    status_msg.enemy_nearby_3m = current_enemy_nearby_3m_;
    status_msg.solution_valid = current_hero_lob_solution_usable_;
    status_msg.publish_active = current_hero_lob_fsm_output_.publish_active;
    status_msg.window_avg_displacement_m = std::isfinite(current_hero_lob_window_output_.avg_displacement_m) ?
      static_cast<float>(current_hero_lob_window_output_.avg_displacement_m) : -1.0F;
    status_msg.nearest_enemy_distance_m = std::isfinite(current_nearest_enemy_distance_m_) ?
      static_cast<float>(current_nearest_enemy_distance_m_) : -1.0F;
    status_msg.reason = current_hero_lob_fsm_output_.reason;
    if (
      status_msg.reason == "solving_no_solution" &&
      !current_hero_lob_solution_.reason.empty())
    {
      status_msg.reason += "|" + current_hero_lob_solution_.reason;
    }
    hero_lob_status_pub_->publish(status_msg);

    radar_decision_comm::msg::HeroLobAssistDebug debug_msg;
    debug_msg.header.stamp = status_msg.header.stamp;
    debug_msg.hero_track = current_hero_track_;
    debug_msg.shooter_point = current_hero_lob_shooter_point_;
    debug_msg.target_point = current_hero_lob_target_point_;
    debug_msg.impact_center_point = current_hero_lob_solution_.impact_center_point;
    debug_msg.solution_valid = current_hero_lob_solution_usable_;
    debug_msg.yaw_field_cw_deg = static_cast<float>(current_hero_lob_solution_.yaw_field_cw_deg);
    debug_msg.pitch_deg = static_cast<float>(current_hero_lob_solution_.pitch_deg);
    debug_msg.flight_time_s = static_cast<float>(current_hero_lob_solution_.flight_time_s);
    debug_msg.impact_error_m = std::isfinite(current_hero_lob_solution_.impact_error_m) ?
      static_cast<float>(current_hero_lob_solution_.impact_error_m) : -1.0F;
    debug_msg.splash_radius_m = static_cast<float>(current_hero_lob_solution_.splash_radius_m);
    debug_msg.muzzle_speed_mps = static_cast<float>(hero_lob_muzzle_speed_mps_);
    debug_msg.drag_k = static_cast<float>(hero_lob_drag_k_);
    debug_msg.yaw_sigma_deg = static_cast<float>(hero_lob_yaw_sigma_deg_);
    debug_msg.pitch_sigma_deg = static_cast<float>(hero_lob_pitch_sigma_deg_);
    debug_msg.yaw_bias_deg = static_cast<float>(hero_lob_yaw_bias_deg_);
    debug_msg.pitch_bias_deg = static_cast<float>(hero_lob_pitch_bias_deg_);
    hero_lob_debug_pub_->publish(debug_msg);
  }

  void update_hero_lob_assist()
  {
    current_hero_track_ = radar_decision_comm::msg::TrackedRobot();
    current_hero_found_ = find_our_hero_track(&current_hero_track_);
    current_hero_track_fresh_ = current_hero_found_ && !last_blackboard_.our_stale;

    if (current_hero_track_fresh_) {
      hero_lob_window_.ingest(current_hero_track_.field_position, last_blackboard_.our_last_update);
    } else {
      hero_lob_window_.reset();
    }
    current_hero_lob_window_output_ = hero_lob_window_.evaluate();

    current_hero_lob_shooter_point_ = current_hero_found_ ?
      current_hero_track_.field_position : geometry_msgs::msg::Point();
    if (current_hero_lob_window_output_.sample_count > 0U) {
      current_hero_lob_shooter_point_ = current_hero_lob_window_output_.mean_position;
    }
    current_hero_lob_shooter_point_.z = hero_lob_shooter_height_m_;

    bool target_valid = false;
    current_hero_lob_target_point_ = resolve_hero_lob_target_point(&target_valid);

    current_nearest_enemy_distance_m_ = std::numeric_limits<double>::infinity();
    current_nearest_enemy_id_ = 0;
    current_enemy_nearby_3m_ = false;
    if (current_hero_found_) {
      const geometry_msgs::msg::Point hero_reference_point =
        (current_hero_lob_window_output_.sample_count > 0U) ?
        current_hero_lob_window_output_.mean_position : current_hero_track_.field_position;

      for (const auto & robot : last_blackboard_.enemy_state.robots) {
        if (hero_lob_require_enemy_valid_ && !robot.is_valid) {
          continue;
        }
        if (robot.confidence < static_cast<float>(hero_lob_min_enemy_confidence_)) {
          continue;
        }
        if (hero_lob_require_enemy_direct_source_ && ((robot.source_mask & 0x01U) == 0U)) {
          continue;
        }

        const double dist = xy_distance(hero_reference_point, robot.field_position);
        if (dist < current_nearest_enemy_distance_m_) {
          current_nearest_enemy_distance_m_ = dist;
          current_nearest_enemy_id_ = robot.robot_id;
        }
      }

      current_enemy_nearby_3m_ =
        std::isfinite(current_nearest_enemy_distance_m_) &&
        current_nearest_enemy_distance_m_ <= hero_lob_enemy_block_radius_m_;
    }

    const bool solve_gate =
      enable_hero_lob_assist_ &&
      current_hero_track_fresh_ &&
      current_hero_lob_window_output_.ready &&
      current_hero_lob_window_output_.stable &&
      !current_enemy_nearby_3m_ &&
      target_valid;

    if (
      solve_gate &&
      (!have_hero_lob_solve_tp_ || due(current_tick_now_, last_hero_lob_solve_tp_, hero_lob_solve_period_ms_)))
    {
      current_hero_lob_solution_ = hero_lob_ballistics_.solve(
        current_hero_lob_shooter_point_, current_hero_lob_target_point_);
      last_hero_lob_solve_tp_ = current_tick_now_;
      have_hero_lob_solve_tp_ = true;
    }

    current_hero_lob_solution_usable_ = solve_gate && current_hero_lob_solution_.valid;
    if (current_hero_lob_solution_usable_) {
      if (!hero_lob_ready_since_.has_value()) {
        hero_lob_ready_since_ = current_tick_now_;
      }
    } else {
      hero_lob_ready_since_.reset();
    }

    HeroLobFSMInput hero_lob_input;
    hero_lob_input.enabled = enable_hero_lob_assist_;
    hero_lob_input.hero_found = current_hero_found_;
    hero_lob_input.our_track_fresh = current_hero_track_fresh_;
    hero_lob_input.window_ready = current_hero_lob_window_output_.ready;
    hero_lob_input.pose_stable = current_hero_lob_window_output_.stable;
    hero_lob_input.enemy_nearby_3m = current_enemy_nearby_3m_;
    hero_lob_input.quiet_enter_satisfied =
      current_hero_lob_solution_usable_ &&
      hero_lob_ready_since_.has_value() &&
      due(current_tick_now_, *hero_lob_ready_since_, hero_lob_enter_quiet_ms_);
    hero_lob_input.solution_valid = current_hero_lob_solution_usable_;

    current_hero_lob_fsm_output_ = hero_lob_fsm_.tick(hero_lob_input);
    if (current_hero_lob_fsm_output_.state != HeroLobState::CLEARING) {
      hero_lob_clear_sequence_ = 0;
    }

    publish_hero_lob_topics();
  }

  bool action_publish_map() { return publish_map_request_if_needed(current_tick_now_); }

  bool action_send_sentry_alert()
  {
    if (!enable_sentry_alert_angle_ || !current_selected_target_.has_target) {
      return false;
    }

    const auto & robot = current_selected_target_.primary_target;
    const float dx = static_cast<float>(robot.field_position.x);
    const float dy = static_cast<float>(robot.field_position.y);
    const float distance = std::sqrt(dx * dx + dy * dy);
    const uint16_t quadrant = quadrant_from_xy(robot.field_position.x, robot.field_position.y);

    const auto payload = InteractionPayloadBuilder::build_sentry_alert_angle_payload(
      robot.robot_id, distance, quadrant);

    const auto req = make_request(
      interaction_cmd_ids::kSentryAlertAngle,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);

    return maybe_publish_interaction("sentry_alert_angle", req, sentry_alert_keepalive_ms_, current_tick_now_);
  }

  bool action_send_sentry_field_state()
  {
    if (!enable_sentry_field_state_) {
      return false;
    }

    std::array<EnemyFieldStateSlot, 6> slots{};
    for (const auto & robot : last_blackboard_.enemy_state.robots) {
      const int slot = enemy_slot_from_robot_id(robot.robot_id);
      if (slot < 0 || slot >= 6) continue;
      slots[static_cast<std::size_t>(slot)].valid = 1;
      slots[static_cast<std::size_t>(slot)].x = static_cast<float>(robot.field_position.x);
      slots[static_cast<std::size_t>(slot)].y = static_cast<float>(robot.field_position.y);
    }

    const auto payload = InteractionPayloadBuilder::build_sentry_field_state_payload(slots);
    const auto req = make_request(
      interaction_cmd_ids::kSentryFieldState,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);

    return maybe_publish_interaction("sentry_field_state", req, sentry_field_keepalive_ms_, current_tick_now_);
  }

  bool action_send_hero_alert()
  {
    if (!enable_hero_alert_flag_) {
      return false;
    }

    const bool hero_alert = evaluate_hero_alert();
    const auto payload = InteractionPayloadBuilder::build_hero_alert_flag_payload(hero_alert);
    const auto req = make_request(
      interaction_cmd_ids::kHeroAlertFlag,
      static_cast<uint16_t>(my_sentinel_id_),
      payload);

    return maybe_publish_interaction("hero_alert_flag", req, hero_alert_keepalive_ms_, current_tick_now_);
  }

  bool action_send_hero_lob_assist()
  {
    if (!enable_hero_lob_assist_ || !current_hero_lob_fsm_output_.action_needed) {
      return false;
    }

    uint8_t flags = 0U;
    if (current_hero_lob_solution_usable_) {
      flags |= 0x01U;
    }
    if (current_hero_lob_window_output_.stable) {
      flags |= 0x02U;
    }
    if (current_enemy_nearby_3m_) {
      flags |= 0x04U;
    }
    if (!current_hero_lob_fsm_output_.clear_mode && current_hero_lob_solution_usable_) {
      flags |= 0x08U;
    }
    if (current_hero_lob_fsm_output_.clear_mode) {
      flags |= 0x10U;
    }

    const uint8_t sequence_tag = current_hero_lob_fsm_output_.clear_mode ? ++hero_lob_clear_sequence_ : 0U;
    const auto payload = InteractionPayloadBuilder::build_hero_lob_assist_payload(
      flags,
      current_hero_lob_solution_.valid ? current_hero_lob_solution_.yaw_field_cw_deg : 0.0,
      current_hero_lob_solution_.valid ? current_hero_lob_solution_.pitch_deg : 0.0,
      current_hero_lob_solution_.valid ? quantized_hero_lob_age_ms() : 0U,
      current_hero_lob_solution_.valid ? current_hero_lob_solution_.impact_error_m : 0.0,
      current_hero_lob_solution_.valid ? current_hero_lob_solution_.splash_radius_m : 0.0,
      static_cast<uint8_t>(current_hero_lob_fsm_output_.state),
      sequence_tag);

    const auto req = make_request(
      static_cast<uint16_t>(hero_lob_data_cmd_id_),
      static_cast<uint16_t>(hero_lob_receiver_id_),
      payload);

    const bool sent = maybe_publish_interaction(
      "hero_lob_assist", req, hero_lob_keepalive_ms_, current_tick_now_);
    if (sent && current_hero_lob_fsm_output_.clear_mode) {
      hero_lob_fsm_.note_action_sent();
    }
    return sent;
  }

  bool action_send_double_forward()
  {
    if (!enable_double_effect_forward_ || !current_fsm_output_.should_forward) {
      return false;
    }

    const auto payload =
      InteractionPayloadBuilder::build_double_effect_forward_payload(current_fsm_output_.forward_times);

    bool changed = false;
    changed |= maybe_publish_interaction(
      "double_effect_3",
      make_request(interaction_cmd_ids::kDoubleEffectForward, static_cast<uint16_t>(infantry_3_id_), payload),
      double_effect_keepalive_ms_,
      current_tick_now_);

    changed |= maybe_publish_interaction(
      "double_effect_4",
      make_request(interaction_cmd_ids::kDoubleEffectForward, static_cast<uint16_t>(infantry_4_id_), payload),
      double_effect_keepalive_ms_,
      current_tick_now_);

    return changed;
  }

  bool action_send_radar_decision()
  {
    bool double_effect_changed = false;
    if (enable_radar_double_effect_command_ && current_fsm_output_.should_send_radar_command) {
      if (!double_effect_request_latched_ && radar_cmd_counter_ < 255U) {
        ++radar_cmd_counter_;
        double_effect_changed = true;
        save_radar_cmd_counter_state(true);
      }
      double_effect_request_latched_ = true;
    } else {
      double_effect_request_latched_ = false;
    }

    uint8_t password_cmd = 0U;
    std::array<uint8_t, 6> password{};
    const bool password_changed = take_password_action(&password_cmd, &password);

    if (!double_effect_changed && !password_changed) {
      return false;
    }

    radar_decision_comm::msg::RadarDecisionCommand cmd;
    cmd.header.stamp = this->now();
    cmd.command_type = 1;
    cmd.command_value = radar_cmd_counter_;
    cmd.radar_cmd = radar_cmd_counter_;
    cmd.password_cmd = password_cmd;
    cmd.password = password;

    std::ostringstream note;
    note << "btcpp_radar_decision";
    if (double_effect_changed && current_selected_target_.has_target) {
      note << "|double_effect=" << static_cast<int>(radar_cmd_counter_);
      note << "|target=" << current_selected_target_.primary_target.robot_id;
      note << "|reason=" << current_selected_target_.reason;
    } else if (double_effect_changed) {
      note << "|double_effect=" << static_cast<int>(radar_cmd_counter_) << "|no_target";
    }
    if (password_changed) {
      note << "|password_cmd=" << static_cast<int>(password_cmd);
      if (password_cmd == 2U) {
        note << "|verify=" << key_to_string(password);
      } else if (password_cmd == 1U) {
        note << "|update";
      }
    }
    cmd.note = note.str();

    return maybe_publish_radar_command(cmd, radar_double_effect_keepalive_ms_, current_tick_now_);
  }

  void register_bt_nodes()
  {
    factory_.registerSimpleAction("Noop", [](BT::TreeNode &) { return BT::NodeStatus::SUCCESS; });

    factory_.registerSimpleAction("PublishMap", [this](BT::TreeNode &) {
      (void)action_publish_map();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendSentryAlertAngle", [this](BT::TreeNode &) {
      (void)action_send_sentry_alert();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendSentryFieldState", [this](BT::TreeNode &) {
      (void)action_send_sentry_field_state();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendHeroAlertFlag", [this](BT::TreeNode &) {
      (void)action_send_hero_alert();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendHeroLobAssist", [this](BT::TreeNode &) {
      (void)action_send_hero_lob_assist();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendDoubleEffectForward", [this](BT::TreeNode &) {
      (void)action_send_double_forward();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendRadarDecision", [this](BT::TreeNode &) {
      (void)action_send_radar_decision();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("HasEnemy", [this](BT::TreeNode &) {
      return current_selected_target_.has_target ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("RefereeFresh", [this](BT::TreeNode &) {
      return last_blackboard_.referee_stale ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("WirelessFresh", [this](BT::TreeNode &) {
      return last_blackboard_.wireless_stale ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("HeroAlertNeeded", [this](BT::TreeNode &) {
      return evaluate_hero_alert() ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("HeroLobActionNeeded", [this](BT::TreeNode &) {
      return current_hero_lob_fsm_output_.action_needed ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("DoubleEffectForwardNeeded", [this](BT::TreeNode &) {
      return current_fsm_output_.should_forward ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("RadarCommandNeeded", [this](BT::TreeNode &) {
      return (current_fsm_output_.should_send_radar_command || enable_interference_key_command_)
        ? BT::NodeStatus::SUCCESS
        : BT::NodeStatus::FAILURE;
    });
  }

  void init_bt()
  {
    register_bt_nodes();

    std::ofstream(bt_models_path_) << BT::writeTreeNodesModelXML(factory_);

    if (bt_xml_path_.empty()) {
      resolved_bt_xml_path_ =
        ament_index_cpp::get_package_share_directory("radar_decision_comm") + "/config/bt/decision_tree.xml";
    } else {
      resolved_bt_xml_path_ = bt_xml_path_;
    }

    bt_blackboard_ = BT::Blackboard::create();
    tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromFile(resolved_bt_xml_path_, bt_blackboard_));

    if (enable_groot_monitoring_ && tree_) {
      groot_publisher_ = std::make_unique<BT::PublisherZMQ>(
        *tree_, 25, static_cast<uint16_t>(groot_publisher_port_), static_cast<uint16_t>(groot_server_port_));
    }
  }

  void update_bt_blackboard()
  {
    if (!bt_blackboard_) return;

    bt_blackboard_->set("has_enemy", current_selected_target_.has_target);
    bt_blackboard_->set("referee_fresh", !last_blackboard_.referee_stale);
    bt_blackboard_->set("wireless_fresh", !last_blackboard_.wireless_stale);
    bt_blackboard_->set("hero_alert", evaluate_hero_alert());
    bt_blackboard_->set("hero_lob_action", current_hero_lob_fsm_output_.action_needed);
    bt_blackboard_->set("double_forward", current_fsm_output_.should_forward);
    bt_blackboard_->set(
      "radar_cmd", current_fsm_output_.should_send_radar_command || enable_interference_key_command_);

    if (current_selected_target_.has_target) {
      bt_blackboard_->set("target_id", static_cast<int>(current_selected_target_.primary_target.robot_id));
      bt_blackboard_->set("target_score", current_selected_target_.threat_score);
      bt_blackboard_->set("target_reason", current_selected_target_.reason);
    }
  }

  void tick_rule_mode()
  {
    (void)action_publish_map();
    (void)action_send_sentry_alert();
    (void)action_send_sentry_field_state();
    (void)action_send_hero_alert();
    (void)action_send_hero_lob_assist();
    (void)action_send_double_forward();
    (void)action_send_radar_decision();
  }

  void tick_btcpp_mode()
  {
    if (tree_) {
      tree_->tickRoot();
    }
  }

  // 主決策循环函数
  void on_timer()
  {
    if (!enabled_ || !have_blackboard_) {
      return;
    }
    // 更新時間戳
    current_tick_now_ = Clock::now();
    update_match_start_counter_reset();
    update_interference_key_votes(current_tick_now_);
    // 選擇雙倍易傷目標
    current_selected_target_ = TargetSelector::select(last_blackboard_, selector_cfg_);
    // 更新雙倍易傷FSM輸出
    current_fsm_output_ = double_effect_fsm_.tick(build_double_effect_input());
    // 英雄抛射辅助更新
    update_hero_lob_assist();
    // 行爲樹黑板更新
    update_bt_blackboard();

    if (decision_mode_ == "rule") {
      tick_rule_mode();
    } else {
      tick_btcpp_mode();
    }
    save_radar_cmd_counter_state(false);
  }

  bool enabled_{true};
  bool have_blackboard_{false};

  bool enable_sentry_alert_angle_{true};
  bool enable_sentry_field_state_{true};
  bool enable_hero_alert_flag_{true};
  bool enable_hero_lob_assist_{false};
  bool enable_double_effect_forward_{true};
  bool enable_radar_double_effect_command_{true};
  bool enable_interference_key_command_{true};
  bool derive_ids_from_my_color_{true};
  bool use_referee_identity_color_{true};

  int my_hero_id_{1};
  int my_sentinel_id_{7};
  int infantry_3_id_{3};
  int infantry_4_id_{4};
  int enemy_hero_id_{101};
  int hero_lob_receiver_id_{1};
  int hero_lob_data_cmd_id_{static_cast<int>(interaction_cmd_ids::kHeroLobAssist)};

  int map_publish_period_ms_{200};
  int map_keepalive_ms_{1000};
  int sentry_alert_keepalive_ms_{1000};
  int sentry_field_keepalive_ms_{1000};
  int hero_alert_keepalive_ms_{1000};
  int hero_lob_keepalive_ms_{500};
  int hero_lob_enter_quiet_ms_{500};
  int hero_lob_clear_repeat_{3};
  int hero_lob_solve_period_ms_{200};
  int double_effect_keepalive_ms_{1500};
  int radar_double_effect_keepalive_ms_{1500};
  int interference_key_vote_window_ms_{10000};
  int interference_key_update_period_ms_{1000};
  int radar_cmd_counter_save_period_ms_{1000};
  int hero_lob_min_samples_{10};

  int low_hp_alert_threshold_{120};
  int low_ammo_threshold_{10};

  double decision_rate_hz_{50.0};
  double buff_bonus_score_{8.0};
  double macro_bonus_score_{3.0};
  double hero_lob_shooter_height_m_{0.30};
  double hero_lob_window_s_{2.0};
  double hero_lob_max_avg_displacement_m_{0.60};
  double hero_lob_enemy_block_radius_m_{3.0};
  double hero_lob_min_enemy_confidence_{0.50};
  bool hero_lob_require_enemy_valid_{true};
  bool hero_lob_require_enemy_direct_source_{false};
  double hero_lob_muzzle_speed_mps_{0.0};
  double hero_lob_drag_k_{0.0};
  double hero_lob_yaw_sigma_deg_{0.0};
  double hero_lob_pitch_sigma_deg_{0.0};
  double hero_lob_yaw_bias_deg_{0.0};
  double hero_lob_pitch_bias_deg_{0.0};
  double hero_lob_integrate_dt_s_{0.005};
  double hero_lob_max_flight_time_s_{5.0};
  double hero_lob_yaw_refine_range_deg_{1.0};
  double hero_lob_yaw_refine_step_deg_{0.1};
  double hero_lob_pitch_search_min_deg_{15.0};
  double hero_lob_pitch_search_max_deg_{75.0};
  double hero_lob_pitch_search_step_deg_{0.2};

  std::string decision_mode_{"btcpp"};
  std::string my_color_{"Red"};
  std::string identity_topic_{"/radar/referee/robot_identity"};
  std::string bt_xml_path_;
  std::string bt_models_path_{"/tmp/radar_decision_nodes.xml"};
  std::string resolved_bt_xml_path_;
  bool enable_groot_monitoring_{true};
  int groot_publisher_port_{1666};
  int groot_server_port_{1667};

  std::string output_map_request_topic_;
  std::string output_robot_interaction_topic_;
  std::string output_radar_command_topic_;
  std::string output_hero_lob_status_topic_;
  std::string output_hero_lob_debug_topic_;
  std::string radar_cmd_counter_state_path_{"/tmp/radar_decision_0121_counter.txt"};
  std::vector<double> hero_lob_target_xyz_when_red_;
  std::vector<double> hero_lob_target_xyz_when_blue_;

  TargetSelectorConfig selector_cfg_;
  TargetSelectorResult current_selected_target_;
  DoubleEffectFSM double_effect_fsm_;
  DoubleEffectFSMOutput current_fsm_output_;
  HeroLobWindow hero_lob_window_;
  HeroLobBallistics hero_lob_ballistics_;
  HeroLobFSM hero_lob_fsm_;
  HeroLobWindowOutput current_hero_lob_window_output_;
  HeroLobFSMOutput current_hero_lob_fsm_output_;
  HeroLobSolution current_hero_lob_solution_;
  radar_decision_comm::msg::TrackedRobot current_hero_track_;
  geometry_msgs::msg::Point current_hero_lob_shooter_point_;
  geometry_msgs::msg::Point current_hero_lob_target_point_;
  double current_nearest_enemy_distance_m_{std::numeric_limits<double>::infinity()};
  uint16_t current_nearest_enemy_id_{0};
  bool current_hero_found_{false};
  bool current_hero_track_fresh_{false};
  bool current_enemy_nearby_3m_{false};
  bool current_hero_lob_solution_usable_{false};
  uint8_t hero_lob_clear_sequence_{0};
  uint8_t radar_cmd_counter_{0};
  bool double_effect_request_latched_{false};
  int last_game_progress_{-1};
  TimePoint current_tick_now_{};
  TimePoint last_radar_cmd_counter_save_tp_{};
  bool have_last_radar_cmd_counter_save_tp_{false};
  TimePoint last_hero_lob_solve_tp_{};
  bool have_hero_lob_solve_tp_{false};
  std::optional<TimePoint> hero_lob_ready_since_;
  std::deque<InterferenceKeySample> interference_key_samples_;
  TimePoint interference_key_window_start_{};
  bool interference_key_window_active_{false};
  TimePoint last_interference_update_tp_{};
  bool have_last_interference_update_tp_{false};
  TimePoint last_interference_verify_tp_{};
  bool have_last_interference_verify_tp_{false};
  int64_t last_interference_key_stamp_ns_{0};
  std::mt19937 rng_{std::random_device{}()};

  radar_decision_comm::msg::BattleBlackboard last_blackboard_;

  MapState map_state_;
  std::unordered_map<std::string, InteractionState> interaction_states_;
  RadarCmdState radar_cmd_state_{};

  BT::BehaviorTreeFactory factory_;
  BT::Blackboard::Ptr bt_blackboard_;
  std::unique_ptr<BT::Tree> tree_;
  std::unique_ptr<BT::PublisherZMQ> groot_publisher_;

  rclcpp::Publisher<radar_decision_comm::msg::MapPublishRequest>::SharedPtr map_request_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr robot_interaction_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::RadarDecisionCommand>::SharedPtr radar_command_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::HeroLobAssistStatus>::SharedPtr hero_lob_status_pub_;
  rclcpp::Publisher<radar_decision_comm::msg::HeroLobAssistDebug>::SharedPtr hero_lob_debug_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::BattleBlackboard>::SharedPtr blackboard_sub_;
  rclcpp::Subscription<radar_decision_comm::msg::RefereeRobotIdentity>::SharedPtr identity_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::DecisionEngineComponent)
