#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
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
#include "radar_decision_comm/decision/double_effect_fsm.hpp"
#include "radar_decision_comm/decision/target_selector.hpp"
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
    decision_mode_ = this->declare_parameter<std::string>("decision_mode", "btcpp");   // rule / btcpp

    my_color_ = this->declare_parameter<std::string>("my_color", "Red");
    my_sentinel_id_ = this->declare_parameter<int>("my_sentinel_id", 7);
    infantry_3_id_ = this->declare_parameter<int>("infantry_3_id", 3);
    infantry_4_id_ = this->declare_parameter<int>("infantry_4_id", 4);
    enemy_hero_id_ = this->declare_parameter<int>("enemy_hero_id", 101);

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
    enable_double_effect_forward_ = this->declare_parameter<bool>("enable_double_effect_forward", true);
    enable_radar_double_effect_command_ = this->declare_parameter<bool>("enable_radar_double_effect_command", true);

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

    blackboard_sub_ = this->create_subscription<radar_decision_comm::msg::BattleBlackboard>(
      "/radar/fusion/battle_blackboard", 10,
      [this](const radar_decision_comm::msg::BattleBlackboard::SharedPtr msg)
      {
        last_blackboard_ = *msg;
        have_blackboard_ = true;
      });

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
    std::string note;
    TimePoint last_sent_tp{};
    bool ever_sent{false};
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
    }
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

  uint16_t quadrant_from_xy(double x, double y) const
  {
    const bool right = x >= 14.0;
    const bool upper = y >= 7.5;
    if (!right && upper) return 1;
    if (right && upper) return 2;
    if (!right && !upper) return 3;
    return 4;
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
    return true;
  }

  bool evaluate_hero_alert() const
  {
    if (!last_blackboard_.referee_stale && last_blackboard_.mark_progress.hero >= 100) {
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
      (!last_blackboard_.referee_stale && last_blackboard_.mark_progress.hero >= 100);
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
    if (!enable_radar_double_effect_command_ || !current_fsm_output_.should_send_radar_command) {
      return false;
    }

    radar_decision_comm::msg::RadarDecisionCommand cmd;
    cmd.header.stamp = this->now();
    cmd.command_type = 1;
    cmd.command_value = current_fsm_output_.radar_command_times;

    std::ostringstream note;
    note << "btcpp_radar_double_effect";
    if (current_selected_target_.has_target) {
      note << "|target=" << current_selected_target_.primary_target.robot_id;
      note << "|reason=" << current_selected_target_.reason;
    } else {
      note << "|no_target";
    }
    cmd.note = note.str();

    return maybe_publish_radar_command(cmd, radar_double_effect_keepalive_ms_, current_tick_now_);
  }

  void register_bt_nodes()
  {
    factory_.registerSimpleAction("Noop", []() { return BT::NodeStatus::SUCCESS; });

    factory_.registerSimpleAction("PublishMap", [this]() {
      (void)action_publish_map();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendSentryAlertAngle", [this]() {
      (void)action_send_sentry_alert();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendSentryFieldState", [this]() {
      (void)action_send_sentry_field_state();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendHeroAlertFlag", [this]() {
      (void)action_send_hero_alert();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendDoubleEffectForward", [this]() {
      (void)action_send_double_forward();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleAction("SendRadarDecision", [this]() {
      (void)action_send_radar_decision();
      return BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("HasEnemy", [this]() {
      return current_selected_target_.has_target ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("RefereeFresh", [this]() {
      return last_blackboard_.referee_stale ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("WirelessFresh", [this]() {
      return last_blackboard_.wireless_stale ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    });

    factory_.registerSimpleCondition("HeroAlertNeeded", [this]() {
      return evaluate_hero_alert() ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("DoubleEffectForwardNeeded", [this]() {
      return current_fsm_output_.should_forward ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory_.registerSimpleCondition("RadarCommandNeeded", [this]() {
      return current_fsm_output_.should_send_radar_command ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });
  }

  void init_bt()
  {
    register_bt_nodes();

    std::ofstream(bt_models_path_) << BT::writeTreeNodesModelXML(factory_);

    if (bt_xml_path_.empty()) {
      resolved_bt_xml_path_ =
        ament_index_cpp::get_package_share_directory("radar_decision_comm") + "/bt/decision_tree.xml";
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
    bt_blackboard_->set("double_forward", current_fsm_output_.should_forward);
    bt_blackboard_->set("radar_cmd", current_fsm_output_.should_send_radar_command);

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
    (void)action_send_double_forward();
    (void)action_send_radar_decision();
  }

  void tick_btcpp_mode()
  {
    if (tree_) {
      tree_->tickRoot();
    }
  }

  void on_timer()
  {
    if (!enabled_ || !have_blackboard_) {
      return;
    }

    current_tick_now_ = Clock::now();
    current_selected_target_ = TargetSelector::select(last_blackboard_, selector_cfg_);
    current_fsm_output_ = double_effect_fsm_.tick(build_double_effect_input());
    update_bt_blackboard();

    if (decision_mode_ == "rule") {
      tick_rule_mode();
    } else {
      tick_btcpp_mode();
    }
  }

  bool enabled_{true};
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

  int map_publish_period_ms_{200};
  int map_keepalive_ms_{1000};
  int sentry_alert_keepalive_ms_{1000};
  int sentry_field_keepalive_ms_{1000};
  int hero_alert_keepalive_ms_{1000};
  int double_effect_keepalive_ms_{1500};
  int radar_double_effect_keepalive_ms_{1500};

  int low_hp_alert_threshold_{120};
  int low_ammo_threshold_{10};

  double decision_rate_hz_{50.0};
  double buff_bonus_score_{8.0};
  double macro_bonus_score_{3.0};

  std::string decision_mode_{"btcpp"};
  std::string my_color_{"Red"};
  std::string bt_xml_path_;
  std::string bt_models_path_{"/tmp/radar_decision_nodes.xml"};
  std::string resolved_bt_xml_path_;
  bool enable_groot_monitoring_{true};
  int groot_publisher_port_{1666};
  int groot_server_port_{1667};

  std::string output_map_request_topic_;
  std::string output_robot_interaction_topic_;
  std::string output_radar_command_topic_;

  TargetSelectorConfig selector_cfg_;
  TargetSelectorResult current_selected_target_;
  DoubleEffectFSM double_effect_fsm_;
  DoubleEffectFSMOutput current_fsm_output_;
  TimePoint current_tick_now_{};

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
  rclcpp::Subscription<radar_decision_comm::msg::BattleBlackboard>::SharedPtr blackboard_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::DecisionEngineComponent)
