#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "radar_decision_comm/msg/robot_interaction_request.hpp"

namespace radar_decision_comm
{

class ClientBridgeComponent : public rclcpp::Node
{
public:
  explicit ClientBridgeComponent(const rclcpp::NodeOptions & options)
  : Node("client_bridge", options)
  {
    enabled_ = this->declare_parameter<bool>("enabled", false);
    mqtt_server_ip_ = this->declare_parameter<std::string>("mqtt_server_ip", "192.168.12.1");
    mqtt_server_port_ = this->declare_parameter<int>("mqtt_server_port", 3333);
    client_ip_ = this->declare_parameter<std::string>("client_ip", "192.168.12.2");
    udp_video_port_ = this->declare_parameter<int>("udp_video_port", 3334);
    proto_version_ = this->declare_parameter<int>("proto_version", 1);

    client_pub_ = this->create_publisher<radar_decision_comm::msg::RobotInteractionRequest>(
      "/radar/client/robot_interaction_request", 10);

    client_sub_ = this->create_subscription<radar_decision_comm::msg::RobotInteractionRequest>(
      "/radar/client/custom_control", 10,
      [this](const radar_decision_comm::msg::RobotInteractionRequest::SharedPtr msg)
      {
        (void)msg;
      });

    RCLCPP_INFO(
      get_logger(),
      "ClientBridgeComponent started. enabled=%s mqtt=%s:%d client_ip=%s udp_video_port=%d proto_version=%d",
      enabled_ ? "true" : "false",
      mqtt_server_ip_.c_str(),
      mqtt_server_port_,
      client_ip_.c_str(),
      udp_video_port_,
      proto_version_);
  }

private:
  bool enabled_{false};
  int mqtt_server_port_{3333};
  int udp_video_port_{3334};
  int proto_version_{1};
  std::string mqtt_server_ip_;
  std::string client_ip_;

  rclcpp::Publisher<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr client_pub_;
  rclcpp::Subscription<radar_decision_comm::msg::RobotInteractionRequest>::SharedPtr client_sub_;
};

}  // namespace radar_decision_comm

RCLCPP_COMPONENTS_REGISTER_NODE(radar_decision_comm::ClientBridgeComponent)
