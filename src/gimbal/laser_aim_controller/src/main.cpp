#include <rclcpp/rclcpp.hpp>
#include "laser_aim_controller/laser_aim_controller_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto node = std::make_shared<laser_aim_controller::LaserAimControllerNode>(options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}