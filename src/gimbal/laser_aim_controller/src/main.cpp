#include <rclcpp/rclcpp.hpp>
#include "laser_aim_controller/laser_aim_controller_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<laser_aim_controller::LaserAimControllerNode>());
  rclcpp::shutdown();
  return 0;
}