#include <rclcpp/rclcpp.hpp>
#include "auto_aim_fast/node/auto_aim_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  cv::setNumThreads(3);
  rclcpp::spin(std::make_shared<auto_aim_fast::AutoAimNode>());
  rclcpp::shutdown();
  return 0;
}
