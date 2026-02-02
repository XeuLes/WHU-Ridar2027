#include <rclcpp/rclcpp.hpp>
#include "livox_cloud_integrator/cloud_integrator.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<livox_cloud_integrator::CloudIntegrator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
