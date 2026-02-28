#include <rclcpp/rclcpp.hpp>
#include "gimbal_aim_solver/gimbal_aim_solver_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gimbal_aim_solver::GimbalAimSolverNode>());
  rclcpp::shutdown();
  return 0;
}