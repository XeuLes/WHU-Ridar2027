cmake_minimum_required(VERSION 3.8)
project(gimbal_aim_solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(auto_aim_fast REQUIRED)

add_executable(gimbal_aim_solver_node
  src/gimbal_aim_solver_node.cpp
)

ament_target_dependencies(gimbal_aim_solver_node
  rclcpp
  tf2
  geometry_msgs
  std_msgs
  auto_aim_fast
)

install(TARGETS gimbal_aim_solver_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
