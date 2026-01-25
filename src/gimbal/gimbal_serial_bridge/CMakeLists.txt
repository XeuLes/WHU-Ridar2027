cmake_minimum_required(VERSION 3.10)
project(gimbal_serial_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(gimbal_serial_bridge_node
  src/gimbal_serial_bridge_node.cpp
)
ament_target_dependencies(gimbal_serial_bridge_node
  rclcpp
  std_msgs
)

add_executable(xxx_test_node
  src/XXX_test_node.cpp
)
ament_target_dependencies(xxx_test_node
  rclcpp
  std_msgs
)

install(TARGETS
  gimbal_serial_bridge_node
  xxx_test_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
