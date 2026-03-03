cmake_minimum_required(VERSION 3.10)
project(laser_aim_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(auto_aim_fast REQUIRED)

add_executable(laser_aim_controller_node
  src/main.cpp
  src/angle_solver.cpp
  src/laser_geometry.cpp
  src/laser_aim_controller_node.cpp
)

target_include_directories(laser_aim_controller_node PUBLIC
  include
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(laser_aim_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  auto_aim_fast
)

install(TARGETS
  laser_aim_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()