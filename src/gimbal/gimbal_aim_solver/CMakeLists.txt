cmake_minimum_required(VERSION 3.10)
project(gimbal_aim_solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(radar_decision_comm REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${radar_decision_comm_INCLUDE_DIRS}
)

add_executable(gimbal_aim_solver_node
  src/main.cpp
  src/extrinsic_transformer.cpp
  src/gimbal_geometry.cpp
  src/angle_solver.cpp
  src/gimbal_aim_solver_node.cpp
)

ament_target_dependencies(gimbal_aim_solver_node
  rclcpp
  std_msgs
  geometry_msgs
  radar_decision_comm
)

target_link_libraries(gimbal_aim_solver_node
  ${OpenCV_LIBS}
)

add_executable(gimbal_camera_translation_debug_node
  src/gimbal_camera_translation_debug_node.cpp
  src/gimbal_geometry.cpp
  src/angle_solver.cpp
)

ament_target_dependencies(gimbal_camera_translation_debug_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
)

target_link_libraries(gimbal_camera_translation_debug_node
  ${OpenCV_LIBS}
)

install(TARGETS
  gimbal_aim_solver_node
  gimbal_camera_translation_debug_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
