cmake_minimum_required(VERSION 3.8)
project(gimbal_tf_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(gimbal_tf_publisher_node
  src/gimbal_tf_publisher_node.cpp
)

ament_target_dependencies(gimbal_tf_publisher_node
  rclcpp
  tf2
  tf2_ros
  geometry_msgs
  std_msgs
)

target_link_libraries(gimbal_tf_publisher_node
  yaml-cpp
)

install(TARGETS gimbal_tf_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
