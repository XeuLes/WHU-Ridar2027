cmake_minimum_required(VERSION 3.10)
project(livox_cloud_integrator)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_library(cloud_integrator_component SHARED
  src/cloud_integrator.cpp
)
ament_target_dependencies(cloud_integrator_component
  rclcpp rclcpp_components sensor_msgs
)
rclcpp_components_register_nodes(cloud_integrator_component
  "livox_cloud_integrator::CloudIntegrator"
)

add_executable(cloud_integrator_node
  src/main.cpp
)
ament_target_dependencies(cloud_integrator_node
  rclcpp sensor_msgs
)
target_link_libraries(cloud_integrator_node
  cloud_integrator_component
)

install(TARGETS
  cloud_integrator_component
  cloud_integrator_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
