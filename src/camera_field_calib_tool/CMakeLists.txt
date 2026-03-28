cmake_minimum_required(VERSION 3.8)
project(camera_field_calib_tool)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------
# Dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# -----------------------------
# Core library
# -----------------------------
add_library(camera_field_calib_tool_core
  src/yaml_io.cpp
  src/camera_model.cpp
  src/ray_plane_intersector.cpp
  src/reprojection_metrics.cpp
  src/interactive_crosshair.cpp
  src/interactive_polygon.cpp
  src/pnp_calibrator.cpp
)

target_include_directories(camera_field_calib_tool_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(camera_field_calib_tool_core
  rclcpp
)

target_link_libraries(camera_field_calib_tool_core
  yaml-cpp
  ${OpenCV_LIBS}
)

# -----------------------------
# Executables
# -----------------------------
add_executable(pnp_calib_node
  src/pnp_calib_node.cpp
)
target_link_libraries(pnp_calib_node camera_field_calib_tool_core)
ament_target_dependencies(pnp_calib_node rclcpp sensor_msgs cv_bridge image_transport)
target_link_libraries(pnp_calib_node ${OpenCV_LIBS} yaml-cpp)

add_executable(pnp_calib_node_new
  src/pnp_calib_node_new.cpp
)
target_link_libraries(pnp_calib_node_new camera_field_calib_tool_core)
ament_target_dependencies(pnp_calib_node_new rclcpp sensor_msgs cv_bridge image_transport)
target_link_libraries(pnp_calib_node_new ${OpenCV_LIBS} yaml-cpp)

add_executable(plane_polygon_node
  src/plane_polygon_node.cpp
)
target_link_libraries(plane_polygon_node camera_field_calib_tool_core)
ament_target_dependencies(plane_polygon_node rclcpp sensor_msgs cv_bridge image_transport)
target_link_libraries(plane_polygon_node ${OpenCV_LIBS} yaml-cpp)
# -----------------------------
# Install / Export
# -----------------------------
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  TARGETS
    camera_field_calib_tool_core
  EXPORT export_camera_field_calib_tool_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executables in ROS2 conventional location
install(
  TARGETS
    pnp_calib_node
    pnp_calib_node_new
    plane_polygon_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_camera_field_calib_tool_core HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

# -----------------------------
# Testing
# -----------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmakeopyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
