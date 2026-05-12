cmake_minimum_required(VERSION 3.8)
project(debug_map)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc)

ament_auto_add_library(debug_map_component SHARED
  src/debug_map.cpp
)

ament_auto_add_executable(debug_map_homo_node
  src/debug_map_homo.cpp
)

ament_auto_add_executable(track_status_window_node
  src/track_status_window.cpp
)

ament_auto_add_executable(referee_status_window_node
  src/referee_status_window.cpp
)

target_link_libraries(debug_map_component
  ${OpenCV_LIBRARIES}
)

target_link_libraries(debug_map_homo_node
  ${OpenCV_LIBRARIES}
)

target_link_libraries(track_status_window_node
  ${OpenCV_LIBRARIES}
)

target_link_libraries(referee_status_window_node
  ${OpenCV_LIBRARIES}
)

target_include_directories(debug_map_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

message(STATUS "Registering debug_map component: debug_map::DebugMap")
rclcpp_components_register_nodes(debug_map_component "debug_map::DebugMap")

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_auto_package()
