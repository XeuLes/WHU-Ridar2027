cmake_minimum_required(VERSION 3.8)
project(rosbag_player)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

find_package(ament_cmake_auto REQUIRED)
# 显式查找 cv_bridge
find_package(cv_bridge REQUIRED)
ament_auto_find_build_dependencies()

set(ROSBAG_PLAYER "rosbag_player")
file(GLOB_RECURSE ROSBAG_PLAYER_SRCS "*.cpp")

ament_auto_add_library(${ROSBAG_PLAYER} SHARED
    ${ROSBAG_PLAYER_SRCS}
)

# 链接 cv_bridge
target_link_libraries(${ROSBAG_PLAYER}
  ${PCL_LIBRARIES}
  cv_bridge::cv_bridge
)

rclcpp_components_register_node(${ROSBAG_PLAYER}
    PLUGIN "RosbagPlayer"
    EXECUTABLE ${ROSBAG_PLAYER}_node
)

# 确保节点也链接 cv_bridge
target_link_libraries(rosbag_player_node
  ${PCL_LIBRARIES}
  cv_bridge::cv_bridge
)

ament_auto_package()