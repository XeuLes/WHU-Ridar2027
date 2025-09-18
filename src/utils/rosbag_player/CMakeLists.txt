cmake_minimum_required(VERSION 3.8)
project(rosbag_player)

# 解决PCL_ROOT警告
cmake_policy(SET CMP0074 NEW)

# 编译器选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++标准配置
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# 依赖查找（关键修正）
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 显式查找所有必要依赖
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)  # 新增cv_bridge依赖
find_package(vision_interface REQUIRED)

# PCL配置
find_package(PCL REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV配置
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
include_directories(${OpenCV_INCLUDE_DIRS})

# 编译目标配置
set(ROSBAG_PLAYER_TARGET "rosbag_player")
# 精确指定源文件（避免引入无关文件）
file(GLOB ROSBAG_PLAYER_SRCS 
  "rosbag_player.cpp"
)

# 生成共享库（修正链接目标）
ament_auto_add_library(${ROSBAG_PLAYER_TARGET} SHARED
  ${ROSBAG_PLAYER_SRCS}
)
target_link_libraries(${ROSBAG_PLAYER_TARGET}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  rclcpp::rclcpp  # 显式链接rclcpp
  rclcpp_components::component  # 修正为正确的组件目标
  rosbag2_cpp::rosbag2_cpp  # 链接rosbag2
  cv_bridge::cv_bridge  # 链接cv_bridge
)

# 注册组件节点
rclcpp_components_register_node(${ROSBAG_PLAYER_TARGET}
  PLUGIN "RosbagPlayer"
  EXECUTABLE ${ROSBAG_PLAYER_TARGET}_node
)

# 链接可执行文件（同步修正）
target_link_libraries(${ROSBAG_PLAYER_TARGET}_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  rclcpp::rclcpp
  rclcpp_components::component
  rosbag2_cpp::rosbag2_cpp
  cv_bridge::cv_bridge
)

# 安装配置
install(TARGETS
  ${ROSBAG_PLAYER_TARGET}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  ${ROSBAG_PLAYER_TARGET}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_auto_package()
