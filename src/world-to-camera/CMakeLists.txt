cmake_minimum_required(VERSION 3.8)
project(world_to_camera_calib)

# 设置C++标准
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 默认C++标准为C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 寻找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 包含目录
include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 编译节点
add_executable(world_to_camera_node src/world_to_camera_node.cpp)
target_link_libraries(world_to_camera_node
  ${OpenCV_LIBS}
)

# 添加依赖
ament_target_dependencies(world_to_camera_node
  rclcpp
  tf2
  tf2_ros
  tf2_eigen
  Eigen3
  nlohmann_json
  ament_index_cpp
)

# 安装可执行文件
install(TARGETS
  world_to_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装launch文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# 安装config目录
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# 测试配置必须在ament_package()之前
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 下面的宏会设置所有常用的linter
  ament_lint_auto_find_test_dependencies()
endif()

# 导出依赖
ament_export_dependencies(
  rclcpp
  tf2
  tf2_ros
  tf2_eigen
  Eigen3
  OpenCV
  nlohmann_json
  ament_index_cpp
)

ament_package()