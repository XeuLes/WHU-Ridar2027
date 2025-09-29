cmake_minimum_required(VERSION 3.8)
project(hk_camera_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ------------------- 新增相机驱动编译配置 -------------------
# 1. 配置海康MVS SDK路径（必须改！换成你的SDK安装路径，默认是/opt/MVS）
set(MVS_SDK_PATH "/opt/MVS")
# 告诉编译器去哪里找SDK的头文件
include_directories(${MVS_SDK_PATH}/include)
# 告诉编译器去哪里找SDK的库文件
link_directories(${MVS_SDK_PATH}/lib/64)

# 2. 编译相机节点（hk_camera_node.cpp → 生成可执行文件hk_camera_node）
add_executable(hk_camera_node src/hk_camera_node.cpp)
# 链接依赖包（告诉编译器需要哪些ROS2工具）
ament_target_dependencies(hk_camera_node
  rclcpp sensor_msgs cv_bridge image_transport OpenCV
)
# 链接海康MVS SDK库（必须加，否则找不到SDK的函数）
target_link_libraries(hk_camera_node MvCameraControl)

# 3. 安装可执行文件和配置文件（让ROS2能找到它们）
# 安装节点到lib目录（ROS2能找到的地方）
install(TARGETS
  hk_camera_node
  DESTINATION lib/${PROJECT_NAME}
)
# 安装launch和config目录到share目录（ROS2能找到的地方）
install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}/
)
# -----------------------------------------------------------


ament_package()
