cmake_minimum_required(VERSION 3.8)
project(radar_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


# 安装launch文件夹（包含所有启动文件）
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/camera_lidar_bag_extract.py
  src/referee_rosbag_recorder.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 radar_bringup 自身的 config 目录（如果后续补充文件会一并安装）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}
  )
endif()

# 安装仓库根目录下的通用配置，供 bringup 统一引用，避免 launch 写死工作区绝对路径
set(PROJECT_ROOT_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../config")
if(EXISTS "${PROJECT_ROOT_CONFIG_DIR}")
  install(DIRECTORY
    "${PROJECT_ROOT_CONFIG_DIR}/"
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()


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

ament_package()
