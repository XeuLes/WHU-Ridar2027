cmake_minimum_required(VERSION 3.8)
project(hik_camera_ros2_driver)

###########################################################################
# 1. 基础编译配置
###########################################################################
## 指定C++14标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## 启用编译警告并视为错误（强制代码规范）
add_definitions(-Wall -Werror)

## 导出编译命令，供clangd等工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###########################################################################
# 2. ROS 2 依赖管理
###########################################################################
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

###########################################################################
# 3. 核心库构建（海康相机驱动库）
###########################################################################
## 生成驱动核心动态库
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/hik_camera_node.cpp
)

## 包含海康SDK头文件
target_include_directories(${PROJECT_NAME} PUBLIC 
  hikSDK/include
)

## 根据架构（x86_64/aarch64）链接SDK库并安装
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME} PUBLIC 
    hikSDK/lib/amd64
  )
  install(
    DIRECTORY hikSDK/lib/amd64/
    DESTINATION lib
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME} PUBLIC 
    hikSDK/lib/arm64
  )
  install(
    DIRECTORY hikSDK/lib/arm64/
    DESTINATION lib
  )
else()
  message(FATAL_ERROR "Unsupported host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

## 链接海康SDK依赖库
target_link_libraries(${PROJECT_NAME}
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
)

###########################################################################
# 4. ROS 2 节点注册
###########################################################################
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "hik_camera_ros2_driver::HikCameraRos2DriverNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

###########################################################################
# 5. 图像增强节点配置（依赖OpenCV）
###########################################################################
## 查找OpenCV依赖
find_package(OpenCV REQUIRED)

## 生成图像增强节点可执行文件
add_executable(image_enhancer_node
  src/image_enhancer_node.cpp
)

## 链接图像增强节点依赖库（ROS 2 + OpenCV）
ament_target_dependencies(image_enhancer_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

## 安装图像增强节点（路径：lib/${PROJECT_NAME}）
install(TARGETS
  image_enhancer_node
  DESTINATION lib/${PROJECT_NAME}
)

###########################################################################
# 6. 测试配置
###########################################################################
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ## 排除部分代码检查（根据项目需求）
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
    ament_cmake_flake8
  )
  ament_lint_auto_find_test_dependencies()
endif()

###########################################################################
# 7. 项目打包（安装launch/config等资源）
###########################################################################
ament_auto_package(
  INSTALL_TO_SHARE
  launch  # 启动文件目录
  config  # 配置文件目录（如相机参数、内参）
)
