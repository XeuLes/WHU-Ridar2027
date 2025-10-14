cmake_minimum_required(VERSION 3.8)
project(debug_map)

# 1. 自动依赖管理（复用kalman_filter的简洁逻辑，无需手动列所有依赖）
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 2. 显式依赖OpenCV（核心依赖，需单独声明以链接其库）
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

# 3. 生成组件库（ament_auto_add_library自动关联依赖，无需手动ament_target_dependencies）
ament_auto_add_library(debug_map_component SHARED
  src/debug_map.cpp  # 仅保留核心源文件，简化路径管理
)

# 4. 链接OpenCV库（kalman_filter中也有target_link_libraries，保持一致）
target_link_libraries(debug_map_component
  ${OpenCV_LIBRARIES}
)

# 5. 包含目录（按需保留，若项目无include目录可删除此段）
target_include_directories(debug_map_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 6. 组件注册（参考kalman_filter的简洁格式，保留日志便于调试）
message(STATUS "Registering debug_map component: debug_map::DebugMap")
rclcpp_components_register_nodes(debug_map_component "debug_map::DebugMap")

# 7. 安装配置文件（与kalman_filter的资源安装逻辑一致）
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# 8. 自动包配置（关键！ament_auto_package替代手动ament_package，自动处理插件文件安装）
# （无需手动install插件文件：ament_cmake_auto会自动识别rclcpp_components生成的.xml并安装到正确路径）
ament_auto_package()
