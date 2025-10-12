cmake_minimum_required(VERSION 3.8)
project(debug_map)

# 1. 自动查找依赖（包含 opencv2、cv_bridge 等，从 package.xml 读取）
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 2. 显式找到 OpenCV 包（确保 CMake 能识别 OpenCV 模块，ament_auto 可能未完全处理）
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

# 3. 生成组件库
ament_auto_add_library(debug_map_component SHARED
  src/debug_map.cpp
)

# 4. 显式链接依赖（修正 OpenCV 目标）
target_link_libraries(debug_map_component
  rclcpp::rclcpp                # ROS 2 节点核心库
  rclcpp_components::component  # 组件注册核心库
  cv_bridge::cv_bridge          # cv_bridge 库（图像转换）
  ${OpenCV_LIBRARIES}           # OpenCV 所有依赖模块
)

# 5. 显式添加包含目录（确保头文件正确包含）
target_include_directories(debug_map_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 6. 注册组件类（使用正确的宏）
rclcpp_components_register_nodes(debug_map_component "tdt_radar::DebugMap")

# 7. 创建并安装插件描述文件
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/debug_map_plugin.xml
  "<library path=\"debug_map_component\">\n"
  "  <class type=\"tdt_radar::DebugMap\" base_class_type=\"rclcpp::Node\">\n"
  "    <description>Debug Map Component for T-DT Radar System</description>\n"
  "  </class>\n"
  "</library>\n"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/debug_map_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# 8. 导出库和依赖（移除有问题的导出）
ament_export_libraries(debug_map_component)
ament_export_dependencies(rclcpp rclcpp_components cv_bridge sensor_msgs vision_interface)

# 9. 安装 config 目录
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# 10. 生成包配置
ament_package()