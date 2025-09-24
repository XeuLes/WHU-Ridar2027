cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)  # 解决PCL_ROOT变量忽略的警告
project(dynamic_cloud)
set(CMAKE_BUILD_TYPE "RELEASE")  # 编译模式：RELEASE（优化）或DEBUG（调试）

# 2. PCL依赖配置：显式查找PCL，关联头文件、库路径
find_package(PCL REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 3. ROS 2依赖自动配置：ament_cmake_auto简化依赖管理
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()  # 自动关联package.xml中的依赖

# 4. 项目头文件路径
include_directories(include)

# 5. 源码与库配置：收集src目录下所有cpp文件，生成共享库
set(DYNAMIC_CLOUD_LIB "dynamic_cloud")
file(GLOB_RECURSE DYNAMIC_CLOUD_SRCS "src/*.cpp")
ament_auto_add_library(${DYNAMIC_CLOUD_LIB} SHARED
    ${DYNAMIC_CLOUD_SRCS}
)
target_link_libraries(${DYNAMIC_CLOUD_LIB} ${PCL_LIBRARIES})

# 6. 注册ROS 2节点：将库封装为组件，生成可执行节点
rclcpp_components_register_node(${DYNAMIC_CLOUD_LIB}
    PLUGIN "tdt_radar::DynamicCloud"  # 需与代码中的节点类命名空间一致
    EXECUTABLE ${DYNAMIC_CLOUD_LIB}_node
)
target_link_libraries(${DYNAMIC_CLOUD_LIB}_node ${PCL_LIBRARIES})

# 7. 安装配置：修正路径（核心修改部分）
# 7.1 安装launch目录（原逻辑保留，路径正确）
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/  # ROS 2标准路径：install/dynamic_cloud/share/dynamic_cloud/launch
)

# 7.2 安装头文件（消除警告，按ROS 2推荐路径安装）
install(DIRECTORY
  include/
  DESTINATION include/${PROJECT_NAME}/  # 推荐路径：install/dynamic_cloud/include/dynamic_cloud/
)

# 7.3 安装config.rviz（核心修正：用ROS 2标准路径替代CATKIN变量）
install(
    DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.rviz"
  PATTERN "*.yaml"
  PATTERN "*.xml"
  PATTERN ".git" EXCLUDE  # 排除版本控制文件
  PATTERN "*~" EXCLUDE    # 排除临时文件
)
# 8. 消除头文件安装路径警告（原逻辑保留）
set(USE_SCOPED_HEADER_INSTALL_DIR ON)

# 9. 完成ROS 2包配置
ament_auto_package()
