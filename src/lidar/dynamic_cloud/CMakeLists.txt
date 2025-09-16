# cmake_minimum_required(VERSION 3.8)
# project(dynamic_cloud)
# set(CMAKE_BUILD_TYPE "RELEASE")
# find_package (ament_cmake_auto REQUIRED) 
# ament_auto_find_build_dependencies ()

# include_directories(include)

# set(DYNAMIC_CLOUD "dynamic_cloud")
# file(GLOB_RECURSE DYNAMIC_CLOUD_SRCS "src/*.cpp")

# ament_auto_add_library(${DYNAMIC_CLOUD} SHARED
#     ${DYNAMIC_CLOUD_SRCS}
# )

# target_link_libraries(${DYNAMIC_CLOUD} ${PCL_LIBRARIES})

# rclcpp_components_register_node(${DYNAMIC_CLOUD}
#     PLUGIN "tdt_radar::DynamicCloud"
#     EXECUTABLE ${DYNAMIC_CLOUD}_node
# )

# target_link_libraries(dynamic_cloud_node ${PCL_LIBRARIES})
# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}/
# )
# set(USE_SCOPED_HEADER_INSTALL_DIR ON)
# ament_auto_package ()

# 1. 基础配置：指定CMake最低版本、政策（消除CMP0074警告）、项目名
cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)  # 解决PCL_ROOT变量忽略的警告
project(dynamic_cloud)
set(CMAKE_BUILD_TYPE "RELEASE")  # 编译模式：RELEASE（优化）或DEBUG（调试）

# 2. PCL依赖配置：显式查找PCL，关联头文件、库路径（解决PCL相关功能警告）
find_package(PCL REQUIRED COMPONENTS common io)  # 按需添加PCL组件（common/io为基础）
include_directories(${PCL_INCLUDE_DIRS})          # 添加PCL头文件路径
link_directories(${PCL_LIBRARY_DIRS})             # 添加PCL库文件路径
add_definitions(${PCL_DEFINITIONS})               # 添加PCL编译宏（如PCL_NO_PRECOMPILE）

# 3. ROS 2依赖自动配置：ament_cmake_auto简化依赖管理（自动读取package.xml）
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()  # 自动关联package.xml中声明的ROS依赖（如tf2_sensor_msgs）

# 4. 项目头文件路径：指定当前项目的include目录（头文件直接放这里）
include_directories(include)

# 5. 源码与库配置：收集src目录下所有cpp文件，生成共享库
set(DYNAMIC_CLOUD_LIB "dynamic_cloud")  # 库名称
file(GLOB_RECURSE DYNAMIC_CLOUD_SRCS "src/*.cpp")  # 递归收集所有src下的cpp文件
ament_auto_add_library(${DYNAMIC_CLOUD_LIB} SHARED  # 生成共享库
    ${DYNAMIC_CLOUD_SRCS}
)
target_link_libraries(${DYNAMIC_CLOUD_LIB} ${PCL_LIBRARIES})  # 为库链接PCL库

# 6. 注册ROS 2节点：将库封装为组件，生成可执行节点
rclcpp_components_register_node(${DYNAMIC_CLOUD_LIB}
    PLUGIN "tdt_radar::DynamicCloud"  # 节点类的完整命名空间（需与代码一致）
    EXECUTABLE ${DYNAMIC_CLOUD_LIB}_node  # 生成的可执行文件名称（如dynamic_cloud_node）
)
target_link_libraries(${DYNAMIC_CLOUD_LIB}_node ${PCL_LIBRARIES})  # 为节点可执行文件链接PCL库

# 7. 安装配置：安装launch文件和头文件（适配情况1的头文件路径）
install(DIRECTORY
  launch  # 安装launch目录下的所有文件（如启动脚本）
  DESTINATION share/${PROJECT_NAME}/  # 安装到ROS 2的共享目录（按规范）
)
install(DIRECTORY
  include/  # 安装当前项目的include目录（头文件直接放这里，无嵌套子目录）
  DESTINATION include  # 安装到系统include目录（ROS 2默认路径）
)

# 8. 头文件安装路径兼容：消除"推荐安装到include/dynamic_cloud"的警告
set(USE_SCOPED_HEADER_INSTALL_DIR ON)

# 9. 完成ROS 2包配置：ament自动处理剩余安装（如库、节点的路径）
ament_auto_package()

