cmake_minimum_required(VERSION 3.8)
project(localization)
set(CMAKE_BUILD_TYPE "RELEASE") 
#gicp配准需要release编译

find_package (ament_cmake_auto REQUIRED) 
ament_auto_find_build_dependencies ()

# 显式查找必要依赖（确保链接时能找到库）
find_package(yaml-cpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)

set(LOCALIZATION_NODE "localization")
file(GLOB_RECURSE LOCALIZATION_NODE_SRCS "src/*.cpp")

ament_auto_add_library(${LOCALIZATION_NODE} SHARED
    ${LOCALIZATION_NODE_SRCS}
)

# 链接所有依赖库（包含新增的rosbag2相关库）
target_link_libraries(${LOCALIZATION_NODE}
    ${PCL_LIBRARIES}
    yaml-cpp
    rosbag2_cpp::rosbag2_cpp
    rosbag2_storage::rosbag2_storage
    ament_index_cpp::ament_index_cpp
)

# 添加编译选项（可选，增强兼容性）
target_compile_options(${LOCALIZATION_NODE} PRIVATE -Wall -Wextra -Wpedantic)

# 配置包含目录（确保头文件能找到）
target_include_directories(${LOCALIZATION_NODE} PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
    ${rosbag2_storage_INCLUDE_DIRS}
    ${ament_index_cpp_INCLUDE_DIRS}
)

# 注册ROS节点插件
rclcpp_components_register_node(${LOCALIZATION_NODE}
    PLUGIN "tdt_radar::Localization"
    EXECUTABLE ${LOCALIZATION_NODE}_node
)

# 安装launch文件、配置文件
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# 可选：安装rosbag示例文件（如果有）
# install(DIRECTORY
#   rosbag
#   DESTINATION share/${PROJECT_NAME}
# )

ament_auto_package ()
