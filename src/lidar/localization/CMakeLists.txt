cmake_minimum_required(VERSION 3.8)
project(localization)
set(CMAKE_BUILD_TYPE "RELEASE") 
#gicp配准需要release编译

find_package (ament_cmake_auto REQUIRED) 
ament_auto_find_build_dependencies ()

# 新增：显式查找yaml-cpp（确保链接时能找到库，ament_cmake_auto可能需辅助）
find_package(yaml-cpp REQUIRED)

set(LOCALIZATION_NODE "localization")
file(GLOB_RECURSE LOCALIZATION_NODE_SRCS "src/*.cpp")

ament_auto_add_library(${LOCALIZATION_NODE} SHARED
    ${LOCALIZATION_NODE_SRCS}
)

target_link_libraries(${LOCALIZATION_NODE}
    ${PCL_LIBRARIES}
    yaml-cpp  # 新增：链接yaml-cpp库（参数持久化必需）
)

rclcpp_components_register_node(${LOCALIZATION_NODE}
    PLUGIN "tdt_radar::Localization"
    EXECUTABLE ${LOCALIZATION_NODE}_node
)
target_link_libraries(${LOCALIZATION_NODE} ${PCL_LIBRARIES})

# 安装launch文件
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)


ament_auto_package ()
