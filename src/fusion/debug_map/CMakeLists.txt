cmake_minimum_required(VERSION 3.8)
project(debug_map)

find_package (ament_cmake_auto REQUIRED) 
include_directories(
  /opt/ros/humble/include  # 强制指定ROS 2头文件根目录
  /opt/ros/humble/include/cv_bridge  # 直接指定cv_bridge的头文件目录
)
ament_auto_find_build_dependencies ()

ament_auto_add_executable(debug_map debug_map.cpp)

ament_auto_package ()