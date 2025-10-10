# ===================== 0. 最优先：CUDA架构核心配置（解决验证失败问题）=====================
cmake_minimum_required(VERSION 3.8)

# 1. CMake策略（必须在project前）
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0104 NEW)

# 2. 明确指定CUDA架构（8.7，CUDA 11.8支持，供CMake验证用）
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA架构（数字格式，87对应8.7）" FORCE)

# 3. 强制传递nvcc编译参数（指定计算架构和运行架构，与上面的87对应）
set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_87,code=sm_87" CACHE STRING "" FORCE)

# 4. 配置CUDA编译器路径
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc CACHE PATH "" FORCE)

# 5. 启用CUDA语言（此时CMake能通过CMAKE_CUDA_ARCHITECTURES=87验证有效性）
enable_language(CUDA)


# ===================== 1. 项目初始化=====================
project(tdt_vision)


# ===================== 2. PCL配置=====================
set(PCL_DIR "/usr/local/share/pcl-1.14" CACHE PATH "" FORCE)
set(PCL_ROOT "/usr/local" CACHE PATH "" FORCE)
find_package(PCL REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# ===================== 3. 基础编译配置=====================
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)

include_directories(  
  "detect/include"
  "resolve/include"
  "calibrate/include"
)


# ===================== 4. 依赖查找=====================
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV 4.5 REQUIRED)
if(NOT OpenCV_VERSION MATCHES "4.5.*")
  message(FATAL_ERROR "需要 OpenCV 4.5.x！当前版本: ${OpenCV_VERSION}")
endif()

# CUDA头文件和库
include_directories(/usr/local/cuda-11.8/include)
link_directories(/usr/local/cuda-11.8/lib64)
find_package(CUDA REQUIRED)


# ===================== 5. TensorRT配置=====================
include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)
message("Using Cuda Version: ${CUDA_VERSION} and TensorRT Version: ${nvinfer_VERSION}")
add_definitions(-D use_tensorrt)
include_directories(${nvinfer_INCLUDE_DIRS})


# ===================== 6. 子目录编译=====================
add_subdirectory(utils)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)

add_subdirectory(infer)
# 给infer子目录设置相同的CUDA架构
set_target_properties(infer PROPERTIES 
  CUDA_ARCHITECTURES 87
  CUDA_FLAGS "-gencode=arch=compute_87,code=sm_87"
)
include_directories(${PROJECT_SOURCE_DIR}/infer/include)


# ===================== 7. 雷达模块编译=====================
set(RADAR_DETECT "radar_detect")
file(GLOB_RECURSE RADAR_DETECT_SRCS "detect/src/*.cpp")
ament_auto_add_library(${RADAR_DETECT} SHARED ${RADAR_DETECT_SRCS})

target_link_libraries(${RADAR_DETECT}
  utils
  infer 
  ${nvinfer_LIBRARY} 
  cudart 
  nvonnxparser 
  fmt 
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

rclcpp_components_register_node(${RADAR_DETECT}
  PLUGIN "tdt_radar::Detect"
  EXECUTABLE ${RADAR_DETECT}_node
)

# 其他模块（resolve/calibrate）保留原逻辑
set(RADAR_RESOLVE "radar_resolve")
file(GLOB_RECURSE RADAR_RESOLVE_SRCS "resolve/src/*.cpp")
ament_auto_add_library(${RADAR_RESOLVE} SHARED ${RADAR_RESOLVE_SRCS})
rclcpp_components_register_node(${RADAR_RESOLVE}
  PLUGIN "tdt_radar::Resolve"
  EXECUTABLE ${RADAR_RESOLVE}_node
)

set(RADAR_CALIBRATE "radar_calibrate")
file(GLOB_RECURSE RADAR_CALIBRATE_SRCS "calibrate/src/*.cpp")
ament_auto_add_library(${RADAR_CALIBRATE} SHARED ${RADAR_CALIBRATE_SRCS})
rclcpp_components_register_node(${RADAR_CALIBRATE}
  PLUGIN "tdt_radar::Calibrate"
  EXECUTABLE ${RADAR_CALIBRATE}_node
)


# ===================== 8. 安装=====================
ament_auto_package(INSTALL_TO_SHARE launch)
