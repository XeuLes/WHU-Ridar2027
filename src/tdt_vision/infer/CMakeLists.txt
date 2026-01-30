cmake_minimum_required(VERSION 3.10)

# 关键：启用必要的CMake策略（与根目录保持一致）
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0104 NEW)  # 解决CUDA_ARCHITECTURES警告

project(infer)

# 1. 编译配置（与根目录保持版本一致）
add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# 2. CUDA配置（继承根目录的架构设置，显式指定RTX 4060的8.9）
find_package(CUDA REQUIRED)
enable_language(CUDA)  # 确保启用CUDA语言支持
set(CMAKE_CUDA_ARCHITECTURES 8.7)  # 关键：匹配RTX 4060的CUDA架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")  # 显式传递给nvcc

# 3. TensorRT和其他依赖（保留原逻辑）
include(../cmake/modules/libnvinferConfig.cmake)
find_package(OpenCV REQUIRED)

# 4. 头文件目录（补充CUDA和根目录头文件）
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${nvinfer_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../utils/include  # 引入上层utils的头文件
)

# 5. 源文件和目标编译（显式指定CUDA文件处理）
file(GLOB_RECURSE CPP_SRCS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE CU_SRCS
    ${PROJECT_SOURCE_DIR}/src/*.cu
)

# ---- 关键：排除旧实现 yolo.cu（一定要命中你的实际文件名）----
list(FILTER CU_SRCS EXCLUDE REGEX ".*/yolo\\.cu$")

# ---- 关键：如果你新文件叫 yolov8.cu，就显式加入它 ----
list(APPEND CU_SRCS ${PROJECT_SOURCE_DIR}/src/yolov8.cu)

set(SRCS ${CPP_SRCS} ${CU_SRCS})
add_library(infer SHARED ${SRCS})

# 6. 确保CUDA目标使用正确的架构
set_property(TARGET infer PROPERTY CUDA_ARCHITECTURES 8.7)

# 7. 链接依赖库（补充可能的PCL依赖）
target_link_libraries(infer 
    ${nvinfer_LIBRARY} 
    cudart 
    nvonnxparser 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}  # 若infer依赖PCL，需添加此句
)

install(
  TARGETS infer  # 目标库名：infer
  EXPORT inferTargets
  LIBRARY DESTINATION lib  # 安装到 install/tdt_vision/lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)