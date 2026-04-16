cmake_minimum_required(VERSION 3.8)
project(auto_aim_fast)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(radar_decision_comm REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# =========================================================
# OpenCV / CUDA selection
# Prefer user-installed CUDA-enabled OpenCV if available.
# =========================================================
option(AUTO_AIM_FAST_USE_CUDA "Enable OpenCV CUDA preprocess" ON)
option(AUTO_AIM_FAST_USE_TRT  "Enable TensorRT YOLO detector" ON)

if(NOT DEFINED OpenCV_DIR)
  set(OpenCV_DIR "$ENV{HOME}/opencv_cuda_install/lib/cmake/opencv4" CACHE PATH "OpenCV config path")
endif()

message(STATUS "AUTO_AIM_FAST_USE_CUDA = ${AUTO_AIM_FAST_USE_CUDA}")
message(STATUS "AUTO_AIM_FAST_USE_TRT  = ${AUTO_AIM_FAST_USE_TRT}")
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")

set(AUTO_AIM_FAST_ENABLE_CUDA OFF)

if(AUTO_AIM_FAST_USE_CUDA)
  find_package(OpenCV QUIET COMPONENTS core imgproc cudaimgproc cudaarithm cudafilters highgui calib3d)

  if(OpenCV_FOUND)
    message(STATUS "OpenCV CUDA modules found, enabling GPU preprocess")
    set(AUTO_AIM_FAST_ENABLE_CUDA ON)
    add_compile_definitions(AUTO_AIM_FAST_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "OpenCV CUDA modules not found, fallback to CPU OpenCV")
    set(AUTO_AIM_FAST_USE_CUDA OFF)
    unset(OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)
  endif()
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# =========================================================
# TensorRT
# =========================================================
set(AUTO_AIM_FAST_ENABLE_TRT OFF)
if(AUTO_AIM_FAST_USE_TRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/include /usr/include)
  find_library(TENSORRT_NVINFER_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)

  if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR} | ${TENSORRT_NVINFER_LIB}")
    set(AUTO_AIM_FAST_ENABLE_TRT ON)
    add_compile_definitions(AUTO_AIM_FAST_USE_TRT)

    if(NOT AUTO_AIM_FAST_ENABLE_CUDA)
      enable_language(CUDA)
      find_package(CUDAToolkit REQUIRED)
      set(AUTO_AIM_FAST_ENABLE_CUDA ON)
    endif()
  else()
    message(WARNING "TensorRT not found, YOLO TensorRT detector will be disabled")
  endif()
endif()

# -----------------------------
# Messages
# -----------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AimTarget.msg"
  "msg/DebugStats.msg"
  DEPENDENCIES builtin_interfaces
)

# -----------------------------
# Core library sources
# -----------------------------
set(AUTO_AIM_FAST_CORE_SOURCES
  src/pipeline/pipeline.cpp
  src/util/time.cpp
  src/util/ring_buffer.cpp

  src/vision/preprocess.cpp
  src/vision/light_extractor.cpp
  src/vision/pairing.cpp
  src/vision/refine.cpp

  src/pose/pnp_solver.cpp
  src/pose/pose_quality.cpp

  src/tracking/target_tracker.cpp
  src/tracking/data_association.cpp

  src/prediction/latency_comp.cpp
  src/prediction/ballistic.cpp
  src/prediction/aim_solution.cpp

  src/legacy/detector.cpp
  src/legacy/solve_pnp.cpp
)

if(AUTO_AIM_FAST_ENABLE_CUDA)
  list(APPEND AUTO_AIM_FAST_CORE_SOURCES
    src/legacy/detect_color_cuda.cu
  )
endif()

if(AUTO_AIM_FAST_ENABLE_TRT)
  list(APPEND AUTO_AIM_FAST_CORE_SOURCES
    src/legacy/yolo_trt_detector.cpp
    src/legacy/yolo_preprocess_cuda.cu
  )
endif()

# -----------------------------
# Core library
# -----------------------------
add_library(auto_aim_fast_core ${AUTO_AIM_FAST_CORE_SOURCES})

target_include_directories(auto_aim_fast_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

if(AUTO_AIM_FAST_ENABLE_TRT)
  target_include_directories(auto_aim_fast_core PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()

target_link_libraries(auto_aim_fast_core
  ${OpenCV_LIBS}
)

if(AUTO_AIM_FAST_ENABLE_CUDA)
  target_link_libraries(auto_aim_fast_core CUDA::cudart)
  set_target_properties(auto_aim_fast_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 89
  )
endif()

if(AUTO_AIM_FAST_ENABLE_TRT)
  target_link_libraries(auto_aim_fast_core ${TENSORRT_NVINFER_LIB})
endif()

ament_target_dependencies(auto_aim_fast_core
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

# -----------------------------
# Node executable
# -----------------------------
add_executable(auto_aim_node
  src/main.cpp
  src/node/auto_aim_node.cpp
)

target_include_directories(auto_aim_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${radar_decision_comm_INCLUDE_DIRS}
)

target_link_libraries(auto_aim_node
  auto_aim_fast_core
)

ament_target_dependencies(auto_aim_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  builtin_interfaces
  cv_bridge
  radar_decision_comm
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(auto_aim_node ${cpp_typesupport_target})

# -----------------------------
# Install
# -----------------------------
install(TARGETS
  auto_aim_fast_core
  auto_aim_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config msg
  DESTINATION share/${PROJECT_NAME}
)

# -----------------------------
# Testing
# -----------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
