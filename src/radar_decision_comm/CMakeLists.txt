cmake_minimum_required(VERSION 3.10)
project(radar_decision_comm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(vision_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)

file(GLOB MSG_FILES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "msg/*.msg")
file(GLOB SRV_FILES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "srv/*.srv")

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

file(GLOB_RECURSE ALL_CPP CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(CORE_SOURCES ${ALL_CPP})

foreach(EXE_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/common/protocol_selfcheck.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/common/wireless_protocol_selfcheck.cpp"
)
  if(EXISTS "${EXE_SRC}")
    list(REMOVE_ITEM CORE_SOURCES "${EXE_SRC}")
  endif()
endforeach()

add_library(${PROJECT_NAME}_core SHARED
  ${CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  rclcpp_components
  ament_index_cpp
  std_msgs
  geometry_msgs
  builtin_interfaces
  vision_interface
  behaviortree_cpp_v3
)

target_link_libraries(${PROJECT_NAME}_core
  ${cpp_typesupport_target}
  BT::behaviortree_cpp_v3
)

rclcpp_components_register_nodes(${PROJECT_NAME}_core
  "radar_decision_comm::ClientBridgeComponent"
  "radar_decision_comm::DecisionEngineComponent"
  "radar_decision_comm::FusionComponent"
  "radar_decision_comm::KalmanTracksAdapterComponent"
  "radar_decision_comm::PerceptionAdapterComponent"
  "radar_decision_comm::RefereeSerialBridgeComponent"
  "radar_decision_comm::RadarWirelessRxComponent"
  "radar_decision_comm::RefereeSerialRxComponent"
  "radar_decision_comm::RefereeSerialTxComponent"
  "radar_decision_comm::StateBlackboardComponent"
  "radar_decision_comm::TxSchedulerComponent"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/common/protocol_selfcheck.cpp")
  add_executable(radar_protocol_selfcheck
    src/common/protocol_selfcheck.cpp
  )
  target_link_libraries(radar_protocol_selfcheck
    ${PROJECT_NAME}_core
  )
  install(TARGETS radar_protocol_selfcheck
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/common/wireless_protocol_selfcheck.cpp")
  add_executable(wireless_protocol_selfcheck
    src/common/wireless_protocol_selfcheck.cpp
  )
  target_link_libraries(wireless_protocol_selfcheck
    ${PROJECT_NAME}_core
  )
  install(TARGETS wireless_protocol_selfcheck
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(TARGETS ${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)

ament_package()
