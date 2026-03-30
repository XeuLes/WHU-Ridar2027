cmake_minimum_required(VERSION 3.8)
project(radar_decision_comm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedRobot.msg"
  "msg/TrackedRobotArray.msg"
  "msg/RefereeGameState.msg"
  "msg/RefereeEnemyHp.msg"
  "msg/RadarMarkProgress.msg"
  "msg/RadarDecisionSync.msg"
  "msg/DartStatus.msg"
  "msg/EnemyWorldState.msg"
  "msg/BattleBlackboard.msg"
  "msg/MapPublishRequest.msg"
  "msg/RobotInteractionRequest.msg"
  "msg/RadarDecisionCommand.msg"
  "msg/WirelessEnemyHp.msg"
  "msg/WirelessEnemyAmmo.msg"
  "msg/WirelessEnemyMacroState.msg"
  "msg/WirelessEnemyBuffs.msg"
  "msg/WirelessInterferenceKey.msg"
  "msg/TxDebugStats.msg"
  "srv/ReopenPort.srv"
  "srv/ResetState.srv"
  "srv/ForceSend.srv"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

include_directories(include)

add_library(radar_decision_comm_core SHARED
  src/codec/referee_decoder.cpp
  src/codec/referee_encoder.cpp
  src/codec/referee_packet_parser.cpp
  src/codec/interaction_payload_builder.cpp
  src/codec/map_payload_builder.cpp
  src/codec/radar_command_payload_builder.cpp
  src/codec/wireless_packet_parser.cpp
  src/state/blackboard.cpp
  src/fusion/fusion_policy.cpp
  src/decision/decision_engine.cpp
  src/decision/double_effect_fsm.cpp
  src/decision/target_selector.cpp
  src/scheduler/tx_scheduler.cpp
  src/common/serial_port.cpp
  src/common/frame_receiver.cpp
  src/common/frame_sender.cpp
  src/common/crc.cpp
)

target_compile_features(radar_decision_comm_core PUBLIC cxx_std_17)

ament_target_dependencies(
  radar_decision_comm_corerclcpp
  rclcpp_components

  ament_index_cpp
  behaviortree_cpp_v3
)

rosidl_target_interfaces(radar_decision_comm_core
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

add_library(referee_serial_rx_component SHARED src/components/referee_serial_rx_component.cpp)
target_compile_features(referee_serial_rx_component PUBLIC cxx_std_17)
ament_target_dependencies(referee_serial_rx_component rclcpp rclcpp_components)
target_link_libraries(referee_serial_rx_component radar_decision_comm_core)
rosidl_target_interfaces(referee_serial_rx_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(referee_serial_rx_component "radar_decision_comm::RefereeSerialRxComponent")

add_library(radar_wireless_rx_component SHARED src/components/radar_wireless_rx_component.cpp)
target_compile_features(radar_wireless_rx_component PUBLIC cxx_std_17)
ament_target_dependencies(radar_wireless_rx_component rclcpp rclcpp_components)
target_link_libraries(radar_wireless_rx_component radar_decision_comm_core)
rosidl_target_interfaces(radar_wireless_rx_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(radar_wireless_rx_component "radar_decision_comm::RadarWirelessRxComponent")

add_library(client_bridge_component SHARED src/components/client_bridge_component.cpp)
target_compile_features(client_bridge_component PUBLIC cxx_std_17)
ament_target_dependencies(client_bridge_component rclcpp rclcpp_components)
target_link_libraries(client_bridge_component radar_decision_comm_core)
rosidl_target_interfaces(client_bridge_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(client_bridge_component "radar_decision_comm::ClientBridgeComponent")

add_library(perception_adapter_component SHARED src/components/perception_adapter_component.cpp)
target_compile_features(perception_adapter_component PUBLIC cxx_std_17)
ament_target_dependencies(perception_adapter_component rclcpp rclcpp_components)
target_link_libraries(perception_adapter_component radar_decision_comm_core)
rosidl_target_interfaces(perception_adapter_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(perception_adapter_component "radar_decision_comm::PerceptionAdapterComponent")

add_library(state_blackboard_component SHARED src/components/state_blackboard_component.cpp)
target_compile_features(state_blackboard_component PUBLIC cxx_std_17)
ament_target_dependencies(state_blackboard_component rclcpp rclcpp_components)
target_link_libraries(state_blackboard_component radar_decision_comm_core)
rosidl_target_interfaces(state_blackboard_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(state_blackboard_component "radar_decision_comm::StateBlackboardComponent")

add_library(fusion_component SHARED src/components/fusion_component.cpp)
target_compile_features(fusion_component PUBLIC cxx_std_17)
ament_target_dependencies(fusion_component rclcpp rclcpp_components)
target_link_libraries(fusion_component radar_decision_comm_core)
rosidl_target_interfaces(fusion_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(fusion_component "radar_decision_comm::FusionComponent")

add_library(decision_engine_component SHARED src/components/decision_engine_component.cpp)
target_compile_features(decision_engine_component PUBLIC cxx_std_17)
ament_target_dependencies(decision_engine_component rclcpp rclcpp_components)
target_link_libraries(decision_engine_component radar_decision_comm_core)
rosidl_target_interfaces(decision_engine_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(decision_engine_component "radar_decision_comm::DecisionEngineComponent")

add_library(tx_scheduler_component SHARED src/components/tx_scheduler_component.cpp)
target_compile_features(tx_scheduler_component PUBLIC cxx_std_17)
ament_target_dependencies(tx_scheduler_component rclcpp rclcpp_components)
target_link_libraries(tx_scheduler_component radar_decision_comm_core)
rosidl_target_interfaces(tx_scheduler_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(tx_scheduler_component "radar_decision_comm::TxSchedulerComponent")

add_library(referee_serial_tx_component SHARED src/components/referee_serial_tx_component.cpp)
target_compile_features(referee_serial_tx_component PUBLIC cxx_std_17)
ament_target_dependencies(referee_serial_tx_component rclcpp rclcpp_components)
target_link_libraries(referee_serial_tx_component radar_decision_comm_core)
rosidl_target_interfaces(referee_serial_tx_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(referee_serial_tx_component "radar_decision_comm::RefereeSerialTxComponent")

add_executable(radar_protocol_selfcheck src/common/protocol_selfcheck.cpp)
target_compile_features(radar_protocol_selfcheck PUBLIC cxx_std_17)
target_link_libraries(radar_protocol_selfcheck radar_decision_comm_core)

add_executable(wireless_protocol_selfcheck src/common/wireless_protocol_selfcheck.cpp)
target_compile_features(wireless_protocol_selfcheck PUBLIC cxx_std_17)
target_link_libraries(wireless_protocol_selfcheck radar_decision_comm_core)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    radar_decision_comm_core
    referee_serial_rx_component
    radar_wireless_rx_component
    client_bridge_component
    perception_adapter_component
    state_blackboard_component
    fusion_component
    decision_engine_component
    tx_scheduler_component
    referee_serial_tx_component
    radar_protocol_selfcheck
    wireless_protocol_selfcheck
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)
ament_package()


target_link_libraries(
  radar_decision_comm_core
  BT::behaviortree_cpp_v3
)


install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
