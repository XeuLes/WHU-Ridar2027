cmake_minimum_required(VERSION 3.10)
project(radar_pdu_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  executables/pdu_subscriber_node
  executables/shm_pdu_broadcast_node
  executables/radar_decision_sync_shm_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
