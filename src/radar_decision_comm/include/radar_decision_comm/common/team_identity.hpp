#pragma once

#include <cstdint>
#include <string>

namespace radar_decision_comm::team_identity
{

inline bool is_blue_robot_id(uint16_t robot_id)
{
  return robot_id >= 101U && robot_id <= 109U;
}

inline bool is_red_robot_id(uint16_t robot_id)
{
  return robot_id >= 1U && robot_id <= 9U;
}

inline bool is_valid_self_color(int self_color)
{
  return self_color == 0 || self_color == 2;
}

inline int self_color_code_from_robot_id(uint16_t robot_id)
{
  if (is_blue_robot_id(robot_id)) {
    return 0;
  }
  if (is_red_robot_id(robot_id)) {
    return 2;
  }
  return -1;
}

inline std::string my_color_from_self_color(int self_color)
{
  if (self_color == 0) {
    return "Blue";
  }
  return "Red";
}

inline std::string my_color_from_robot_id(uint16_t robot_id)
{
  return my_color_from_self_color(self_color_code_from_robot_id(robot_id));
}

inline std::string enemy_color_from_self_color(int self_color)
{
  if (self_color == 0) {
    return "Red";
  }
  return "Blue";
}

inline int enemy_auto_aim_target_color_from_self_color(int self_color)
{
  return (self_color == 0) ? 0 : 1;
}

inline bool enemy_yolo_swap_rb_from_self_color(int self_color)
{
  return self_color == 0;
}

inline std::string enemy_target_side_from_self_color(int self_color)
{
  if (self_color == 0) {
    return "red";
  }
  return "blue";
}

}  // namespace radar_decision_comm::team_identity
