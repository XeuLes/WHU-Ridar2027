#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "radar_decision_comm/msg/map_publish_request.hpp"

namespace radar_decision_comm
{

struct MapPayloadBuildResult
{
  std::vector<uint8_t> payload;
  std::string layout_used;
  bool used_fallback{false};
};

class MapPayloadBuilder
{
public:
  static MapPayloadBuildResult build_payload(
    const radar_decision_comm::msg::MapPublishRequest & msg,
    const std::string & my_color,
    const std::string & layout);

private:
  struct PackedTarget
  {
    uint16_t x_cm{0};
    uint16_t y_cm{0};
    bool valid{false};
    uint8_t slot_index{0};
  };

  static int slot_from_robot_id(const std::string & my_color, uint16_t robot_id);
  static uint16_t meter_to_cm_u16(double value_m);
  static void append_u16_le(std::vector<uint8_t> & out, uint16_t value);

  static std::array<PackedTarget, 6> build_slots(
    const radar_decision_comm::msg::MapPublishRequest & msg,
    const std::string & my_color);

  static std::vector<uint8_t> build_legacy24(
    const std::array<PackedTarget, 6> & slots);

  static std::vector<uint8_t> build_proto48_operational(
    const std::array<PackedTarget, 6> & slots);
};

}  // namespace radar_decision_comm
