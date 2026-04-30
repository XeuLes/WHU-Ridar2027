#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/time.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace auto_aim_fast::pipeline
{

struct ROI
{
  int x{0}, y{0}, w{0}, h{0};
  bool valid{false};
};

struct Budget
{
  int max_lights{64};
  int max_pairs{128};
  int max_pnp{2};
};

struct Frame
{
  sensor_msgs::msg::Image::ConstSharedPtr image;
  rclcpp::Time stamp_ros;
  int64_t stamp_ns{0};
  int64_t recv_ns{0};
  std::string frame_id;

  ROI roi;
  Budget budget;
};

}  // namespace auto_aim_fast::pipeline
