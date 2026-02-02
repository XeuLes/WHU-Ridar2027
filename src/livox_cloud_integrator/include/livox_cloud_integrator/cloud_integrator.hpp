#pragma once
#include <deque>
#include <string>
#include <vector>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace livox_cloud_integrator
{

class CloudIntegrator : public rclcpp::Node
{
public:
  explicit CloudIntegrator(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  struct CloudFrame
  {
    rclcpp::Time stamp;
    sensor_msgs::msg::PointCloud2 msg;  // 存一份完整帧（含 data）
    size_t points = 0;
  };

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  bool compatible_layout(const sensor_msgs::msg::PointCloud2& a,
                         const sensor_msgs::msg::PointCloud2& b) const;

  void pop_expired(const rclcpp::Time& now_stamp);
  sensor_msgs::msg::PointCloud2 compose_integrated(const sensor_msgs::msg::PointCloud2& ref_header_layout) const;

private:
  // params
  std::string input_topic_;
  std::string output_topic_;
  double window_sec_;
  bool use_header_stamp_;
  size_t max_points_;

  // io
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  // state
  std::deque<CloudFrame> queue_;
  // 用第一帧作为布局参考
  bool has_ref_layout_ = false;
  sensor_msgs::msg::PointCloud2 ref_layout_;
};

}  // namespace livox_cloud_integrator
