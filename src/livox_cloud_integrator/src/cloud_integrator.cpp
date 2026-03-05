#include "livox_cloud_integrator/cloud_integrator.hpp"

#include <algorithm>
#include <numeric>

namespace livox_cloud_integrator
{

static rclcpp::QoS qos_from_sensor_data_reliable()
{
  // 你 topic info 看到 Reliability=RELIABLE，Durability=VOLATILE
  // depth unknown -> 订阅侧用 keep_last(10) 通常够；关键是 Reliability 选 RELIABLE
  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.reliable();
  qos.durability_volatile();
  return qos;
}

CloudIntegrator::CloudIntegrator(const rclcpp::NodeOptions& options)
: Node("cloud_integrator", options)
{
  input_topic_ = this->declare_parameter<std::string>("input_topic", "/livox/lidar");
  output_topic_ = this->declare_parameter<std::string>("output_topic", "/livox/lidar_integrated");
  window_sec_ = this->declare_parameter<double>("window_sec", 0.2);
  use_header_stamp_ = this->declare_parameter<bool>("use_header_stamp", true);
  max_points_ = static_cast<size_t>(this->declare_parameter<int>("max_points", 200000));

  pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos_from_sensor_data_reliable());
  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, qos_from_sensor_data_reliable(),
    std::bind(&CloudIntegrator::on_cloud, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "CloudIntegrator: subscribe=%s publish=%s window=%.3fs use_header_stamp=%s max_points=%zu",
              input_topic_.c_str(), output_topic_.c_str(), window_sec_,
              use_header_stamp_ ? "true" : "false", max_points_);
}

bool CloudIntegrator::compatible_layout(const sensor_msgs::msg::PointCloud2& a,
                                        const sensor_msgs::msg::PointCloud2& b) const
{
  if (a.point_step != b.point_step) return false;
  if (a.is_bigendian != b.is_bigendian) return false;
  if (a.is_dense != b.is_dense) return false;

  if (a.fields.size() != b.fields.size()) return false;
  for (size_t i = 0; i < a.fields.size(); ++i)
  {
    const auto& fa = a.fields[i];
    const auto& fb = b.fields[i];
    if (fa.name != fb.name) return false;
    if (fa.offset != fb.offset) return false;
    if (fa.datatype != fb.datatype) return false;
    if (fa.count != fb.count) return false;
  }
  return true;
}

void CloudIntegrator::pop_expired(const rclcpp::Time& now_stamp)
{
  const rclcpp::Duration win = rclcpp::Duration::from_seconds(std::max(0.0, window_sec_));
  while (!queue_.empty())
  {
    if ((now_stamp - queue_.front().stamp) > win) queue_.pop_front();
    else break;
  }
}

sensor_msgs::msg::PointCloud2 CloudIntegrator::compose_integrated(
  const sensor_msgs::msg::PointCloud2& ref_header_layout) const
{
  sensor_msgs::msg::PointCloud2 out;
  out.header = ref_header_layout.header;
  out.height = 1;
  out.fields = ref_header_layout.fields;
  out.is_bigendian = ref_header_layout.is_bigendian;
  out.point_step = ref_header_layout.point_step;
  out.is_dense = ref_header_layout.is_dense;

  // 总点数
  size_t total_points = 0;
  for (const auto& f : queue_) total_points += f.points;
  total_points = std::min(total_points, max_points_);

  out.width = static_cast<uint32_t>(total_points);
  out.row_step = out.point_step * out.width;

  const size_t total_bytes = static_cast<size_t>(out.row_step) * out.height;
  out.data.resize(total_bytes);

  // 逐帧拷贝 data
  size_t copied_points = 0;
  size_t dst_offset = 0;

  for (const auto& f : queue_)
  {
    if (copied_points >= total_points) break;

    const size_t can_copy_points = std::min(f.points, total_points - copied_points);
    const size_t bytes = can_copy_points * static_cast<size_t>(out.point_step);

    if (bytes == 0) continue;

    // 拷贝二进制点数据
    std::copy_n(f.msg.data.data(), bytes, out.data.data() + dst_offset);

    dst_offset += bytes;
    copied_points += can_copy_points;
  }

  // 如果因为 max_points_ 截断导致 dst_offset < out.data.size()，裁掉多余空间
  out.data.resize(dst_offset);
  out.width = static_cast<uint32_t>(copied_points);
  out.row_step = out.point_step * out.width;

  return out;
}

void CloudIntegrator::on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  const rclcpp::Time stamp = use_header_stamp_ ? rclcpp::Time(msg->header.stamp) : this->now();

  // 初始化参考布局
  if (!has_ref_layout_)
  {
    ref_layout_ = *msg;
    // 参考布局不需要保存 data，只要布局字段
    ref_layout_.data.clear();
    ref_layout_.width = 0;
    ref_layout_.height = 1;
    ref_layout_.row_step = 0;
    has_ref_layout_ = true;
  }
  else
  {
    // layout 必须一致
    if (!compatible_layout(*msg, ref_layout_))
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Incoming cloud layout differs from reference. Skip this frame.");
      return;
    }
  }

  // 计算点数（常规：height=1，width=点数）
  const size_t points = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);

  CloudFrame frame;
  frame.stamp = stamp;
  frame.msg = *msg;   // 拷贝一份（包含 data）
  frame.points = points;

  queue_.push_back(std::move(frame));

  // 清理过期
  pop_expired(stamp);

  // 用最新帧 header 作为输出 header（frame_id/时间戳）
  sensor_msgs::msg::PointCloud2 header_layout = ref_layout_;
  header_layout.header = msg->header;

  auto out = compose_integrated(header_layout);
  pub_->publish(std::move(out));
}

}  // namespace livox_cloud_integrator

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(livox_cloud_integrator::CloudIntegrator)
