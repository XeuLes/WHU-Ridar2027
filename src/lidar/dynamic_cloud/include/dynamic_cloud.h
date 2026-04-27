#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <chrono>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <vision_interface/msg/radar_warn.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <std_msgs/msg/u_int8.hpp>
#include <mutex>
#include <unordered_map>
#include <string>

namespace tdt_radar {

enum class LocalizationStatus : uint8_t {
    LOCALIZING    = 0,
    PRIOR_SUCCESS = 1,
    INIT_ONLY     = 2
};

enum class RuntimeMapMode : uint8_t {
    PRIOR_MAP   = 0,
    ACTUAL_MAP  = 1,
    NO_VALID_MAP = 2
};
class DynamicCloud : public rclcpp::Node {
public:
    DynamicCloud(const rclcpp::NodeOptions& node_options);
    ~DynamicCloud() {}


    

private:
    int accumulate_time = 3;
    int accumulate_count = 0;
    
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        other_accumulated_clouds_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr other_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr roi_pub_;
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr  detect_pub_;
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void GetDynamicCloud(pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>& output_cloud,
                         float threshold, int thread_num);
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_Tree;
    tf2_ros::Buffer                 tf_buffer_;
    tf2_ros::TransformListener      tf_listener_;

    void localizationStatusCallback(const std_msgs::msg::UInt8::SharedPtr msg);
    void refreshRuntimeMapMode();

    void resetActualMapBuildState();
    void updateActualMapAccumulator(const pcl::PointCloud<pcl::PointXYZ>& roi_cloud);
    bool finalizeActualMap();
    std::string makeVoxelKey(const pcl::PointXYZ& pt) const;

    const char* mapModeToString(RuntimeMapMode mode) const;
    const char* localizationStatusToString(LocalizationStatus status) const;

    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr localization_status_sub_;

    LocalizationStatus localization_status_{LocalizationStatus::LOCALIZING};
    RuntimeMapMode runtime_map_mode_{RuntimeMapMode::PRIOR_MAP};

    std::mutex status_mutex_;
    std::mutex actual_map_mutex_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prior_map_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr actual_map_cloud_;

    bool actual_map_ready_{false};
    bool actual_map_building_{false};
    bool actual_map_finalized_once_{false};

    int actual_map_min_frames_{30};
    int actual_map_min_stable_voxels_{200};
    double actual_map_voxel_size_{0.20};
    double actual_map_min_observe_ratio_{0.60};

    double uav_other_min_x_{0.0};
    double uav_other_max_x_{28.0};
    double uav_other_min_y_{0.0};
    double uav_other_max_y_{15.0};
    double uav_other_min_z_{1.2};
    double uav_other_max_z_{4.5};

    int actual_map_accumulated_frames_{0};

    std::unordered_map<std::string, int> actual_voxel_hit_count_;
    std::unordered_map<std::string, Eigen::Vector3f> actual_voxel_sum_;
};
}  // namespace tdt_radar
