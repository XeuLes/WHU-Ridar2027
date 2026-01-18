#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {
struct XYZRPY {
    double x{0}, y{0}, z{0};
    double roll{0}, pitch{0}, yaw{0};
};

static tf2::Quaternion quatFromRPY(double roll, double pitch, double yaw)
{
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    q.normalize();
    return q;
}

static geometry_msgs::msg::TransformStamped
makeTF(const rclcpp::Time& stamp, const std::string& parent,
       const std::string& child, const XYZRPY& tr)
{
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = stamp;
    ts.header.frame_id = parent;
    ts.child_frame_id = child;

    ts.transform.translation.x = tr.x;
    ts.transform.translation.y = tr.y;
    ts.transform.translation.z = tr.z;

    const auto q = quatFromRPY(tr.roll, tr.pitch, tr.yaw);
    ts.transform.rotation.x = q.x();
    ts.transform.rotation.y = q.y();
    ts.transform.rotation.z = q.z();
    ts.transform.rotation.w = q.w();
    return ts;
}

static tf2::Quaternion axisAngleQuat(const std::string& axis, double angle)
{
    tf2::Vector3 ax(0, 0, 1);
    if (axis == "X" || axis == "x")
        ax = tf2::Vector3(1, 0, 0);
    else if (axis == "Y" || axis == "y")
        ax = tf2::Vector3(0, 1, 0);
    else
        ax = tf2::Vector3(0, 0, 1);

    tf2::Quaternion q;
    q.setRotation(ax, angle);
    q.normalize();
    return q;
}

static XYZRPY readXYZRPYParam(rclcpp::Node*      node,
                              const std::string& xyz_key,
                              const std::string& rpy_key)
{
    XYZRPY     out;
    const auto xyz = node->get_parameter(xyz_key).as_double_array();
    const auto rpy = node->get_parameter(rpy_key).as_double_array();
    if (xyz.size() == 3) {
        out.x = xyz[0];
        out.y = xyz[1];
        out.z = xyz[2];
    }
    if (rpy.size() == 3) {
        out.roll = rpy[0];
        out.pitch = rpy[1];
        out.yaw = rpy[2];
    }
    return out;
}

static YAML::Node getNodeByPath(const YAML::Node&  root,
                                const std::string& path)
{
    if (path.empty())
        return YAML::Node();
    YAML::Node cur = root;
    size_t     start = 0;
    while (start < path.size()) {
        size_t      dot = path.find('.', start);
        std::string key = (dot == std::string::npos)
                              ? path.substr(start)
                              : path.substr(start, dot - start);
        if (!cur || !cur[key])
            return YAML::Node();
        cur = cur[key];
        if (dot == std::string::npos)
            break;
        start = dot + 1;
    }
    return cur;
}

static std::optional<XYZRPY>
loadExtrinsicFromYaml(const std::string&              path,
                      const std::vector<std::string>& candidate_node_paths,
                      const std::string&              xyz_key,
                      const std::string&              rpy_key)
{
    try {
        YAML::Node root = YAML::LoadFile(path);

        YAML::Node node;
        for (const auto& p : candidate_node_paths) {
            auto n = getNodeByPath(root, p);
            if (n) {
                node = n;
                break;
            }
        }
        if (!node) {
            for (const auto& p : candidate_node_paths) {
                if (root[p]) {
                    node = root[p];
                    break;
                }
            }
        }
        if (!node)
            return std::nullopt;

        XYZRPY out;

        if (node[xyz_key] && node[xyz_key].IsSequence() &&
            node[xyz_key].size() == 3) {
            out.x = node[xyz_key][0].as<double>();
            out.y = node[xyz_key][1].as<double>();
            out.z = node[xyz_key][2].as<double>();
        } else {
            if (node["x"])
                out.x = node["x"].as<double>();
            if (node["y"])
                out.y = node["y"].as<double>();
            if (node["z"])
                out.z = node["z"].as<double>();
        }

        if (node[rpy_key] && node[rpy_key].IsSequence() &&
            node[rpy_key].size() == 3) {
            out.roll = node[rpy_key][0].as<double>();
            out.pitch = node[rpy_key][1].as<double>();
            out.yaw = node[rpy_key][2].as<double>();
        } else {
            if (node["roll"])
                out.roll = node["roll"].as<double>();
            if (node["pitch"])
                out.pitch = node["pitch"].as<double>();
            if (node["yaw"])
                out.yaw = node["yaw"].as<double>();
        }

        return out;
    }
    catch (...) {
        return std::nullopt;
    }
}

static double deg2rad(double deg)
{
    return deg * M_PI / 180.0;
}
}  // namespace

class GimbalTFPublisher final : public rclcpp::Node {
public:
    explicit GimbalTFPublisher(const rclcpp::NodeOptions& opts)
        : rclcpp::Node("gimbal_tf_publisher", opts),
          tf_broadcaster_(
              std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
          static_broadcaster_(
              std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this))
    {
        declareParams();
        loadParams();

        angle_sub_ =
            this->create_subscription<std_msgs::msg::Float64MultiArray>(
                angle_topic_, rclcpp::SensorDataQoS(),
                std::bind(&GimbalTFPublisher::onAngles, this,
                          std::placeholders::_1));

        const auto period =
            std::chrono::duration<double>(1.0 / publish_rate_hz_);
        dyn_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&GimbalTFPublisher::publishDynamicTF, this));

        status_timer_ = this->create_wall_timer(
            1000ms, std::bind(&GimbalTFPublisher::statusTick, this));

        publishStaticTFOnce();

        RCLCPP_INFO(this->get_logger(),
                    "Started. angles topic=%s ([yaw,pitch]) axis1_joint=%s "
                    "axis2_joint=%s",
                    angle_topic_.c_str(), axis1_joint_frame_.c_str(),
                    axis2_joint_frame_.c_str());
    }

private:
    void declareParams()
    {
        // Frames
        this->declare_parameter<std::string>("root_frame", "rm_frame");
        this->declare_parameter<std::string>("cam_install_frame",
                                             "cam_install_frame");
        this->declare_parameter<std::string>("gimbal_base_frame",
                                             "gimbal_base_frame");

        // NEW: joint-center frames (static)
        this->declare_parameter<std::string>("axis1_joint_frame",
                                             "gimbal_axis1_joint_frame");
        this->declare_parameter<std::string>("axis2_joint_frame",
                                             "gimbal_axis2_joint_frame");

        // Rotated output frames (dynamic)
        this->declare_parameter<std::string>("axis1_frame",
                                             "gimbal_axis1_frame");
        this->declare_parameter<std::string>("axis2_frame",
                                             "gimbal_axis2_frame");

        this->declare_parameter<std::string>("gimbal_cam_frame",
                                             "gimbal_cam_frame");
        this->declare_parameter<std::string>("laser_frame",
                                             "laser_emitter_frame");

        // Angles topic
        this->declare_parameter<std::string>("angle_topic",
                                             "/gimbal/angles");
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);
        this->declare_parameter<std::string>("angle_unit",
                                             "rad");  // "rad" or "deg"

        this->declare_parameter<double>("publish_rate_hz", 200.0);

        // Static transforms (mechanical)
        this->declare_parameter<std::vector<double>>(
            "T_cam_install_to_gimbal_base_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_cam_install_to_gimbal_base_rpy", {0.0, 0.0, 0.0});

        // NEW: fixed offsets to axis centers
        // gimbal_base -> axis1_joint  (fixed translation to yaw motor axis
        // center)
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_base_to_axis1_joint_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_base_to_axis1_joint_rpy", {0.0, 0.0, 0.0});

        // axis1_frame -> axis2_joint  (fixed translation to pitch motor
        // axis center, after yaw rotation)
        this->declare_parameter<std::vector<double>>(
            "T_axis1_to_axis2_joint_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_axis1_to_axis2_joint_rpy", {0.0, 0.0, 0.0});

        // axis2_frame -> gimbal_cam
        this->declare_parameter<std::vector<double>>(
            "T_axis2_to_gimbal_cam_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_axis2_to_gimbal_cam_rpy", {0.0, 0.0, 0.0});

        // gimbal_cam -> laser
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_cam_to_laser_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_cam_to_laser_rpy", {0.0, 0.0, 0.0});

        // Optional extrinsic yaml
        this->declare_parameter<std::string>("extrinsic_yaml", "");
        this->declare_parameter<bool>("publish_rm_to_cam_install_from_yaml",
                                      true);
        this->declare_parameter<std::vector<std::string>>(
            "extrinsic_node_paths",
            std::vector<std::string>{"rm_to_cam_install", "extrinsic"});
        this->declare_parameter<std::string>("extrinsic_xyz_key", "xyz");
        this->declare_parameter<std::string>("extrinsic_rpy_key", "rpy");

        // Rotation conventions
        this->declare_parameter<double>("yaw_offset", 0.0);
        this->declare_parameter<double>("pitch_offset", 0.0);
        this->declare_parameter<std::string>("yaw_axis", "Z");
        this->declare_parameter<std::string>("pitch_axis", "Y");
        this->declare_parameter<int>("yaw_sign", 1);
        this->declare_parameter<int>("pitch_sign", 1);

        // Offline policy
        this->declare_parameter<std::string>(
            "offline_publish_policy", "none");  // none/zero/hold_last
        this->declare_parameter<bool>("warn_when_offline", false);
        this->declare_parameter<double>("angle_timeout_sec", 0.5);
    }

    void loadParams()
    {
        root_frame_ = this->get_parameter("root_frame").as_string();
        cam_install_frame_ =
            this->get_parameter("cam_install_frame").as_string();
        gimbal_base_frame_ =
            this->get_parameter("gimbal_base_frame").as_string();

        axis1_joint_frame_ =
            this->get_parameter("axis1_joint_frame").as_string();
        axis2_joint_frame_ =
            this->get_parameter("axis2_joint_frame").as_string();
        axis1_frame_ = this->get_parameter("axis1_frame").as_string();
        axis2_frame_ = this->get_parameter("axis2_frame").as_string();

        gimbal_cam_frame_ =
            this->get_parameter("gimbal_cam_frame").as_string();
        laser_frame_ = this->get_parameter("laser_frame").as_string();

        angle_topic_ = this->get_parameter("angle_topic").as_string();
        yaw_index_ = this->get_parameter("yaw_index").as_int();
        pitch_index_ = this->get_parameter("pitch_index").as_int();
        angle_unit_ = this->get_parameter("angle_unit").as_string();
        if (angle_unit_ != "deg")
            angle_unit_ = "rad";

        publish_rate_hz_ =
            this->get_parameter("publish_rate_hz").as_double();
        if (publish_rate_hz_ <= 0.0)
            publish_rate_hz_ = 200.0;

        extrinsic_yaml_ = this->get_parameter("extrinsic_yaml").as_string();
        publish_rm_to_cam_install_from_yaml_ =
            this->get_parameter("publish_rm_to_cam_install_from_yaml")
                .as_bool();
        extrinsic_node_paths_ =
            this->get_parameter("extrinsic_node_paths").as_string_array();
        extrinsic_xyz_key_ =
            this->get_parameter("extrinsic_xyz_key").as_string();
        extrinsic_rpy_key_ =
            this->get_parameter("extrinsic_rpy_key").as_string();

        yaw_offset_ = this->get_parameter("yaw_offset").as_double();
        pitch_offset_ = this->get_parameter("pitch_offset").as_double();
        yaw_axis_ = this->get_parameter("yaw_axis").as_string();
        pitch_axis_ = this->get_parameter("pitch_axis").as_string();

        yaw_sign_ = this->get_parameter("yaw_sign").as_int();
        pitch_sign_ = this->get_parameter("pitch_sign").as_int();
        yaw_sign_ = (yaw_sign_ == -1) ? -1 : 1;
        pitch_sign_ = (pitch_sign_ == -1) ? -1 : 1;

        offline_publish_policy_ =
            this->get_parameter("offline_publish_policy").as_string();
        if (offline_publish_policy_ != "zero" &&
            offline_publish_policy_ != "hold_last")
            offline_publish_policy_ = "none";
        warn_when_offline_ =
            this->get_parameter("warn_when_offline").as_bool();
        angle_timeout_sec_ =
            this->get_parameter("angle_timeout_sec").as_double();
        if (angle_timeout_sec_ <= 0.0)
            angle_timeout_sec_ = 0.5;

        T_cam_install_to_gimbal_base_ =
            readXYZRPYParam(this, "T_cam_install_to_gimbal_base_xyz",
                            "T_cam_install_to_gimbal_base_rpy");
        T_base_to_axis1_joint_ =
            readXYZRPYParam(this, "T_gimbal_base_to_axis1_joint_xyz",
                            "T_gimbal_base_to_axis1_joint_rpy");
        T_axis1_to_axis2_joint_ =
            readXYZRPYParam(this, "T_axis1_to_axis2_joint_xyz",
                            "T_axis1_to_axis2_joint_rpy");
        T_axis2_to_gimbal_cam_ = readXYZRPYParam(
            this, "T_axis2_to_gimbal_cam_xyz", "T_axis2_to_gimbal_cam_rpy");
        T_gimbal_cam_to_laser_ = readXYZRPYParam(
            this, "T_gimbal_cam_to_laser_xyz", "T_gimbal_cam_to_laser_rpy");
    }

    void publishStaticTFOnce()
    {
        const auto stamp = this->now();
        std::vector<geometry_msgs::msg::TransformStamped> out;

        if (publish_rm_to_cam_install_from_yaml_) {
            if (!extrinsic_yaml_.empty()) {
                auto ext = loadExtrinsicFromYaml(
                    extrinsic_yaml_, extrinsic_node_paths_,
                    extrinsic_xyz_key_, extrinsic_rpy_key_);
                if (ext.has_value())
                    out.push_back(makeTF(stamp, root_frame_,
                                         cam_install_frame_, ext.value()));
                else
                    RCLCPP_WARN(this->get_logger(),
                                "Extrinsic yaml parse failed: %s",
                                extrinsic_yaml_.c_str());
            } else {
                RCLCPP_WARN(
                    this->get_logger(), "extrinsic_yaml empty. Skip %s->%s",
                    root_frame_.c_str(), cam_install_frame_.c_str());
            }
        }

        // cam_install -> gimbal_base
        out.push_back(makeTF(stamp, cam_install_frame_, gimbal_base_frame_,
                             T_cam_install_to_gimbal_base_));

        // NEW: gimbal_base -> axis1_joint (offset to yaw axis center)
        out.push_back(makeTF(stamp, gimbal_base_frame_, axis1_joint_frame_,
                             T_base_to_axis1_joint_));

        // NEW: axis1_frame -> axis2_joint (offset to pitch axis center,
        // after yaw)
        out.push_back(makeTF(stamp, axis1_frame_, axis2_joint_frame_,
                             T_axis1_to_axis2_joint_));

        // axis2_frame -> gimbal_cam
        out.push_back(makeTF(stamp, axis2_frame_, gimbal_cam_frame_,
                             T_axis2_to_gimbal_cam_));

        // gimbal_cam -> laser
        out.push_back(makeTF(stamp, gimbal_cam_frame_, laser_frame_,
                             T_gimbal_cam_to_laser_));

        static_broadcaster_->sendTransform(out);
    }

    void onAngles(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.empty())
            return;
        const int n = static_cast<int>(msg->data.size());
        if (yaw_index_ < 0 || yaw_index_ >= n || pitch_index_ < 0 ||
            pitch_index_ >= n) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                                 2000,
                                 "Angle array size=%d but yaw_index=%d "
                                 "pitch_index=%d out of range.",
                                 n, yaw_index_, pitch_index_);
            return;
        }

        std::lock_guard<std::mutex> lk(mutex_);
        yaw_raw_ = msg->data[static_cast<size_t>(yaw_index_)];
        pitch_raw_ = msg->data[static_cast<size_t>(pitch_index_)];
        last_angle_rx_time_ = this->now();
        ever_received_angle_ = true;
    }

    bool isOfflineLocked(const rclcpp::Time& now) const
    {
        if (!ever_received_angle_)
            return true;
        return ((now - last_angle_rx_time_).seconds() > angle_timeout_sec_);
    }

    void publishDynamicTF()
    {
        const auto now = this->now();

        double yaw_raw = 0.0, pitch_raw = 0.0;
        bool   offline;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            offline = isOfflineLocked(now);
            if (offline) {
                if (offline_publish_policy_ == "none")
                    return;
                if (offline_publish_policy_ == "hold_last") {
                    yaw_raw = yaw_raw_;
                    pitch_raw = pitch_raw_;
                } else  // zero
                {
                    yaw_raw = 0.0;
                    pitch_raw = 0.0;
                }
            } else {
                yaw_raw = yaw_raw_;
                pitch_raw = pitch_raw_;
            }
        }

        if (angle_unit_ == "deg") {
            yaw_raw = deg2rad(yaw_raw);
            pitch_raw = deg2rad(pitch_raw);
        }

        const double yaw = yaw_sign_ * (yaw_raw + yaw_offset_);
        const double pitch = pitch_sign_ * (pitch_raw + pitch_offset_);

        // Dynamic 1: axis1_joint -> axis1 (yaw rotation)
        geometry_msgs::msg::TransformStamped t1;
        t1.header.stamp = now;
        t1.header.frame_id = axis1_joint_frame_;
        t1.child_frame_id = axis1_frame_;
        t1.transform.translation.x = 0.0;
        t1.transform.translation.y = 0.0;
        t1.transform.translation.z = 0.0;
        {
            const auto q = axisAngleQuat(yaw_axis_, yaw);
            t1.transform.rotation.x = q.x();
            t1.transform.rotation.y = q.y();
            t1.transform.rotation.z = q.z();
            t1.transform.rotation.w = q.w();
        }

        // Dynamic 2: axis2_joint -> axis2 (pitch rotation)
        geometry_msgs::msg::TransformStamped t2;
        t2.header.stamp = now;
        t2.header.frame_id = axis2_joint_frame_;
        t2.child_frame_id = axis2_frame_;
        t2.transform.translation.x = 0.0;
        t2.transform.translation.y = 0.0;
        t2.transform.translation.z = 0.0;
        {
            const auto q = axisAngleQuat(pitch_axis_, pitch);
            t2.transform.rotation.x = q.x();
            t2.transform.rotation.y = q.y();
            t2.transform.rotation.z = q.z();
            t2.transform.rotation.w = q.w();
        }

        tf_broadcaster_->sendTransform(t1);
        tf_broadcaster_->sendTransform(t2);
    }

    void statusTick()
    {
        const auto now = this->now();
        bool       offline;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            offline = isOfflineLocked(now);
        }
        if (offline && warn_when_offline_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Angles offline (normal if gimbal not powered). Listening "
                "on %s. offline_publish_policy=%s",
                angle_topic_.c_str(), offline_publish_policy_.c_str());
        } else if (offline && !warn_when_offline_) {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 15000,
                "Angles offline (normal). Listening on %s.",
                angle_topic_.c_str());
        }
    }

private:
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster>
        static_broadcaster_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
                                 angle_sub_;
    rclcpp::TimerBase::SharedPtr dyn_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;

    // frames
    std::string root_frame_;
    std::string cam_install_frame_;
    std::string gimbal_base_frame_;

    std::string axis1_joint_frame_;
    std::string axis2_joint_frame_;
    std::string axis1_frame_;
    std::string axis2_frame_;

    std::string gimbal_cam_frame_;
    std::string laser_frame_;

    // angles
    std::string angle_topic_;
    int         yaw_index_{0};
    int         pitch_index_{1};
    std::string angle_unit_{"rad"};
    double      publish_rate_hz_{200.0};

    // extrinsic yaml
    std::string              extrinsic_yaml_;
    bool                     publish_rm_to_cam_install_from_yaml_{true};
    std::vector<std::string> extrinsic_node_paths_;
    std::string              extrinsic_xyz_key_{"xyz"};
    std::string              extrinsic_rpy_key_{"rpy"};

    // static transforms
    XYZRPY T_cam_install_to_gimbal_base_;
    XYZRPY T_base_to_axis1_joint_;
    XYZRPY T_axis1_to_axis2_joint_;
    XYZRPY T_axis2_to_gimbal_cam_;
    XYZRPY T_gimbal_cam_to_laser_;

    // rotation conventions
    double      yaw_offset_{0.0};
    double      pitch_offset_{0.0};
    std::string yaw_axis_{"Z"};
    std::string pitch_axis_{"Y"};
    int         yaw_sign_{1};
    int         pitch_sign_{1};

    // offline
    double      angle_timeout_sec_{0.5};
    bool        warn_when_offline_{false};
    std::string offline_publish_policy_{"none"};

    // state
    std::mutex   mutex_;
    bool         ever_received_angle_{false};
    double       yaw_raw_{0.0};
    double       pitch_raw_{0.0};
    rclcpp::Time last_angle_rx_time_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(
        std::make_shared<GimbalTFPublisher>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
