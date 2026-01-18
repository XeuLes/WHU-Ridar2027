#include <rclcpp/rclcpp.hpp>

#include <auto_aim_fast/msg/aim_target.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>

namespace {
static tf2::Transform
tfFromMsg(const geometry_msgs::msg::TransformStamped& ts)
{
    tf2::Transform T;
    T.setOrigin(tf2::Vector3(ts.transform.translation.x,
                             ts.transform.translation.y,
                             ts.transform.translation.z));
    tf2::Quaternion q(ts.transform.rotation.x, ts.transform.rotation.y,
                      ts.transform.rotation.z, ts.transform.rotation.w);
    q.normalize();
    T.setRotation(q);
    return T;
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

static double deg2rad(double deg)
{
    return deg * M_PI / 180.0;
}
static double rad2deg(double rad)
{
    return rad * 180.0 / M_PI;
}

static double clampAbs(double v, double max_abs)
{
    if (v > max_abs)
        return max_abs;
    if (v < -max_abs)
        return -max_abs;
    return v;
}

// Solve 2x2: [a b; c d] [x;y] = [e;f]
static bool solve2x2(double a, double b, double c, double d, double e,
                     double f, double& x, double& y)
{
    const double det = a * d - b * c;
    if (std::fabs(det) < 1e-12)
        return false;
    x = (e * d - b * f) / det;
    y = (-e * c + a * f) / det;
    return true;
}

static rclcpp::Time timeFromBuiltin(const builtin_interfaces::msg::Time& t,
                                    rclcpp::Clock::SharedPtr clock)
{
    const int64_t ns = static_cast<int64_t>(t.sec) * 1000000000LL +
                       static_cast<int64_t>(t.nanosec);
    return rclcpp::Time(ns, clock->get_clock_type());
}
}  // namespace

class GimbalAimSolver final : public rclcpp::Node {
public:
    explicit GimbalAimSolver(const rclcpp::NodeOptions& opts)
        : rclcpp::Node("gimbal_aim_solver", opts),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        declareParams();
        loadParams();

        sub_target_ =
            this->create_subscription<auto_aim_fast::msg::AimTarget>(
                aim_target_topic_, rclcpp::QoS(10),
                std::bind(&GimbalAimSolver::onAimTarget, this,
                          std::placeholders::_1));

        sub_angles_ =
            this->create_subscription<std_msgs::msg::Float64MultiArray>(
                current_angle_topic_, rclcpp::SensorDataQoS(),
                std::bind(&GimbalAimSolver::onAngles, this,
                          std::placeholders::_1));

        pub_cmd_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            cmd_topic_, rclcpp::QoS(10));
        pub_target_base_ =
            this->create_publisher<geometry_msgs::msg::PointStamped>(
                target_base_topic_, rclcpp::QoS(10));

        RCLCPP_INFO(this->get_logger(),
                    "Started TF-based solver. base=%s cam=%s laser=%s "
                    "a1j=%s a1=%s a2j=%s a2=%s",
                    base_frame_.c_str(), cam_frame_.c_str(),
                    laser_frame_.c_str(), axis1_joint_frame_.c_str(),
                    axis1_frame_.c_str(), axis2_joint_frame_.c_str(),
                    axis2_frame_.c_str());
    }

private:
    void declareParams()
    {
        this->declare_parameter<std::string>("aim_target_topic",
                                             "/auto_aim/target");
        this->declare_parameter<std::string>("current_angle_topic",
                                             "/gimbal/angles");

        this->declare_parameter<std::string>("cmd_topic",
                                             "/gimbal/command_angles");
        this->declare_parameter<std::string>("target_base_topic",
                                             "/gimbal/target_base");

        // Frames: must match your TF publisher
        this->declare_parameter<std::string>("base_frame",
                                             "gimbal_base_frame");
        this->declare_parameter<std::string>(
            "cam_frame",
            "gimbal_cam_frame");  // the frame where tx/ty/tz is expressed
        this->declare_parameter<std::string>("laser_frame",
                                             "laser_emitter_frame");

        this->declare_parameter<std::string>("axis1_joint_frame",
                                             "gimbal_axis1_joint_frame");
        this->declare_parameter<std::string>("axis1_frame",
                                             "gimbal_axis1_frame");
        this->declare_parameter<std::string>("axis2_joint_frame",
                                             "gimbal_axis2_joint_frame");
        this->declare_parameter<std::string>("axis2_frame",
                                             "gimbal_axis2_frame");

        // Angle array format
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);
        this->declare_parameter<std::string>("current_angle_unit",
                                             "rad");  // rad/deg
        this->declare_parameter<std::string>("output_angle_unit",
                                             "rad");  // rad/deg

        // Axis conventions (must match TF publisher / your mechanical
        // definition)
        this->declare_parameter<std::string>("yaw_axis", "Z");
        this->declare_parameter<std::string>("pitch_axis", "Y");
        this->declare_parameter<int>("yaw_sign", 1);
        this->declare_parameter<int>("pitch_sign", 1);
        this->declare_parameter<double>("yaw_offset", 0.0);
        this->declare_parameter<double>("pitch_offset", 0.0);

        // IK params
        this->declare_parameter<int>("ik.max_iter", 15);
        this->declare_parameter<double>("ik.tol_xy_m", 1e-4);
        this->declare_parameter<double>("ik.eps_rad", 1e-4);
        this->declare_parameter<double>("ik.max_step_rad", 0.2);

        // TF query timeout
        this->declare_parameter<double>("tf_timeout_sec", 0.05);
    }

    void loadParams()
    {
        aim_target_topic_ =
            this->get_parameter("aim_target_topic").as_string();
        current_angle_topic_ =
            this->get_parameter("current_angle_topic").as_string();
        cmd_topic_ = this->get_parameter("cmd_topic").as_string();
        target_base_topic_ =
            this->get_parameter("target_base_topic").as_string();

        base_frame_ = this->get_parameter("base_frame").as_string();
        cam_frame_ = this->get_parameter("cam_frame").as_string();
        laser_frame_ = this->get_parameter("laser_frame").as_string();

        axis1_joint_frame_ =
            this->get_parameter("axis1_joint_frame").as_string();
        axis1_frame_ = this->get_parameter("axis1_frame").as_string();
        axis2_joint_frame_ =
            this->get_parameter("axis2_joint_frame").as_string();
        axis2_frame_ = this->get_parameter("axis2_frame").as_string();

        yaw_index_ = this->get_parameter("yaw_index").as_int();
        pitch_index_ = this->get_parameter("pitch_index").as_int();
        current_angle_unit_ =
            this->get_parameter("current_angle_unit").as_string();
        if (current_angle_unit_ != "deg")
            current_angle_unit_ = "rad";
        output_angle_unit_ =
            this->get_parameter("output_angle_unit").as_string();
        if (output_angle_unit_ != "deg")
            output_angle_unit_ = "rad";

        yaw_axis_ = this->get_parameter("yaw_axis").as_string();
        pitch_axis_ = this->get_parameter("pitch_axis").as_string();
        yaw_sign_ =
            (this->get_parameter("yaw_sign").as_int() == -1) ? -1 : 1;
        pitch_sign_ =
            (this->get_parameter("pitch_sign").as_int() == -1) ? -1 : 1;
        yaw_offset_ = this->get_parameter("yaw_offset").as_double();
        pitch_offset_ = this->get_parameter("pitch_offset").as_double();

        const int64_t max_iter_i64 =
            this->get_parameter("ik.max_iter").as_int();
        ik_max_iter_ = std::max<int>(1, static_cast<int>(max_iter_i64));
        ik_tol_xy_m_ = this->get_parameter("ik.tol_xy_m").as_double();
        ik_eps_rad_ = this->get_parameter("ik.eps_rad").as_double();
        ik_max_step_rad_ =
            this->get_parameter("ik.max_step_rad").as_double();
        if (ik_eps_rad_ <= 0.0)
            ik_eps_rad_ = 1e-4;
        if (ik_max_step_rad_ <= 0.0)
            ik_max_step_rad_ = 0.2;

        tf_timeout_sec_ = this->get_parameter("tf_timeout_sec").as_double();
        if (tf_timeout_sec_ <= 0.0)
            tf_timeout_sec_ = 0.05;
    }

    // raw -> physical
    double rawToPhysicalYaw(double raw) const
    {
        return static_cast<double>(yaw_sign_) * (raw + yaw_offset_);
    }
    double rawToPhysicalPitch(double raw) const
    {
        return static_cast<double>(pitch_sign_) * (raw + pitch_offset_);
    }

    // physical -> raw
    double physicalToRawYaw(double phy) const
    {
        return static_cast<double>(yaw_sign_) * phy - yaw_offset_;
    }
    double physicalToRawPitch(double phy) const
    {
        return static_cast<double>(pitch_sign_) * phy - pitch_offset_;
    }

    void onAngles(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        const int n = static_cast<int>(msg->data.size());
        if (n <= 0)
            return;
        if (yaw_index_ < 0 || yaw_index_ >= n || pitch_index_ < 0 ||
            pitch_index_ >= n)
            return;

        double yaw_raw = msg->data[static_cast<size_t>(yaw_index_)];
        double pitch_raw = msg->data[static_cast<size_t>(pitch_index_)];

        if (current_angle_unit_ == "deg") {
            yaw_raw = deg2rad(yaw_raw);
            pitch_raw = deg2rad(pitch_raw);
        }

        std::lock_guard<std::mutex> lk(mutex_);
        has_current_angle_ = true;
        yaw_raw_cur_ = yaw_raw;
        pitch_raw_cur_ = pitch_raw;
    }

    bool lookupTF(const std::string& target, const std::string& source,
                  const rclcpp::Time&                   stamp,
                  geometry_msgs::msg::TransformStamped& out)
    {
        try {
            out = tf_buffer_.lookupTransform(
                target, source, stamp,
                rclcpp::Duration::from_seconds(tf_timeout_sec_));
            return true;
        }
        catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "TF lookup failed: %s <- %s at t=%.3f : %s", target.c_str(),
                source.c_str(), stamp.seconds(), e.what());
            return false;
        }
    }

    // Build base->laser transform using TF chain, but allowing "virtual"
    // perturbation on joint rotations. delta_yaw: additional rotation
    // applied at axis1_joint->axis1 delta_pitch: additional rotation
    // applied at axis2_joint->axis2
    bool computeBaseToLaserVirtual(const rclcpp::Time& stamp,
                                   double delta_yaw, double delta_pitch,
                                   tf2::Transform& T_base_laser_out)
    {
        geometry_msgs::msg::TransformStamped ts_base_a1j;
        geometry_msgs::msg::TransformStamped ts_a1j_a1;
        geometry_msgs::msg::TransformStamped ts_a1_a2j;
        geometry_msgs::msg::TransformStamped ts_a2j_a2;
        geometry_msgs::msg::TransformStamped ts_a2_laser;

        // Note: lookupTransform(target, source) returns source->target
        if (!lookupTF(axis1_joint_frame_, base_frame_, stamp, ts_base_a1j))
            return false;
        if (!lookupTF(axis1_frame_, axis1_joint_frame_, stamp, ts_a1j_a1))
            return false;
        if (!lookupTF(axis2_joint_frame_, axis1_frame_, stamp, ts_a1_a2j))
            return false;
        if (!lookupTF(axis2_frame_, axis2_joint_frame_, stamp, ts_a2j_a2))
            return false;
        if (!lookupTF(laser_frame_, axis2_frame_, stamp, ts_a2_laser))
            return false;

        tf2::Transform T_base_a1j = tfFromMsg(ts_base_a1j);
        tf2::Transform T_a1j_a1 = tfFromMsg(ts_a1j_a1);
        tf2::Transform T_a1_a2j = tfFromMsg(ts_a1_a2j);
        tf2::Transform T_a2j_a2 = tfFromMsg(ts_a2j_a2);
        tf2::Transform T_a2_laser = tfFromMsg(ts_a2_laser);

        // Apply virtual yaw perturbation at axis1_joint->axis1 rotation
        // (keep translation)
        {
            tf2::Transform d;
            d.setOrigin(tf2::Vector3(0, 0, 0));
            d.setRotation(axisAngleQuat(yaw_axis_, delta_yaw));
            // Since joint axis is defined in axis1_joint frame,
            // left-multiply is appropriate.
            T_a1j_a1.setRotation(d.getRotation() * T_a1j_a1.getRotation());
        }

        // Apply virtual pitch perturbation at axis2_joint->axis2 rotation
        {
            tf2::Transform d;
            d.setOrigin(tf2::Vector3(0, 0, 0));
            d.setRotation(axisAngleQuat(pitch_axis_, delta_pitch));
            T_a2j_a2.setRotation(d.getRotation() * T_a2j_a2.getRotation());
        }

        // Chain: base->laser = (a2->laser) * (a2j->a2) * (a1->a2j) *
        // (a1j->a1) * (base->a1j)
        T_base_laser_out =
            T_a2_laser * T_a2j_a2 * T_a1_a2j * T_a1j_a1 * T_base_a1j;
        return true;
    }

    bool computeCamToBase(const rclcpp::Time& stamp,
                          tf2::Transform&     T_cam_base_out)
    {
        geometry_msgs::msg::TransformStamped ts_base_cam;
        if (!lookupTF(base_frame_, cam_frame_, stamp, ts_base_cam))
            return false;
        // ts_base_cam is cam->base (source=cam, target=base) by our lookup
        T_cam_base_out = tfFromMsg(ts_base_cam);
        return true;
    }

    bool computeLaserToBase(const rclcpp::Time& stamp,
                            tf2::Transform&     T_laser_base_out)
    {
        geometry_msgs::msg::TransformStamped ts_laser_base;
        if (!lookupTF(laser_frame_, base_frame_, stamp, ts_laser_base))
            return false;  // base->laser? actually source=base target=laser
                           // => base->laser; we want base->laser and invert
        tf2::Transform T_base_laser = tfFromMsg(ts_laser_base);
        T_laser_base_out = T_base_laser.inverse();
        return true;
    }

    bool solveDeltaIK(const rclcpp::Time& stamp, const tf2::Vector3& p_base,
                      double& dyaw_out, double& dpitch_out)
    {
        // Newton in delta space
        double dyaw = 0.0;
        double dpit = 0.0;

        for (int it = 0; it < ik_max_iter_; ++it) {
            tf2::Transform T_base_laser;
            if (!computeBaseToLaserVirtual(stamp, dyaw, dpit, T_base_laser))
                return false;

            const tf2::Vector3 p_laser =
                T_base_laser * p_base;  // base->laser
            const double fx = p_laser.x();
            const double fy = p_laser.y();

            const double err = std::sqrt(fx * fx + fy * fy);
            if (err < ik_tol_xy_m_) {
                dyaw_out = dyaw;
                dpitch_out = dpit;
                return true;
            }

            const double eps = ik_eps_rad_;

            tf2::Transform T_eps;
            // yaw derivative
            if (!computeBaseToLaserVirtual(stamp, dyaw + eps, dpit, T_eps))
                return false;
            const tf2::Vector3 p_y = T_eps * p_base;

            // pitch derivative
            if (!computeBaseToLaserVirtual(stamp, dyaw, dpit + eps, T_eps))
                return false;
            const tf2::Vector3 p_p = T_eps * p_base;

            const double dfx_dyaw = (p_y.x() - fx) / eps;
            const double dfy_dyaw = (p_y.y() - fy) / eps;
            const double dfx_dpit = (p_p.x() - fx) / eps;
            const double dfy_dpit = (p_p.y() - fy) / eps;

            double step_yaw = 0.0, step_pit = 0.0;
            // J * step = -f
            if (!solve2x2(dfx_dyaw, dfx_dpit, dfy_dyaw, dfy_dpit, -fx, -fy,
                          step_yaw, step_pit))
                break;

            step_yaw = clampAbs(step_yaw, ik_max_step_rad_);
            step_pit = clampAbs(step_pit, ik_max_step_rad_);

            dyaw += step_yaw;
            dpit += step_pit;
        }

        dyaw_out = dyaw;
        dpitch_out = dpit;
        return false;
    }

    void onAimTarget(const auto_aim_fast::msg::AimTarget::SharedPtr msg)
    {
        if (!msg->has_pose)
            return;

        const rclcpp::Time stamp =
            timeFromBuiltin(msg->stamp, this->get_clock());

        // Target in cam frame (assume AimTarget.tx/ty/tz is in cam_frame_)
        const tf2::Vector3 p_cam(msg->tx, msg->ty, msg->tz);

        // cam->base
        tf2::Transform T_cam_base;
        if (!computeCamToBase(stamp, T_cam_base))
            return;

        const tf2::Vector3 p_base = T_cam_base * p_cam;

        // publish target in base
        {
            geometry_msgs::msg::PointStamped out;
            out.header.stamp = stamp;
            out.header.frame_id = base_frame_;
            out.point.x = p_base.x();
            out.point.y = p_base.y();
            out.point.z = p_base.z();
            pub_target_base_->publish(out);
        }

        // current angles as absolute setpoint initial
        double yaw_phy = 0.0;
        double pitch_phy = 0.0;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (has_current_angle_) {
                yaw_phy = rawToPhysicalYaw(yaw_raw_cur_);
                pitch_phy = rawToPhysicalPitch(pitch_raw_cur_);
            }
        }

        // compute delta to make laser z-axis go through target (x=y=0 in
        // laser frame)
        double dyaw = 0.0, dpit = 0.0;
        (void)solveDeltaIK(stamp, p_base, dyaw, dpit);

        const double yaw_cmd_phy = yaw_phy + dyaw;
        const double pitch_cmd_phy = pitch_phy + dpit;

        // output raw command
        double yaw_raw_cmd = physicalToRawYaw(yaw_cmd_phy);
        double pitch_raw_cmd = physicalToRawPitch(pitch_cmd_phy);
        if (output_angle_unit_ == "deg") {
            yaw_raw_cmd = rad2deg(yaw_raw_cmd);
            pitch_raw_cmd = rad2deg(pitch_raw_cmd);
        }

        std_msgs::msg::Float64MultiArray cmd;
        cmd.data.resize(2);
        cmd.data[0] = yaw_raw_cmd;
        cmd.data[1] = pitch_raw_cmd;
        pub_cmd_->publish(cmd);
    }

private:
    // TF
    tf2_ros::Buffer            tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // topics
    std::string aim_target_topic_;
    std::string current_angle_topic_;
    std::string cmd_topic_;
    std::string target_base_topic_;

    // frames
    std::string base_frame_;
    std::string cam_frame_;
    std::string laser_frame_;
    std::string axis1_joint_frame_;
    std::string axis1_frame_;
    std::string axis2_joint_frame_;
    std::string axis2_frame_;

    // angles
    int         yaw_index_{0};
    int         pitch_index_{1};
    std::string current_angle_unit_{"rad"};
    std::string output_angle_unit_{"rad"};
    std::string yaw_axis_{"Z"};
    std::string pitch_axis_{"Y"};
    int         yaw_sign_{1};
    int         pitch_sign_{1};
    double      yaw_offset_{0.0};
    double      pitch_offset_{0.0};

    // IK
    int    ik_max_iter_{15};
    double ik_tol_xy_m_{1e-4};
    double ik_eps_rad_{1e-4};
    double ik_max_step_rad_{0.2};

    double tf_timeout_sec_{0.05};

    // interfaces
    rclcpp::Subscription<auto_aim_fast::msg::AimTarget>::SharedPtr
        sub_target_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
        sub_angles_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
        pub_target_base_;

    // current angle cache
    std::mutex mutex_;
    bool       has_current_angle_{false};
    double     yaw_raw_cur_{0.0};
    double     pitch_raw_cur_{0.0};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalAimSolver>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
