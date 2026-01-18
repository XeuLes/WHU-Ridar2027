#include <rclcpp/rclcpp.hpp>

#include <auto_aim_fast/msg/aim_target.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

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

static tf2::Transform tfFromXYZRPY(const XYZRPY& t)
{
    tf2::Transform T;
    T.setOrigin(tf2::Vector3(t.x, t.y, t.z));
    T.setRotation(quatFromRPY(t.roll, t.pitch, t.yaw));
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

static rclcpp::Time timeFromBuiltin(const builtin_interfaces::msg::Time& t,
                                    rclcpp::Clock::SharedPtr clock)
{
    const int64_t ns = static_cast<int64_t>(t.sec) * 1000000000LL +
                       static_cast<int64_t>(t.nanosec);
    return rclcpp::Time(ns, clock->get_clock_type());
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
}  // namespace

class GimbalAimSolver final : public rclcpp::Node {
public:
    explicit GimbalAimSolver(const rclcpp::NodeOptions& opts)
        : rclcpp::Node("gimbal_aim_solver", opts)
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
        pub_cmd_pred_ =
            this->create_publisher<std_msgs::msg::Float64MultiArray>(
                cmd_pred_topic_, rclcpp::QoS(10));

        pub_target_base_ =
            this->create_publisher<geometry_msgs::msg::PointStamped>(
                target_base_topic_, rclcpp::QoS(10));
        pub_target_base_pred_ =
            this->create_publisher<geometry_msgs::msg::PointStamped>(
                target_base_pred_topic_, rclcpp::QoS(10));

        RCLCPP_INFO(this->get_logger(),
                    "Started. aim_target=%s current_angles=%s cmd=%s "
                    "cmd_pred=%s base_frame=%s",
                    aim_target_topic_.c_str(), current_angle_topic_.c_str(),
                    cmd_topic_.c_str(), cmd_pred_topic_.c_str(),
                    gimbal_base_frame_.c_str());
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
        this->declare_parameter<std::string>("cmd_pred_topic",
                                             "/gimbal/command_angles_pred");

        this->declare_parameter<std::string>("target_base_topic",
                                             "/gimbal/target_base");
        this->declare_parameter<std::string>("target_base_pred_topic",
                                             "/gimbal/target_base_pred");

        this->declare_parameter<std::string>("gimbal_base_frame",
                                             "gimbal_base_frame");

        // output units
        this->declare_parameter<std::string>("output_angle_unit",
                                             "rad");  // rad/deg
        this->declare_parameter<std::string>("current_angle_unit",
                                             "rad");  // rad/deg

        // current angles array format
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);

        // Optional fixed mapping from AimTarget (tx,ty,tz) frame to
        // gimbal_cam frame
        this->declare_parameter<std::vector<double>>(
            "T_targetmsg_to_gimbal_cam_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_targetmsg_to_gimbal_cam_rpy", {0.0, 0.0, 0.0});

        // Kinematic fixed transforms (must match your TF publisher
        // geometry)
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_base_to_axis1_joint_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_base_to_axis1_joint_rpy", {0.0, 0.0, 0.0});

        this->declare_parameter<std::vector<double>>(
            "T_axis1_to_axis2_joint_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_axis1_to_axis2_joint_rpy", {0.0, 0.0, 0.0});

        this->declare_parameter<std::vector<double>>(
            "T_axis2_to_gimbal_cam_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_axis2_to_gimbal_cam_rpy", {0.0, 0.0, 0.0});

        this->declare_parameter<std::vector<double>>(
            "T_gimbal_cam_to_laser_xyz", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "T_gimbal_cam_to_laser_rpy", {0.0, 0.0, 0.0});

        // Axis/offset/sign (must match your TF publisher)
        this->declare_parameter<double>("yaw_offset", 0.0);
        this->declare_parameter<double>("pitch_offset", 0.0);
        this->declare_parameter<std::string>("yaw_axis", "Z");
        this->declare_parameter<std::string>("pitch_axis", "Y");
        this->declare_parameter<int>("yaw_sign", 1);
        this->declare_parameter<int>("pitch_sign", 1);

        // IK
        this->declare_parameter<int>("ik.max_iter", 20);
        this->declare_parameter<double>("ik.tol_xy_m", 1e-4);
        this->declare_parameter<double>("ik.eps_rad", 1e-4);
        this->declare_parameter<double>("ik.max_step_rad", 0.2);

        // prediction
        this->declare_parameter<bool>("prediction.enable", true);
        this->declare_parameter<double>("prediction.max_dt_sec", 0.2);
    }

    void loadParams()
    {
        aim_target_topic_ =
            this->get_parameter("aim_target_topic").as_string();
        current_angle_topic_ =
            this->get_parameter("current_angle_topic").as_string();

        cmd_topic_ = this->get_parameter("cmd_topic").as_string();
        cmd_pred_topic_ = this->get_parameter("cmd_pred_topic").as_string();

        target_base_topic_ =
            this->get_parameter("target_base_topic").as_string();
        target_base_pred_topic_ =
            this->get_parameter("target_base_pred_topic").as_string();

        gimbal_base_frame_ =
            this->get_parameter("gimbal_base_frame").as_string();

        output_angle_unit_ =
            this->get_parameter("output_angle_unit").as_string();
        if (output_angle_unit_ != "deg")
            output_angle_unit_ = "rad";

        current_angle_unit_ =
            this->get_parameter("current_angle_unit").as_string();
        if (current_angle_unit_ != "deg")
            current_angle_unit_ = "rad";

        yaw_index_ = this->get_parameter("yaw_index").as_int();
        pitch_index_ = this->get_parameter("pitch_index").as_int();

        yaw_sign_ =
            (this->get_parameter("yaw_sign").as_int() == -1) ? -1 : 1;
        pitch_sign_ =
            (this->get_parameter("pitch_sign").as_int() == -1) ? -1 : 1;

        yaw_offset_ = this->get_parameter("yaw_offset").as_double();
        pitch_offset_ = this->get_parameter("pitch_offset").as_double();
        yaw_axis_ = this->get_parameter("yaw_axis").as_string();
        pitch_axis_ = this->get_parameter("pitch_axis").as_string();

        T_base_to_axis1_joint_ = tfFromXYZRPY(
            readXYZRPYParam(this, "T_gimbal_base_to_axis1_joint_xyz",
                            "T_gimbal_base_to_axis1_joint_rpy"));
        T_axis1_to_axis2_joint_ =
            tfFromXYZRPY(readXYZRPYParam(this, "T_axis1_to_axis2_joint_xyz",
                                         "T_axis1_to_axis2_joint_rpy"));
        T_axis2_to_cam_ =
            tfFromXYZRPY(readXYZRPYParam(this, "T_axis2_to_gimbal_cam_xyz",
                                         "T_axis2_to_gimbal_cam_rpy"));
        T_cam_to_laser_ =
            tfFromXYZRPY(readXYZRPYParam(this, "T_gimbal_cam_to_laser_xyz",
                                         "T_gimbal_cam_to_laser_rpy"));
        T_targetmsg_to_cam_ = tfFromXYZRPY(
            readXYZRPYParam(this, "T_targetmsg_to_gimbal_cam_xyz",
                            "T_targetmsg_to_gimbal_cam_rpy"));

        const int64_t max_iter_i64 =
            this->get_parameter("ik.max_iter").as_int();
        ik_max_iter_ = std::max<int>(
            1, static_cast<int>(max_iter_i64));  // FIX: int64_t -> int
        ik_tol_xy_m_ = this->get_parameter("ik.tol_xy_m").as_double();
        ik_eps_rad_ = this->get_parameter("ik.eps_rad").as_double();
        ik_max_step_rad_ =
            this->get_parameter("ik.max_step_rad").as_double();
        if (ik_eps_rad_ <= 0.0)
            ik_eps_rad_ = 1e-4;
        if (ik_max_step_rad_ <= 0.0)
            ik_max_step_rad_ = 0.2;

        pred_enable_ = this->get_parameter("prediction.enable").as_bool();
        pred_max_dt_sec_ =
            this->get_parameter("prediction.max_dt_sec").as_double();
        if (pred_max_dt_sec_ <= 0.0)
            pred_max_dt_sec_ = 0.2;
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

    void forwardKinematics(double yaw_phy, double pitch_phy,
                           tf2::Transform& T_base_cam,
                           tf2::Transform& T_base_laser) const
    {
        tf2::Transform T = T_base_to_axis1_joint_;

        tf2::Transform R_yaw;
        R_yaw.setOrigin(tf2::Vector3(0, 0, 0));
        R_yaw.setRotation(axisAngleQuat(yaw_axis_, yaw_phy));
        T *= R_yaw;

        T *= T_axis1_to_axis2_joint_;

        tf2::Transform R_pitch;
        R_pitch.setOrigin(tf2::Vector3(0, 0, 0));
        R_pitch.setRotation(axisAngleQuat(pitch_axis_, pitch_phy));
        T *= R_pitch;

        T_base_cam = T * T_axis2_to_cam_;
        T_base_laser = T_base_cam * T_cam_to_laser_;
    }

    bool solveIK(const tf2::Vector3& target_base, double& yaw_phy_io,
                 double& pitch_phy_io) const
    {
        double yaw = yaw_phy_io;
        double pitch = pitch_phy_io;

        for (int it = 0; it < ik_max_iter_; ++it) {
            tf2::Transform T_base_cam, T_base_laser;
            forwardKinematics(yaw, pitch, T_base_cam, T_base_laser);

            const tf2::Vector3 p_laser =
                T_base_laser.inverse() * target_base;
            const double fx = p_laser.x();
            const double fy = p_laser.y();

            const double err = std::sqrt(fx * fx + fy * fy);
            if (err < ik_tol_xy_m_) {
                yaw_phy_io = yaw;
                pitch_phy_io = pitch;
                return true;
            }

            const double eps = ik_eps_rad_;

            tf2::Transform T_bc2, T_bl2;
            forwardKinematics(yaw + eps, pitch, T_bc2, T_bl2);
            const tf2::Vector3 p_y = T_bl2.inverse() * target_base;

            forwardKinematics(yaw, pitch + eps, T_bc2, T_bl2);
            const tf2::Vector3 p_p = T_bl2.inverse() * target_base;

            const double dfx_dyaw = (p_y.x() - fx) / eps;
            const double dfy_dyaw = (p_y.y() - fy) / eps;
            const double dfx_dpit = (p_p.x() - fx) / eps;
            const double dfy_dpit = (p_p.y() - fy) / eps;

            double dyaw = 0.0, dpit = 0.0;
            if (!solve2x2(dfx_dyaw, dfx_dpit, dfy_dyaw, dfy_dpit, -fx, -fy,
                          dyaw, dpit))
                break;

            dyaw = clampAbs(dyaw, ik_max_step_rad_);
            dpit = clampAbs(dpit, ik_max_step_rad_);
            yaw += dyaw;
            pitch += dpit;
        }

        return false;
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

    void onAimTarget(const auto_aim_fast::msg::AimTarget::SharedPtr msg)
    {
        if (!msg->has_pose)
            return;

        const rclcpp::Time stamp =
            timeFromBuiltin(msg->stamp, this->get_clock());

        // target in message frame
        const tf2::Vector3 p_msg(msg->tx, msg->ty, msg->tz);
        const tf2::Vector3 p_cam = T_targetmsg_to_cam_ * p_msg;

        // init from current angles if available
        double yaw_phy0 = 0.0, pitch_phy0 = 0.0;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (has_current_angle_) {
                yaw_phy0 = rawToPhysicalYaw(yaw_raw_cur_);
                pitch_phy0 = rawToPhysicalPitch(pitch_raw_cur_);
            }
        }

        tf2::Transform T_base_cam, T_base_laser;
        forwardKinematics(yaw_phy0, pitch_phy0, T_base_cam, T_base_laser);

        const tf2::Vector3 target_base = T_base_cam * p_cam;
        publishPoint(pub_target_base_, stamp, target_base);

        double yaw_cmd = yaw_phy0;
        double pitch_cmd = pitch_phy0;
        (void)solveIK(target_base, yaw_cmd, pitch_cmd);
        publishCmd(pub_cmd_, yaw_cmd, pitch_cmd);

        // prediction
        tf2::Vector3 target_base_pred = target_base;
        if (pred_enable_) {
            double dt = 0.0;
            if (msg->latency_ns > 0)
                dt += static_cast<double>(msg->latency_ns) / 1e9;
            if (msg->extra_latency_ns > 0)
                dt += static_cast<double>(msg->extra_latency_ns) / 1e9;
            dt = std::clamp(dt, 0.0, pred_max_dt_sec_);

            tf2::Vector3 vel(0, 0, 0);
            {
                std::lock_guard<std::mutex> lk(pred_mutex_);
                if (pred_has_last_ && (stamp > pred_last_stamp_)) {
                    const double dtm = (stamp - pred_last_stamp_).seconds();
                    if (dtm > 1e-6)
                        vel = (target_base - pred_last_base_) * (1.0 / dtm);
                }
                pred_last_base_ = target_base;
                pred_last_stamp_ = stamp;
                pred_has_last_ = true;
            }

            target_base_pred = target_base + vel * dt;
        }

        publishPoint(pub_target_base_pred_, stamp, target_base_pred);

        double yaw_cmd_p = yaw_cmd;
        double pitch_cmd_p = pitch_cmd;
        (void)solveIK(target_base_pred, yaw_cmd_p, pitch_cmd_p);
        publishCmd(pub_cmd_pred_, yaw_cmd_p, pitch_cmd_p);
    }

    void publishPoint(const rclcpp::Publisher<
                          geometry_msgs::msg::PointStamped>::SharedPtr& pub,
                      const rclcpp::Time& stamp, const tf2::Vector3& p_base)
    {
        geometry_msgs::msg::PointStamped out;
        out.header.stamp = stamp;
        out.header.frame_id = gimbal_base_frame_;
        out.point.x = p_base.x();
        out.point.y = p_base.y();
        out.point.z = p_base.z();
        pub->publish(out);
    }

    void publishCmd(const rclcpp::Publisher<
                        std_msgs::msg::Float64MultiArray>::SharedPtr& pub,
                    double yaw_phy, double pitch_phy)
    {
        double yaw_raw = physicalToRawYaw(yaw_phy);
        double pitch_raw = physicalToRawPitch(pitch_phy);

        if (output_angle_unit_ == "deg") {
            yaw_raw = rad2deg(yaw_raw);
            pitch_raw = rad2deg(pitch_raw);
        }

        std_msgs::msg::Float64MultiArray out;
        out.data.resize(2);
        out.data[0] = yaw_raw;
        out.data[1] = pitch_raw;
        pub->publish(out);
    }

private:
    // topics
    std::string aim_target_topic_;
    std::string current_angle_topic_;
    std::string cmd_topic_;
    std::string cmd_pred_topic_;
    std::string target_base_topic_;
    std::string target_base_pred_topic_;
    std::string gimbal_base_frame_;

    // units/array indices
    std::string output_angle_unit_{"rad"};
    std::string current_angle_unit_{"rad"};
    int         yaw_index_{0};
    int         pitch_index_{1};

    // axis/offset/sign
    int         yaw_sign_{1};
    int         pitch_sign_{1};
    double      yaw_offset_{0.0};
    double      pitch_offset_{0.0};
    std::string yaw_axis_{"Z"};
    std::string pitch_axis_{"Y"};

    // transforms
    tf2::Transform T_base_to_axis1_joint_;
    tf2::Transform T_axis1_to_axis2_joint_;
    tf2::Transform T_axis2_to_cam_;
    tf2::Transform T_cam_to_laser_;
    tf2::Transform T_targetmsg_to_cam_;

    // IK params
    int    ik_max_iter_{20};
    double ik_tol_xy_m_{1e-4};
    double ik_eps_rad_{1e-4};
    double ik_max_step_rad_{0.2};

    // prediction
    bool   pred_enable_{true};
    double pred_max_dt_sec_{0.2};

    // interfaces
    rclcpp::Subscription<auto_aim_fast::msg::AimTarget>::SharedPtr
        sub_target_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
        sub_angles_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
        pub_cmd_pred_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
        pub_target_base_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
        pub_target_base_pred_;

    // current angles cache
    std::mutex mutex_;
    bool       has_current_angle_{false};
    double     yaw_raw_cur_{0.0};
    double     pitch_raw_cur_{0.0};

    // prediction cache
    std::mutex   pred_mutex_;
    bool         pred_has_last_{false};
    tf2::Vector3 pred_last_base_{0, 0, 0};
    rclcpp::Time pred_last_stamp_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalAimSolver>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
