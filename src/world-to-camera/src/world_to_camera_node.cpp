#include <filesystem>
#include <fstream>
#include <string>
#include "Eigen/Dense"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class WorldToCameraNode : public rclcpp::Node {
public:
    WorldToCameraNode() : Node("world_to_camera_calib_node")
    {
        // 声明参数
        this->declare_parameter<std::string>("json_path", "");
        this->declare_parameter<std::string>("output_yaml_path", "");
        this->declare_parameter<double>("tf_update_interval", 1.0);
        this->declare_parameter<double>("tf_timeout", 0.5);
        this->declare_parameter<double>("tf_delay_tolerance", 0.1);

        // 获取参数
        std::string json_path =
            this->get_parameter("json_path").as_string();
        output_yaml_path_ =
            this->get_parameter("output_yaml_path").as_string();
        double tf_update_interval =
            this->get_parameter("tf_update_interval").as_double();
        tf_timeout_ = this->get_parameter("tf_timeout").as_double();
        tf_delay_tolerance_ =
            this->get_parameter("tf_delay_tolerance").as_double();

        // 确保输出目录存在
        fs::path output_path(output_yaml_path_);
        fs::create_directories(output_path.parent_path());

        // 检查JSON文件
        if (!fs::exists(json_path)) {
            RCLCPP_ERROR(this->get_logger(), "JSON file not found: %s",
                         json_path.c_str());
            return;
        }

        // 读取并修正lidar→camera的变换（关键修正）
        if (!readAndCorrectLidarToCameraTransform(json_path)) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to process lidar→camera transform");
            return;
        }

        // 初始化TF
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(),
                    "Node started. Listening for rm_frame→livox_frame...");
        startContinuousTfListening(tf_update_interval);
    }

private:
    // 读取JSON中的camera→lidar变换，并修正为lidar→camera（含Y轴翻转）
    bool readAndCorrectLidarToCameraTransform(const std::string& json_path)
    {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open JSON: %s",
                         json_path.c_str());
            return false;
        }

        json data;
        try {
            file >> data;
        }
        catch (const json::parse_error& e) {
            RCLCPP_ERROR(this->get_logger(), "JSON parse error: %s",
                         e.what());
            return false;
        }

        if (!data.contains("results") ||
            !data["results"].contains("T_lidar_camera")) {
            RCLCPP_ERROR(this->get_logger(),
                         "Missing T_lidar_camera in JSON");
            return false;
        }

        auto& t = data["results"]["T_lidar_camera"];
        if (t.size() != 7) {
            RCLCPP_ERROR(this->get_logger(),
                         "T_lidar_camera must have 7 elements, got %zu",
                         t.size());
            return false;
        }

        // 步骤1：解析JSON中的变换（实际是camera→lidar，记为T_cl）
        Eigen::Vector3d t_cl(t[0], t[1], t[2]);  // camera→lidar的平移
        Eigen::Quaterniond q_cl(t[6], t[3], t[4],
                                t[5]);  // camera→lidar的旋转（w, x, y, z）
        q_cl.normalize();               // 确保单位四元数

        // 步骤2：计算T_cl的逆变换，得到lidar→camera的变换（T_lc = T_cl⁻¹）
        Eigen::Quaterniond q_lc = q_cl.conjugate();  // 旋转逆（四元数共轭）
        Eigen::Matrix3d R_cl =
            q_cl.toRotationMatrix();  // camera→lidar的旋转矩阵
        Eigen::Vector3d t_lc =
            -R_cl.transpose() * t_cl;  // 平移逆（关键公式）

        // 步骤3：修正Y轴方向（相机Y轴与激光雷达Y轴相反）
        Eigen::Matrix3d flip_y;  // Y轴翻转矩阵
        flip_y << 1, 0, 0, 0, -1, 0, 0, 0, 1;
        // 应用Y轴翻转到旋转和平移
        q_lc = Eigen::Quaterniond(flip_y *
                                  q_lc.toRotationMatrix());  // 旋转修正
        t_lc = flip_y * t_lc;                                // 平移修正

        // 步骤4：构建最终的lidar→camera变换矩阵
        T_lidar_camera_.setIdentity();
        T_lidar_camera_.rotate(q_lc);
        T_lidar_camera_.translate(t_lc);

        // 输出修正后的变换信息（调试用）
        RCLCPP_INFO(this->get_logger(),
                    "Corrected lidar→camera transform:");
        RCLCPP_INFO(this->get_logger(), "  Translation: [%.3f, %.3f, %.3f]",
                    t_lc.x(), t_lc.y(), t_lc.z());
        RCLCPP_INFO(
            this->get_logger(),
            "  Rotation (quaternion w,x,y,z): [%.3f, %.3f, %.3f, %.3f]",
            q_lc.w(), q_lc.x(), q_lc.y(), q_lc.z());
        return true;
    }

    void startContinuousTfListening(double interval)
    {
        tf_timer_ =
            this->create_wall_timer(std::chrono::duration<double>(interval),
                                    [this]() { continuousTfLookup(); });
    }

    void continuousTfLookup()
    {
        static bool         first_success = false;
        static rclcpp::Time last_update = this->now();

        try {
            rclcpp::Time target_time =
                this->now() -
                rclcpp::Duration::from_seconds(tf_delay_tolerance_);
            if (tf_buffer_->canTransform("rm_frame", "livox_frame",
                                         target_time)) {
                auto transform = tf_buffer_->lookupTransform(
                    "rm_frame", "livox_frame", target_time,
                    rclcpp::Duration::from_seconds(tf_timeout_));

                auto now = this->now();
                if (!first_success || (now - last_update).seconds() > 5.0) {
                    // 转换TF为world→lidar变换矩阵
                    Eigen::Isometry3d T_world_lidar =
                        tf2::transformToEigen(transform);

                    // 计算世界→相机变换：T_world_camera =
                    // T_lidar_camera_（lidar→camera） *
                    // T_world_lidar（world→lidar）
                    Eigen::Isometry3d T_world_camera =
                        T_lidar_camera_ * T_world_lidar;

                    saveResult(T_world_camera);

                    if (!first_success) {
                        RCLCPP_INFO(
                            this->get_logger(),
                            "Initial world→camera transform saved to: %s",
                            output_yaml_path_.c_str());
                        first_success = true;
                    } else {
                        RCLCPP_DEBUG(this->get_logger(),
                                     "Updated world→camera transform "
                                     "(%.1fs since last update)",
                                     (now - last_update).seconds());
                    }
                    last_update = now;
                }
            }
        }
        catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                                 5000, "TF error: %s", ex.what());
        }
    }

    void saveResult(const Eigen::Isometry3d& T)
    {
        Eigen::Matrix3d R = T.rotation();     // 提取旋转矩阵
        Eigen::Vector3d t = T.translation();  // 提取平移向量

        // 旋转矩阵转旋转向量（Rodrigues变换）
        cv::Mat cv_R(3, 3, CV_64F), cv_rvec(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cv_R.at<double>(i, j) = R(i, j);
        cv::Rodrigues(cv_R, cv_rvec);

        // 写入YAML文件
        std::ofstream f(output_yaml_path_);
        if (!f.is_open()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to open output YAML: %s",
                         output_yaml_path_.c_str());
            return;
        }

        f << "%YAML:1.0\n---\n";
        f << "world_rvec: !!opencv-matrix\n";
        f << "  rows: 3\n  cols: 1\n  dt: d\n  data: [ "
          << cv_rvec.at<double>(0) << ", " << cv_rvec.at<double>(1) << ", "
          << cv_rvec.at<double>(2) << " ]\n";
        f << "world_tvec: !!opencv-matrix\n";
        f << "  rows: 3\n  cols: 1\n  dt: d\n  data: [ " << t(0) << ", "
          << t(1) << ", " << t(2) << " ]\n";
        f.close();
    }

    std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr                tf_timer_;
    Eigen::Isometry3d T_lidar_camera_;  // 修正后的lidar→camera变换矩阵
    std::string       output_yaml_path_;
    double            tf_timeout_;
    double            tf_delay_tolerance_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WorldToCameraNode>());
    rclcpp::shutdown();
    return 0;
}
