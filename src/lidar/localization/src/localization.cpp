#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
// 新增：YAML文件读写和文件系统头文件
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

namespace tdt_radar {
struct Grid {
    pcl::PointXYZ farthestPoint;
    double        maxDistance = -1.0;
};
class Localization : public rclcpp::Node {
public:
    Localization(const rclcpp::NodeOptions& node_options)
        : Node("localization", node_options)
    {
        // 1. 加载目标地图点云（场地点云）
        std::string target_pcd_file = "/home/kakaka/code/WHU-Radar-2026/src/lidar/localization/config/aligned.pcd";
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(target_pcd_file, *target_cloud_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load %s", target_pcd_file.c_str());
            return;
        }
        // 目标点云仅下采样一次（避免重复下采样导致稀疏）
        downsample_cloud(target_cloud_, target_cloud_, 0.1f);
        RCLCPP_INFO(this->get_logger(), "目标地图加载完成，下采样后点数：%lu", target_cloud_->size());

        // 2. 初始化订阅器/发布器
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10, std::bind(&Localization::callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/map", 10);
        filter_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filter_map", 10);
        manual_aligned_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/manual_aligned_cloud", 10);

        // 3. 地图发布计时器
        timer_ = this->create_wall_timer(std::chrono::seconds(10), [this]() {
            sensor_msgs::msg::PointCloud2 target_msg;
            pcl::toROSMsg(*target_cloud_, target_msg);
            target_msg.header.frame_id = "rm_frame";
            publisher_->publish(target_msg);
        });

        // 4. 初始化TF广播
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // 5. 声明手动配准参数（含参数保存/加载相关）
        declare_manual_params();
        // 6. 读取配置文件路径，加载历史参数
        this->get_parameter("manual_calib_config_path", manual_calib_config_path_);
        load_manual_params(); // 启动时自动加载参数

        // 7. 设置参数回调（实时响应参数修改）
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&Localization::param_callback, this, std::placeholders::_1));

        // 8. 参数重置定时器（处理trigger和save_trigger的重置）
        param_reset_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&Localization::param_reset_callback, this));

        // 打印功能提示
        RCLCPP_INFO(this->get_logger(), "手动配准模式已启动！（支持参数持久化）");
        RCLCPP_INFO(this->get_logger(), "预览话题：/livox/manual_aligned_cloud");
        RCLCPP_INFO(this->get_logger(), "参数配置文件：%s", manual_calib_config_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "可用参数列表：");
        RCLCPP_INFO(this->get_logger(), "  manual_trans_x/y/z: 平移（米，范围±100）");
        RCLCPP_INFO(this->get_logger(), "  manual_roll/pitch/yaw_deg: 旋转（度）");
        RCLCPP_INFO(this->get_logger(), "  manual_mode: 手动模式开关（true/false）");
        RCLCPP_INFO(this->get_logger(), "  trigger_auto_calib: 触发自动精配准（true）");
        RCLCPP_INFO(this->get_logger(), "  param_save_trigger: 保存当前参数到文件（true）");
    }

private:
    // -------------------------- 参数重置定时器回调 --------------------------
    void param_reset_callback() {
        // 重置自动配准触发参数
        if (need_reset_trigger_) {
            this->set_parameter(rclcpp::Parameter("trigger_auto_calib", false));
            need_reset_trigger_ = false;
        }
        // 重置参数保存触发参数
        if (need_reset_save_trigger_) {
            this->set_parameter(rclcpp::Parameter("param_save_trigger", false));
            need_reset_save_trigger_ = false;
        }
    }

    // -------------------------- 声明手动配准参数（含保存/加载参数） --------------------------
    void declare_manual_params() {
        // 平移参数（单位：米）
        this->declare_parameter<double>("manual_trans_x", 0.0);
        this->declare_parameter<double>("manual_trans_y", 0.0);
        this->declare_parameter<double>("manual_trans_z", 0.0);
        // 旋转参数（单位：度）
        this->declare_parameter<double>("manual_roll_deg", 0.0);
        this->declare_parameter<double>("manual_pitch_deg", 0.0);
        this->declare_parameter<double>("manual_yaw_deg", 0.0);
        // 模式控制参数
        this->declare_parameter<bool>("manual_mode", true);
        this->declare_parameter<bool>("trigger_auto_calib", false);
        // 新增：参数保存触发和配置文件路径
        this->declare_parameter<bool>("param_save_trigger", false);
        this->declare_parameter<std::string>(
            "manual_calib_config_path",
            "/home/kakaka/code/WHU-Radar-2026/src/lidar/localization/config/manual_calib_params.yaml"
        );
        // 新增：控制是否自动触发配准（默认true，即自动触发）
        this->declare_parameter<bool>("auto_trigger_after_load", true);
    }

    // -------------------------- 新增：同步节点参数到成员变量（关键修复） --------------------------
    void sync_params_to_member() {
        // 从节点参数中读取当前值，同步到成员变量
        manual_trans_x_ = this->get_parameter("manual_trans_x").get_value<double>();
        manual_trans_y_ = this->get_parameter("manual_trans_y").get_value<double>();
        manual_trans_z_ = this->get_parameter("manual_trans_z").get_value<double>();
        
        // 旋转参数：节点参数是“度”，成员变量是“弧度”，需转换
        double roll_deg = this->get_parameter("manual_roll_deg").get_value<double>();
        double pitch_deg = this->get_parameter("manual_pitch_deg").get_value<double>();
        double yaw_deg = this->get_parameter("manual_yaw_deg").get_value<double>();
        manual_roll_ = roll_deg * M_PI / 180.0;
        manual_pitch_ = pitch_deg * M_PI / 180.0;
        manual_yaw_ = yaw_deg * M_PI / 180.0;
        
        manual_mode_ = this->get_parameter("manual_mode").get_value<bool>();

        // 打印同步后的成员变量（验证是否生效）
        RCLCPP_INFO(this->get_logger(), "成员变量同步完成：");
        RCLCPP_INFO(this->get_logger(), "  trans(x:%.2f, y:%.2f, z:%.2f)",
                    manual_trans_x_, manual_trans_y_, manual_trans_z_);
        RCLCPP_INFO(this->get_logger(), "  roll:%.1f° (%.3frad) | pitch:%.1f° (%.3frad) | yaw:%.1f° (%.3frad)",
                    roll_deg, manual_roll_, pitch_deg, manual_pitch_, yaw_deg, manual_yaw_);
        RCLCPP_INFO(this->get_logger(), "  manual_mode: %s", manual_mode_ ? "true" : "false");
    }

    // -------------------------- 加载历史参数（从YAML文件）+ 同步成员变量（关键修复） --------------------------
    void load_manual_params() {
        // 输出实际读取的文件路径（方便排查路径错误）
        RCLCPP_INFO(this->get_logger(), "尝试加载参数文件：%s", manual_calib_config_path_.c_str());

        // 检查文件是否存在
        if (!std::filesystem::exists(manual_calib_config_path_)) {
            RCLCPP_WARN(this->get_logger(), "参数文件不存在！将使用默认值");
            // 即使无文件，也同步默认参数到成员变量
            sync_params_to_member();
            return;
        }

        try {
            // 读取YAML文件
            YAML::Node config = YAML::LoadFile(manual_calib_config_path_);

            // 定义需要加载的参数（参数名必须和节点声明完全一致）
            std::map<std::string, double> double_params = {
                {"manual_trans_x", 0.0}, {"manual_trans_y", 0.0}, {"manual_trans_z", 0.0},
                {"manual_roll_deg", 0.0}, {"manual_pitch_deg", 0.0}, {"manual_yaw_deg", 0.0}
            };
            std::map<std::string, bool> bool_params = {
                {"manual_mode", true}
            };

            // 加载浮点型参数
            for (auto& [param_name, default_val] : double_params) {
                double val = default_val;
                if (config[param_name]) {
                    val = config[param_name].as<double>();
                    RCLCPP_INFO(this->get_logger(), "从文件加载参数 %-15s: %.2f", param_name.c_str(), val);
                } else {
                    RCLCPP_WARN(this->get_logger(), "文件中无参数 %s，使用默认值 %.2f", param_name.c_str(), default_val);
                }
                // 设置到节点参数
                this->set_parameter(rclcpp::Parameter(param_name, val));
            }

            // 加载布尔型参数
            for (auto& [param_name, default_val] : bool_params) {
                bool val = default_val;
                if (config[param_name]) {
                    val = config[param_name].as<bool>();
                    RCLCPP_INFO(this->get_logger(), "从文件加载参数 %-15s: %s", param_name.c_str(), val ? "true" : "false");
                } else {
                    RCLCPP_WARN(this->get_logger(), "文件中无参数 %s，使用默认值 %s", param_name.c_str(), default_val ? "true" : "false");
                }
                // 设置到节点参数
                this->set_parameter(rclcpp::Parameter(param_name, val));
            }

            // 关键修复：加载参数后，同步到成员变量
            sync_params_to_member();

            RCLCPP_INFO(this->get_logger(), "参数文件加载并同步完成！");
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "参数文件解析失败（格式错误）：%s", e.what());
            // 解析失败时，使用默认参数同步成员变量
            sync_params_to_member();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "加载参数失败：%s", e.what());
            sync_params_to_member();
        }

        // 读取自动触发控制参数（允许用户通过文件/命令行关闭）
        auto_trigger_calib_after_load_ = this->get_parameter("auto_trigger_after_load").get_value<bool>();
        if (auto_trigger_calib_after_load_) {
            RCLCPP_INFO(this->get_logger(), "已开启「加载参数后自动触发精配准」");
            RCLCPP_INFO(this->get_logger(), "第一帧雷达点云到来后，将自动使用加载的位姿启动GICP配准");
        } else {
            RCLCPP_INFO(this->get_logger(), "已关闭「自动触发精配准」，将保持手动模式");
        }
    }


    // -------------------------- 保存当前参数（到YAML文件） --------------------------
    void save_manual_params() {
        try {
            // 创建父目录（避免目录不存在导致保存失败）
            std::filesystem::path config_path(manual_calib_config_path_);
            std::filesystem::create_directories(config_path.parent_path());

            // 读取当前节点的参数值
            YAML::Node config;
            config["manual_trans_x"] = this->get_parameter("manual_trans_x").get_value<double>();
            config["manual_trans_y"] = this->get_parameter("manual_trans_y").get_value<double>();
            config["manual_trans_z"] = this->get_parameter("manual_trans_z").get_value<double>();
            config["manual_roll_deg"] = this->get_parameter("manual_roll_deg").get_value<double>();
            config["manual_pitch_deg"] = this->get_parameter("manual_pitch_deg").get_value<double>();
            config["manual_yaw_deg"] = this->get_parameter("manual_yaw_deg").get_value<double>();
            config["manual_mode"] = this->get_parameter("manual_mode").get_value<bool>();

            // 写入文件
            std::ofstream fout(manual_calib_config_path_);
            if (!fout.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "无法打开文件：%s", manual_calib_config_path_.c_str());
                return;
            }
            fout << config; // 写入YAML（格式自动美化）
            fout.close();

            RCLCPP_INFO(this->get_logger(), "参数已保存到：%s", manual_calib_config_path_.c_str());
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "保存参数文件失败：%s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "保存参数失败：%s", e.what());
        }
    }

    // -------------------------- 参数回调（处理参数修改和保存触发） --------------------------
    rcl_interfaces::msg::SetParametersResult param_callback(
        const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto& param : parameters) {
            // 平移参数更新
            if (param.get_name() == "manual_trans_x") update_param_with_range(param, manual_trans_x_, -100.0, 100.0);
            else if (param.get_name() == "manual_trans_y") update_param_with_range(param, manual_trans_y_, -100.0, 100.0);
            else if (param.get_name() == "manual_trans_z") update_param_with_range(param, manual_trans_z_, -100.0, 100.0);
            // 旋转参数更新（度转弧度）
            else if (param.get_name() == "manual_roll_deg") {
                double deg = param.as_double();
                if (deg < -180 || deg > 180) { result.successful = false; result.reason = "roll范围±180度"; return result; }
                manual_roll_ = deg * M_PI / 180.0;
            }
            else if (param.get_name() == "manual_pitch_deg") {
                double deg = param.as_double();
                if (deg < -90 || deg > 90) { result.successful = false; result.reason = "pitch范围±90度"; return result; }
                manual_pitch_ = deg * M_PI / 180.0;
            }
            else if (param.get_name() == "manual_yaw_deg") {
                double deg = param.as_double();
                if (deg < -180 || deg > 180) { result.successful = false; result.reason = "yaw范围±180度"; return result; }
                manual_yaw_ = deg * M_PI / 180.0;
            }
            // 模式切换
            else if (param.get_name() == "manual_mode") {
                manual_mode_ = param.as_bool();
                RCLCPP_WARN(this->get_logger(), "切换到【%s】", manual_mode_ ? "手动配准模式" : "自动精配准模式");
                if (manual_mode_) has_aligned_ = false;
            }
            // 触发自动精配准
            else if (param.get_name() == "trigger_auto_calib" && param.as_bool()) {
                trigger_auto_calib_ = true;
                need_reset_trigger_ = true;
            }
            // 触发参数保存（核心新增）
            else if (param.get_name() == "param_save_trigger" && param.as_bool()) {
                param_save_trigger_ = true;
                need_reset_save_trigger_ = true;
                save_manual_params(); // 立即保存当前参数
            }
        }
        return result;
    }

    // -------------------------- 辅助函数：参数范围检查 --------------------------
    template<typename T>
    void update_param_with_range(const rclcpp::Parameter& param, T& var, T min, T max) {
        T val = param.as_double();
        if (val < min || val > max) {
            RCLCPP_WARN(this->get_logger(), "%s超出范围[%f, %f]，忽略修改", param.get_name().c_str(), min, max);
            return;
        }
        var = val;
    }

    // -------------------------- 生成手动变换矩阵 --------------------------
    Eigen::Matrix4f create_manual_transform() {
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        // 旋转矩阵：roll(X) → pitch(Y) → yaw(Z)
        Eigen::AngleAxisf roll(manual_roll_, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitch(manual_pitch_, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yaw(manual_yaw_, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = yaw * pitch * roll;
        transform.block<3, 3>(0, 0) = q.matrix();
        // 平移向量
        transform(0, 3) = manual_trans_x_;
        transform(1, 3) = manual_trans_y_;
        transform(2, 3) = manual_trans_z_;
        return transform;
    }

    // -------------------------- 原有核心功能（无修改） --------------------------
    void downsample_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                          float leaf_size) {
        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxelgrid.setInputCloud(input);
        voxelgrid.filter(*output);
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 转换原始雷达点云到PCL格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *source_cloud);

        // -------------------------- 手动配准模式 --------------------------
        if (manual_mode_) {
            // 1. 生成手动变换矩阵
            Eigen::Matrix4f manual_transform = create_manual_transform();

            // 2. 应用变换到雷达点云
            pcl::PointCloud<pcl::PointXYZ>::Ptr manual_aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*source_cloud, *manual_aligned_cloud, manual_transform);

            // // 3. 发布手动调整后的点云
            // sensor_msgs::msg::PointCloud2 manual_msg;
            // pcl::toROSMsg(*manual_aligned_cloud, manual_msg);
            // manual_msg.header.frame_id = "rm_frame";
            // manual_msg.header.stamp = msg->header.stamp;
            // manual_aligned_publisher_->publish(manual_msg);

            // 4. 发布手动TF
            publishTF(manual_transform);

            // 5. 打印当前参数（1秒一次）
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "手动配准参数：trans(x:%.2f, y:%.2f, z:%.2f) | "
                                "roll:%.1f° pitch:%.1f° yaw:%.1f°",
                                manual_trans_x_, manual_trans_y_, manual_trans_z_,
                                manual_roll_ * 180/M_PI, manual_pitch_*180/M_PI, manual_yaw_*180/M_PI);

            // 新增：自动触发配准（仅在参数加载后触发一次）
            if (auto_trigger_calib_after_load_ && !has_auto_triggered_ && !trigger_auto_calib_) {
                RCLCPP_WARN(this->get_logger(), "触发自动精配准（来自参数加载后自动触发）");
                manual_mode_ = false;
                initial_transform_ = manual_transform;  // 使用加载的手动位姿作为初始猜测
                gridMap.clear();
                accumulated_clouds_.clear();
                has_auto_triggered_ = true;  // 标记已触发，避免重复
                return;  // 直接进入自动配准模式
            }
            // 6. 触发自动精配准
            if (trigger_auto_calib_) {
                manual_mode_ = false;
                initial_transform_ = manual_transform;
                RCLCPP_WARN(this->get_logger(), "触发自动精配准！初始位姿已设置为手动调整值");
                gridMap.clear();
                accumulated_clouds_.clear();
                trigger_auto_calib_ = false;
            }
            return;
        }

        // -------------------------- 自动精配准模式 --------------------------
        if (!has_aligned_) {
            // 0. 每帧先清空栅格地图（避免旧点叠加）
            gridMap.clear();

            // 1. 累积20帧点云
            if (accumulated_clouds_.size() < accumulate_time) {
                accumulated_clouds_.push_back(source_cloud);
                return;
            } else {
                accumulated_clouds_.erase(accumulated_clouds_.begin());
                accumulated_clouds_.push_back(source_cloud);
            }

            // 2. 栅格地图构建
            for (auto& cloud : accumulated_clouds_) {
                for (const auto& point : *cloud) {
                    if (point.x <= 0) continue;
                    double azimuth = std::atan2(point.y, point.x);
                    double elevation = std::atan2(point.z, std::sqrt(point.x*point.x + point.y*point.y));
                    int az_idx = static_cast<int>(floor(azimuth * 180/M_PI / gridSizeDegrees));
                    int el_idx = static_cast<int>(floor(elevation * 180/M_PI / gridSizeDegrees));
                    double dist = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);

                    auto& grid = gridMap[{az_idx, el_idx}];
                    if (dist > grid.maxDistance) {
                        grid.farthestPoint = point;
                        grid.maxDistance = dist;
                    }
                }
            }

            // 3. 提取栅格有效点
            pcl::PointCloud<pcl::PointXYZ>::Ptr grid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& [key, grid] : gridMap) {
                if (grid.maxDistance > 0.0) grid_cloud->push_back(grid.farthestPoint);
            }

            // 4. ROI过滤
            pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& point : grid_cloud->points) {
                if (point.x > 10 && point.x < 20 && point.y > -10 && point.y < 8 && point.z < 7) {
                    roi_cloud->push_back(point);
                }
            }
            RCLCPP_WARN(this->get_logger(), "ROI过滤后点数：%lu", roi_cloud->size());
            if (roi_cloud->empty()) {
                RCLCPP_ERROR(this->get_logger(), "ROI过滤后无有效点！请调整范围");
                return;
            }

            // 5. 源点云下采样
            pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_source(new pcl::PointCloud<pcl::PointXYZ>());
            downsample_cloud(roi_cloud, downsampled_source, 0.3f);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "下采样后点数：%lu", downsampled_source->size());  // 新增日志，验证减点效果

            // // 6. 发布过滤后的点云
            // sensor_msgs::msg::PointCloud2 filter_msg;
            // pcl::toROSMsg(*downsampled_source, filter_msg);
            // filter_msg.header.frame_id = "livox_frame";
            // filter_publisher_->publish(filter_msg);

            // 7. GICP精配准
            RCLCPP_INFO(this->get_logger(), "开始GICP精配准（初始位姿来自手动调整）");
            auto gicp = pcl::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
            gicp->setMaximumIterations(500);
            gicp->setTransformationEpsilon(1e-8);
            gicp->setEuclideanFitnessEpsilon(1e-6);
            transform = align(gicp, target_cloud_, downsampled_source, initial_transform_);
        }

        // 发布最终TF变换
        publishTF(transform);
    }

    Eigen::Matrix4f
    align(pcl::shared_ptr<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> gicp,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
          const Eigen::Matrix4f& initial_transform) {
        gicp->setInputTarget(target_cloud);
        gicp->setInputSource(source_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        auto t1 = std::chrono::system_clock::now();
        gicp->align(*aligned_cloud, initial_transform);
        auto t2 = std::chrono::system_clock::now();

        // 打印配准信息
        double fit_score = gicp->getFitnessScore();
        long long calib_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        RCLCPP_WARN(this->get_logger(), "配准耗时：%lldms | 配准分数：%.4f", calib_time, fit_score);

        // 配准成功判断
        if (fit_score < 0.15) {
            has_aligned_ = true;
            RCLCPP_INFO(this->get_logger(), "配准成功！TF变换已固定");
        //     // 打印最终变换矩阵
        //     RCLCPP_INFO(this->get_logger(), "最终变换矩阵：");
        //     RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
        //                 transform(0,0), transform(0,1), transform(0,2), transform(0,3));
        //     RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
        //                 transform(1,0), transform(1,1), transform(1,2), transform(1,3));
        //     RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
        //                 transform(2,0), transform(2,1), transform(2,2), transform(2,3));
        }

        return gicp->getFinalTransformation();
    }

    void publishTF(const Eigen::Matrix4f& transform) {
        geometry_msgs::msg::TransformStamped ts;
        ts.header.stamp = this->now();
        ts.header.frame_id = "rm_frame";
        ts.child_frame_id = "livox_frame";
        // 平移
        ts.transform.translation.x = transform(0, 3);
        ts.transform.translation.y = transform(1, 3);
        ts.transform.translation.z = transform(2, 3);
        // 旋转（四元数）
        Eigen::Quaternionf q(transform.block<3,3>(0,0));
        ts.transform.rotation.x = q.x();
        ts.transform.rotation.y = q.y();
        ts.transform.rotation.z = q.z();
        ts.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(ts);
    }

    // -------------------------- 成员变量（新增参数保存/加载相关） --------------------------
    // 手动配准核心参数
    bool manual_mode_ = true;
    double manual_trans_x_ = 0.0;
    double manual_trans_y_ = 0.0;
    double manual_trans_z_ = 0.0;
    double manual_roll_ = 0.0;
    double manual_pitch_ = 0.0;
    double manual_yaw_ = 0.0;
    bool trigger_auto_calib_ = false;
    bool need_reset_trigger_ = false;

    // 新增：参数保存/加载相关
    std::string manual_calib_config_path_;  // 参数配置文件路径
    bool param_save_trigger_ = false;       // 保存参数触发标志
    bool need_reset_save_trigger_ = false;  // 重置保存触发标志
    // 新增：自动触发配准相关
    bool auto_trigger_calib_after_load_ = true;  // 加载参数后自动触发配准的标志
    bool has_auto_triggered_ = false;            // 避免重复触发的标志


    // 原有变量
    bool has_aligned_ = false;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f initial_transform_ = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    int accumulate_time = 20;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr manual_aligned_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    double gridSizeDegrees = 0.1;
    std::map<std::pair<int, int>, Grid> gridMap;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rclcpp::TimerBase::SharedPtr param_reset_timer_;
};
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)
