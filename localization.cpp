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
        // 日志格式符 %d → %lu（匹配 std::size_t 类型）
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

        // 5. 声明手动配准参数（移除 ParameterDescriptor，兼容旧ROS2版本）
        declare_manual_params();
        // 6. 设置参数回调（实时响应参数修改）
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&Localization::param_callback, this, std::placeholders::_1));

        // -------------------------- 新增：参数重置定时器 --------------------------
        // 用于在参数回调外重置 trigger_auto_calib 参数（避免线程冲突）
        param_reset_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), // 100ms周期检查
            std::bind(&Localization::param_reset_callback, this));

        RCLCPP_INFO(this->get_logger(), "手动配准模式已启动！");
        RCLCPP_INFO(this->get_logger(), "可通过参数调整位姿，预览话题：/livox/manual_aligned_cloud");
        RCLCPP_INFO(this->get_logger(), "可用参数列表：");
        RCLCPP_INFO(this->get_logger(), "  manual_trans_x/y/z: 平移（米，范围±50）");
        RCLCPP_INFO(this->get_logger(), "  manual_roll/pitch/yaw_deg: 旋转（度）");
        RCLCPP_INFO(this->get_logger(), "  manual_mode: 手动模式开关（true/false）");
        RCLCPP_INFO(this->get_logger(), "  trigger_auto_calib: 触发自动精配准（true）");
    }

private:
    // -------------------------- 新增：参数重置定时器回调 --------------------------
    // 独立线程中重置 trigger_auto_calib 参数，避免参数回调内修改
    void param_reset_callback() {
        if (need_reset_trigger_) {
            this->set_parameter(rclcpp::Parameter("trigger_auto_calib", false)); // 安全重置
            need_reset_trigger_ = false; // 重置标志位
        }
    }

    // -------------------------- 手动配准核心功能 --------------------------
    // 声明手动配准参数（移除 ParameterDescriptor，兼容旧版本）
    void declare_manual_params() {
        // 平移参数（单位：米，范围±50米，已扩大）
        this->declare_parameter<double>("manual_trans_x", 0.0);
        this->declare_parameter<double>("manual_trans_y", 0.0);
        this->declare_parameter<double>("manual_trans_z", 0.0);
        // 旋转参数（单位：度，用户易操作，内部转弧度）
        this->declare_parameter<double>("manual_roll_deg", 0.0);
        this->declare_parameter<double>("manual_pitch_deg", 0.0);
        this->declare_parameter<double>("manual_yaw_deg", 0.0);
        // 模式控制参数
        this->declare_parameter<bool>("manual_mode", true);
        this->declare_parameter<bool>("trigger_auto_calib", false);
    }

    // 参数回调：实时更新手动配准参数（不再直接修改参数）
    rcl_interfaces::msg::SetParametersResult param_callback(
        const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto& param : parameters) {
            // 平移参数更新（范围±50米）
            if (param.get_name() == "manual_trans_x") update_param_with_range(param, manual_trans_x_, -100.0, 100.0);
            else if (param.get_name() == "manual_trans_y") update_param_with_range(param, manual_trans_y_, -100.0, 100.0);
            else if (param.get_name() == "manual_trans_z") update_param_with_range(param, manual_trans_z_, -100.0, 100.0);
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
                if (manual_mode_) {
                    RCLCPP_WARN(this->get_logger(), "切换到【手动配准模式】");
                    has_aligned_ = false; // 重置自动配准状态
                } else {
                    RCLCPP_WARN(this->get_logger(), "切换到【自动精配准模式】");
                }
            }
            // 触发自动精配准（关键修改：仅设置标志位，不直接修改参数）
            else if (param.get_name() == "trigger_auto_calib" && param.as_bool()) {
                trigger_auto_calib_ = true;
                need_reset_trigger_ = true; // 标记需要重置参数
            }
        }
        return result;
    }

    // 辅助函数：更新参数并检查范围
    template<typename T>
    void update_param_with_range(const rclcpp::Parameter& param, T& var, T min, T max) {
        T val = param.as_double();
        if (val < min || val > max) {
            RCLCPP_WARN(this->get_logger(), "%s超出范围[%f, %f]，忽略修改", param.get_name().c_str(), min, max);
            return;
        }
        var = val;
    }

    // 生成手动调整的变换矩阵（Z-Y-X旋转顺序，符合ROS坐标系）
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

    // -------------------------- 原有核心功能 --------------------------
    // 下采样辅助函数（复用代码）
    void downsample_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                          float leaf_size) {
        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxelgrid.setInputCloud(input);
        voxelgrid.filter(*output);
    }

    // 回调函数：整合手动/自动配准逻辑
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 转换原始雷达点云到PCL格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *source_cloud);

        // -------------------------- 手动配准模式 --------------------------
        if (manual_mode_) {
            // 1. 生成手动变换矩阵
            Eigen::Matrix4f manual_transform = create_manual_transform();

            // 2. 应用变换到雷达点云（转换到地图坐标系，方便RViz叠加预览）
            pcl::PointCloud<pcl::PointXYZ>::Ptr manual_aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*source_cloud, *manual_aligned_cloud, manual_transform);

            // 3. 发布手动调整后的点云（RViz中查看叠加效果）
            sensor_msgs::msg::PointCloud2 manual_msg;
            pcl::toROSMsg(*manual_aligned_cloud, manual_msg);
            manual_msg.header.frame_id = "rm_frame"; // 地图坐标系
            manual_msg.header.stamp = msg->header.stamp;
            manual_aligned_publisher_->publish(manual_msg);

            // 4. 发布手动TF（实时预览位姿）
            publishTF(manual_transform);

            // 5. 打印当前参数（1秒打印一次，避免刷屏）
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "手动配准参数：trans(x:%.2f, y:%.2f, z:%.2f) | "
                                "roll:%.1f° pitch:%.1f° yaw:%.1f°",
                                manual_trans_x_, manual_trans_y_, manual_trans_z_,
                                manual_roll_ * 180/M_PI, manual_pitch_*180/M_PI, manual_yaw_*180/M_PI);

            // 6. 若触发自动精配准，切换模式并初始化GICP初始位姿
            if (trigger_auto_calib_) {
                manual_mode_ = false;
                initial_transform_ = manual_transform; // 手动位姿作为GICP初始猜测
                RCLCPP_WARN(this->get_logger(), "触发自动精配准！初始位姿已设置为手动调整值");
                gridMap.clear(); // 重置栅格地图
                accumulated_clouds_.clear(); // 重置累积点云
                trigger_auto_calib_ = false; // 重置触发标志
            }
            return;
        }

        // -------------------------- 自动精配准模式 --------------------------
        if (!has_aligned_) {
            // 1. 累积20帧点云（提高稳定性）
            if (accumulated_clouds_.size() < accumulate_time) {
                accumulated_clouds_.push_back(source_cloud);
                return;
            } else {
                accumulated_clouds_.erase(accumulated_clouds_.begin());
                accumulated_clouds_.push_back(source_cloud);
            }

            // 2. 栅格地图构建（保留最远点）
            for (auto& cloud : accumulated_clouds_) {
                for (const auto& point : *cloud) {
                    if (point.x <= 0) continue; // 只处理前方点
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

            // 4. ROI过滤（根据实际场景调整范围）
            pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& point : grid_cloud->points) {
                if (point.x > 5 && point.x < 30 && point.y > -10 && point.y < 8 && point.z < 7) {
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
            downsample_cloud(roi_cloud, downsampled_source, 0.1f);

            // 6. 发布过滤后的点云（调试用）
            sensor_msgs::msg::PointCloud2 filter_msg;
            pcl::toROSMsg(*downsampled_source, filter_msg);
            filter_msg.header.frame_id = "livox_frame";
            filter_publisher_->publish(filter_msg);

            // 7. GICP精配准（兼容所有PCL版本）
            RCLCPP_INFO(this->get_logger(), "开始GICP精配准（初始位姿来自手动调整）");
            auto gicp = pcl::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
            gicp->setMaximumIterations(500);       // 增加迭代次数
            gicp->setTransformationEpsilon(1e-8);  // 收敛阈值
            gicp->setEuclideanFitnessEpsilon(1e-6); // 配准精度阈值
            transform = align(gicp, target_cloud_, downsampled_source, initial_transform_);
        }

        // 发布最终TF变换（自动配准成功后固定）
        publishTF(transform);
    }

    // -------------------------- align函数（兼容所有PCL版本） --------------------------
    Eigen::Matrix4f
    align(pcl::shared_ptr<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> gicp,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
          const Eigen::Matrix4f& initial_transform) {
        gicp->setInputTarget(target_cloud);
        gicp->setInputSource(source_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        auto t1 = std::chrono::system_clock::now();
        gicp->align(*aligned_cloud, initial_transform); // 直接传入初始变换
        auto t2 = std::chrono::system_clock::now();

        // 打印配准信息
        double fit_score = gicp->getFitnessScore();
        long long calib_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        RCLCPP_WARN(this->get_logger(), "配准耗时：%lldms | 配准分数：%.4f", calib_time, fit_score);

        // 配准成功判断（分数越低越好，可根据场景调整阈值）
        if (fit_score < 0.15) {
            has_aligned_ = true;
            RCLCPP_INFO(this->get_logger(), "配准成功！TF变换已固定");
            // 打印最终变换矩阵（可选保存到文件）
            RCLCPP_INFO(this->get_logger(), "最终变换矩阵：");
            RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
                        transform(0,0), transform(0,1), transform(0,2), transform(0,3));
            RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
                        transform(1,0), transform(1,1), transform(1,2), transform(1,3));
            RCLCPP_INFO(this->get_logger(), "%.3f %.3f %.3f %.3f",
                        transform(2,0), transform(2,1), transform(2,2), transform(2,3));
        }

        return gicp->getFinalTransformation();
    }

    // TF发布函数（无修改）
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

    // -------------------------- 成员变量 --------------------------
    // 手动配准参数
    bool manual_mode_ = true;                // 手动模式开关
    double manual_trans_x_ = 0.0;            // X平移（米）
    double manual_trans_y_ = 0.0;            // Y平移（米）
    double manual_trans_z_ = 0.0;            // Z平移（米）
    double manual_roll_ = 0.0;               // 滚转（弧度）
    double manual_pitch_ = 0.0;              // 俯仰（弧度）
    double manual_yaw_ = 0.0;                // 偏航（弧度）
    bool trigger_auto_calib_ = false;        // 自动精配准触发标志
    bool need_reset_trigger_ = false;        // 新增：需要重置trigger的标志位
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rclcpp::TimerBase::SharedPtr param_reset_timer_; // 新增：参数重置定时器

    // 原有变量 + 初始变换矩阵
    bool has_aligned_ = false;               // 自动配准完成标志
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(); // 最终位姿变换矩阵
    Eigen::Matrix4f initial_transform_ = Eigen::Matrix4f::Identity(); // GICP初始位姿（来自手动调整）
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;       // 目标地图点云
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_; // 累积点云
    int accumulate_time = 20;                // 累积帧数
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr manual_aligned_publisher_; // 手动配准预览发布器
    rclcpp::TimerBase::SharedPtr timer_;     // 地图发布计时器
    double gridSizeDegrees = 0.1;            // 栅格大小（0.1°×0.1°）
    std::map<std::pair<int, int>, Grid> gridMap; // 栅格地图
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Localization)
