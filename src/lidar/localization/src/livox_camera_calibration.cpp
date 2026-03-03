#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

/**
 * @brief 根据相机全局位姿和量测距离，计算雷达在场地坐标系下的 6DOF 位姿
 */
Eigen::VectorXd calculateLivoxPoseInWorldFrame(const Eigen::Matrix4d& T_world_camera, double dx, double dy, double dz) {
    // 1. 构建相机到雷达的外参矩阵 T_camera_livox (4x4)
    Eigen::Matrix4d T_camera_livox = Eigen::Matrix4d::Identity();

    T_camera_livox(0, 3) = dx;
    T_camera_livox(1, 3) = dy;
    T_camera_livox(2, 3) = dz;

    // 基础旋转矩阵 (假设雷达和相机物理正向朝向一致)
    T_camera_livox.block<3, 3>(0, 0) <<  0, -1,  0,
                                         0,  0, -1,
                                         1,  0,  0;

    // 2. 矩阵相乘：计算雷达在 world_frame 下的变换矩阵
    Eigen::Matrix4d T_world_livox = T_world_camera * T_camera_livox;

    // 3. 提取 6 个位姿参数
    Eigen::VectorXd pose_6d(6);
    
    pose_6d(0) = T_world_livox(0, 3); // x
    pose_6d(1) = T_world_livox(1, 3); // y
    pose_6d(2) = T_world_livox(2, 3); // z

    Eigen::Matrix3d rotation_matrix = T_world_livox.block<3, 3>(0, 0);
    // 注意: Eigen 的 eulerAngles 返回的范围可能是 [0, pi] 或 [-pi, pi]
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // Z-Y-X 顺序 (Yaw, Pitch, Roll)

    pose_6d(3) = euler_angles(2); // Roll
    pose_6d(4) = euler_angles(1); // Pitch
    pose_6d(5) = euler_angles(0); // Yaw

    return pose_6d;
}

int main() {
    std::string input_yaml = "world_to_camera.yaml";
    std::string output_yaml = "livox_rm_manual.yaml";

    // ================= 1. 读取 OpenCV PnP 结果 =================
    cv::FileStorage fs(input_yaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法打开输入文件 " << input_yaml << std::endl;
        return -1;
    }

    cv::Mat rvec_cv, tvec_cv;
    fs["world_rvec"] >> rvec_cv;
    fs["world_tvec"] >> tvec_cv;
    fs.release();

    // 转换为 Eigen 矩阵并求逆
    cv::Mat R_cv;
    cv::Rodrigues(rvec_cv, R_cv);

    Eigen::Matrix3d R_world_to_camera;
    Eigen::Vector3d t_world_to_camera;
    cv::cv2eigen(R_cv, R_world_to_camera);
    cv::cv2eigen(tvec_cv, t_world_to_camera);

    Eigen::Matrix4d T_camera_world = Eigen::Matrix4d::Identity();
    T_camera_world.block<3, 3>(0, 0) = R_world_to_camera;
    T_camera_world.block<3, 1>(0, 3) = t_world_to_camera;

    Eigen::Matrix4d T_world_camera = T_camera_world.inverse();

    // ================= 2. 填入你的量测数据 =================
    // 请在此处修改为你实际量测的物理距离 (单位: 米)
    double dx =  0.05; // 右侧
    double dy = -0.10; // 下方
    double dz = -0.08; // 前方

    // ================= 3. 执行计算 =================
    Eigen::VectorXd final_pose = calculateLivoxPoseInWorldFrame(T_world_camera, dx, dy, dz);

    // ================= 4. 格式化写入 ROS2 参数 YAML 文件 =================
    std::ofstream outfile(output_yaml);
    if (!outfile.is_open()) {
        std::cerr << "错误: 无法创建输出文件 " << output_yaml << std::endl;
        return -1;
    }

    // 设置高精度，以匹配你示例文件中的浮点数精度
    outfile << std::setprecision(16);

    // 按照 ROS2 parameters 格式写入
    outfile << "/**:\n";
    outfile << "  ros__parameters: {";
    outfile << "initial_x: " << final_pose(0) << ", ";
    outfile << "initial_y: " << final_pose(1) << ", ";
    outfile << "initial_z: " << final_pose(2) << ", ";
    outfile << "initial_roll: " << final_pose(3) << ", ";
    outfile << "initial_pitch: " << final_pose(4) << ", ";
    outfile << "initial_yaw: " << final_pose(5);
    outfile << "}\n";

    outfile.close();

    std::cout << "成功! 位姿参数已保存至: " << output_yaml << std::endl;
    
    // 顺便在终端打印一下方便查看
    std::cout << "x: " << final_pose(0) << ", y: " << final_pose(1) << ", z: " << final_pose(2) << std::endl;
    std::cout << "roll: " << final_pose(3) << ", pitch: " << final_pose(4) << ", yaw: " << final_pose(5) << std::endl;

    return 0;
}