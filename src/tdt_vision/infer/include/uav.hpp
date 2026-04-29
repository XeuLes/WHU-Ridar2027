#ifndef UAV_HPP
#define UAV_HPP

#include "yolos.hpp" // 引用基础的 BoxArray 和 Image 定义
#include <memory>
#include <string>
#include <vector>

namespace uav {

/**
 * @brief 加载无人机第一层检测模型（ROI 提取模型）
 * * @param engine_file TensorRT 引擎文件路径 (.engine)
 * @param confidence_threshold 置信度阈值 (例如 0.5)
 * @param nms_threshold 非极大值抑制阈值 (例如 0.45)
 * @return std::shared_ptr<yolo::Infer<yolo::BoxArray>> 返回推理接口指针
 */
std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> load(
    const std::string& engine_file, 
    float confidence_threshold = 0.5f, 
    float nms_threshold = 0.2f
);

} // namespace uav

#endif // UAV_HPP
