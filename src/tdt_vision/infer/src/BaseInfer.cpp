//
// Created by mozihe on 24-11-21.
//
// 包含基础推理工具类头文件（定义了Norm和AffineMatrix结构体）
#include "BaseInfer.hpp"
// 用于内存拷贝（memcpy函数）
#include <cstring>

// 雷达视觉推理相关功能的命名空间
using namespace tdt_radar;

/**
 * @brief Norm结构体：用于配置图像预处理的归一化参数
 * 支持三种归一化模式：均值标准差归一化（MeanStd）、线性缩放归一化（AlphaBeta）、无归一化（None）
 * 核心作用：将图像像素值转换为模型输入要求的数值范围（如[-1,1]或[0,1]）
 */

/**
 * @brief 静态方法：创建“均值标准差归一化”配置
 * 公式：output = (input - mean) / std * alpha
 * @param mean[3] 3通道图像的均值数组（如RGB/BGR各通道的均值）
 * @param std[3] 3通道图像的标准差数组（对应通道的标准差，需非零）
 * @param alpha 额外缩放因子（可选，用于调整输出范围）
 * @param channel_type
 * 通道顺序类型（如ChannelType::RGB或ChannelType::BGR，匹配模型输入要求）
 * @return 配置好的Norm实例
 */
Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type)
{
    Norm out;                      // 初始化归一化配置实例
    out.type = NormType::MeanStd;  // 设置归一化类型为“均值标准差”
    out.alpha = alpha;             // 赋值缩放因子
    out.channel_type = channel_type;  // 赋值通道顺序
    // 内存拷贝：将输入的mean数组复制到out.mean（避免浅拷贝，确保数据独立）
    memcpy(out.mean, mean, sizeof(out.mean));
    // 内存拷贝：将输入的std数组复制到out.std
    memcpy(out.std, std, sizeof(out.std));
    return out;  // 返回配置好的实例
}

/**
 * @brief 静态方法：创建“线性缩放归一化”配置
 * 公式：output = input * alpha + beta
 * @param alpha 缩放系数（如1/255，将像素从[0,255]缩放到[0,1]）
 * @param beta 偏移量（如-0.5，将缩放后的值偏移到[-0.5,0.5]）
 * @param channel_type 通道顺序类型（匹配模型输入要求）
 * @return 配置好的Norm实例
 */
Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
{
    Norm out;                        // 初始化归一化配置实例
    out.type = NormType::AlphaBeta;  // 设置归一化类型为“线性缩放”
    out.alpha = alpha;               // 赋值缩放系数
    out.beta = beta;                 // 赋值偏移量
    out.channel_type = channel_type;  // 赋值通道顺序
    return out;                       // 返回配置好的实例
}

/**
 * @brief 静态方法：创建“无归一化”配置
 * 作用：当图像已提前处理好（无需额外归一化）时使用，所有参数默认初始化
 * @return 空配置的Norm实例（type默认为NormType::None）
 */
Norm Norm::None()
{
    return Norm();  // 调用默认构造函数，返回空配置
}

/**
 * @brief AffineMatrix结构体：用于计算图像尺寸变换的仿射矩阵
 * 核心作用：
 * 1. 计算“原始图像坐标 →
 * 目标图像坐标”的正向矩阵（i2d）：用于图像resize+居中裁剪
 * 2. 计算“目标图像坐标 →
 * 原始图像坐标”的反向矩阵（d2i）：用于将模型输出的检测框映射回原始图像
 * 仿射矩阵格式（6元素数组对应2x3矩阵）：
 * [ i2d[0]  i2d[1]  i2d[2] ]
 * [ i2d[3]  i2d[4]  i2d[5] ]
 */

/**
 * @brief 计算正向（i2d）和反向（d2i）仿射矩阵
 * @param from 原始图像尺寸（tuple<int, int>：width, height）
 * @param to 目标图像尺寸（tuple<int, int>：width, height）
 * 逻辑：保持原始图像宽高比，将图像缩放到目标尺寸内并居中，避免拉伸变形
 */
void AffineMatrix::compute(const std::tuple<int, int>& from,
                           const std::tuple<int, int>& to)
{
    // 1. 计算原始图像到目标图像的缩放比例
    // 从tuple中提取原始宽度（from_w）和目标宽度（to_w），计算x方向缩放比
    float from_w = std::get<0>(from);
    float to_w = std::get<0>(to);
    float scale_x = to_w / from_w;

    // 从tuple中提取原始高度（from_h）和目标高度（to_h），计算y方向缩放比
    float from_h = std::get<1>(from);
    float to_h = std::get<1>(to);
    float scale_y = to_h / from_h;

    // 取x/y缩放比的最小值（保持宽高比，避免图像拉伸）
    float scale = std::min(scale_x, scale_y);

    // 2. 计算正向仿射矩阵（i2d：image → destination，原始→目标）
    i2d[0] = scale;  // x方向缩放系数（a）
    i2d[1] = 0;      // x方向剪切系数（b，0表示无剪切）
    // x方向偏移量（c）：确保缩放后的图像在目标尺寸中水平居中
    // 计算逻辑：目标宽度的一半 - 原始宽度缩放后的一半 +
    // 0.5*scale（像素对齐补偿）
    i2d[2] = -scale * from_w * 0.5 + to_w * 0.5 + scale * 0.5 - 0.5;

    i2d[3] = 0;      // y方向剪切系数（d，0表示无剪切）
    i2d[4] = scale;  // y方向缩放系数（e）
    // y方向偏移量（f）：确保缩放后的图像在目标尺寸中垂直居中
    i2d[5] = -scale * from_h * 0.5 + to_h * 0.5 + scale * 0.5 - 0.5;

    // 3. 计算反向仿射矩阵（d2i：destination → image，目标→原始）
    // 2x3仿射矩阵的逆矩阵计算：先求左上角2x2矩阵的逆，再计算偏移量
    // 左上角2x2矩阵的行列式 D = a*e - b*d（此处b=d=0，故D=scale*scale）
    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    // 处理行列式为0的异常情况（避免除以零，此时逆矩阵无效）
    D = D != 0. ? double(1.) / D : double(0.);

    // 计算2x2逆矩阵的元素（A11, A12, A21, A22）
    double A11 = i2d[4] * D;   // e/D
    double A22 = i2d[0] * D;   // a/D
    double A12 = -i2d[1] * D;  // -b/D（此处b=0，故A12=0）
    double A21 = -i2d[3] * D;  // -d/D（此处d=0，故A21=0）

    // 计算逆矩阵的偏移量（b1, b2）
    double b1 = -A11 * i2d[2] - A12 * i2d[5];  // 反向x偏移
    double b2 = -A21 * i2d[2] - A22 * i2d[5];  // 反向y偏移

    // 赋值反向仿射矩阵（d2i）
    d2i[0] = A11;  // 逆矩阵a
    d2i[1] = A12;  // 逆矩阵b
    d2i[2] = b1;   // 逆矩阵c
    d2i[3] = A21;  // 逆矩阵d
    d2i[4] = A22;  // 逆矩阵e
    d2i[5] = b2;   // 逆矩阵f
}
