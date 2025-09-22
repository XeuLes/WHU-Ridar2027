// 包含自定义推理工具头文件（声明仿射变换+归一化函数）
#include "InferTool.hpp"
// 包含Nvidia CUDA接口头文件（封装CUDA错误检查、内存管理等）
#include <NvidiaInterface.hpp>

// 雷达视觉相关命名空间
namespace tdt_radar {

    /**
     * @brief CUDA核函数：在GPU上完成图像双线性插值仿射变换 + 通道归一化（核心预处理操作）
     * 作用：将原始图像（uint8，BGR格式）缩放到模型输入尺寸，同时完成归一化（如MeanStd/AlphaBeta），
     *       输出为float格式的CHW平面（符合TensorRT模型输入要求）
     * 注：__global__修饰表示该函数是GPU核函数，由主机端调用，在设备端（GPU）并行执行
     * 
     * @param src          输入图像GPU指针（uint8_t，BGR格式，内存布局：每行按src_line_size字节对齐）
     * @param src_line_size 输入图像每行的实际字节数（含内存对齐，可能大于src_width*3）
     * @param src_width    输入图像宽度（像素数）
     * @param src_height   输入图像高度（像素数）
     * @param dst          输出图像GPU指针（float，CHW格式：3个通道连续存储，先R、再G、最后B）
     * @param dst_width    输出图像宽度（模型输入宽度，如224）
     * @param dst_height   输出图像高度（模型输入高度，如224）
     * @param const_value_st 边界填充值（输入坐标超出原图范围时用该值填充，如114）
     * @param warp_affine_matrix_2_3 仿射矩阵GPU指针（2x3矩阵，float类型，实现输出→输入坐标映射）
     * @param norm         归一化配置（含归一化类型、均值/标准差、通道顺序等）
     */
    __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
        uint8_t* src, int src_line_size, int src_width, int src_height,
        float* dst, int dst_width, int dst_height, uint8_t const_value_st,
        float* warp_affine_matrix_2_3, Norm norm)
    {
        // 1. 计算当前线程对应的输出图像坐标（dx：列，dy：行）
        // blockDim：线程块内线程数（如32x32）；blockIdx：线程块索引；threadIdx：线程块内线程索引
        int dx = blockDim.x * blockIdx.x + threadIdx.x;  // 输出图像的x坐标（横向）
        int dy = blockDim.y * blockIdx.y + threadIdx.y;  // 输出图像的y坐标（纵向）

        // 若线程坐标超出输出图像范围，直接返回（避免越界访问）
        if (dx >= dst_width || dy >= dst_height)
            return;

        // 2. 提取2x3仿射矩阵的元素（矩阵作用：将输出坐标(dx,dy)映射到输入图像坐标(src_x,src_y)）
        // 仿射矩阵格式：[m_x1, m_y1, m_z1; m_x2, m_y2, m_z2]
        float m_x1 = warp_affine_matrix_2_3[0];  // 第一行第一列（x方向缩放/剪切系数）
        float m_y1 = warp_affine_matrix_2_3[1];  // 第一行第二列（y方向剪切/缩放系数）
        float m_z1 = warp_affine_matrix_2_3[2];  // 第一行第三列（x方向偏移量）
        float m_x2 = warp_affine_matrix_2_3[3];  // 第二行第一列（y方向剪切/缩放系数）
        float m_y2 = warp_affine_matrix_2_3[4];  // 第二行第二列（y方向缩放/剪切系数）
        float m_z2 = warp_affine_matrix_2_3[5];  // 第二行第三列（y方向偏移量）

        // 3. 计算输出坐标(dx,dy)在输入图像中对应的原始坐标（浮点数，可能落在像素之间）
        float src_x = m_x1 * dx + m_y1 * dy + m_z1;  // 输入图像的x坐标（横向）
        float src_y = m_x2 * dx + m_y2 * dy + m_z2;  // 输入图像的y坐标（纵向）

        // 存储当前像素的三个通道值（c0:B, c1:G, c2:R，初始化为填充值）
        float c0, c1, c2;

        // 4. 边界条件处理：若原始坐标超出输入图像范围，用const_value_st填充
        if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
            c0 = const_value_st;  // B通道填充
            c1 = const_value_st;  // G通道填充
            c2 = const_value_st;  // R通道填充
        } else {
            // 5. 双线性插值：计算原始坐标周围4个邻域像素的索引（下取整得到整数坐标）
            int y_low = floorf(src_y);    // 原始y坐标的下边界像素行
            int x_low = floorf(src_x);    // 原始x坐标的左边界像素列
            int y_high = y_low + 1;       // 原始y坐标的上边界像素行
            int x_high = x_low + 1;       // 原始x坐标的右边界像素列

            // 边界填充值数组（邻域像素超出原图时使用）
            uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
            // 计算插值权重（基于原始坐标与邻域像素的距离）
            float ly = src_y - y_low;  // y方向小数部分（距离y_low的比例，0~1）
            float lx = src_x - x_low;  // x方向小数部分（距离x_low的比例，0~1）
            float hy = 1 - ly;         // y方向互补权重（距离y_high的比例）
            float hx = 1 - lx;         // x方向互补权重（距离x_high的比例）
            // 四个邻域像素的插值权重（权重和为1）
            float w1 = hy * hx;  // (y_low, x_low)的权重
            float w2 = hy * lx;  // (y_low, x_high)的权重
            float w3 = ly * hx;  // (y_high, x_low)的权重
            float w4 = ly * lx;  // (y_high, x_high)的权重

            // 初始化四个邻域像素的指针（默认指向填充值）
            uint8_t* v1 = const_value;  // (y_low, x_low)的像素指针
            uint8_t* v2 = const_value;  // (y_low, x_high)的像素指针
            uint8_t* v3 = const_value;  // (y_high, x_low)的像素指针
            uint8_t* v4 = const_value;  // (y_high, x_high)的像素指针

            // 若y_low在输入图像范围内，更新v1和v2的指针（指向实际像素）
            if (y_low >= 0) {
                // x_low在范围内→v1指向输入图像的(y_low, x_low)像素（BGR格式，每个像素3字节）
                if (x_low >= 0)
                    v1 = src + y_low * src_line_size + x_low * 3;
                // x_high在范围内→v2指向输入图像的(y_low, x_high)像素
                if (x_high < src_width)
                    v2 = src + y_low * src_line_size + x_high * 3;
            }

            // 若y_high在输入图像范围内，更新v3和v4的指针（指向实际像素）
            if (y_high < src_height) {
                // x_low在范围内→v3指向输入图像的(y_high, x_low)像素
                if (x_low >= 0)
                    v3 = src + y_high * src_line_size + x_low * 3;
                // x_high在范围内→v4指向输入图像的(y_high, x_high)像素
                if (x_high < src_width)
                    v4 = src + y_high * src_line_size + x_high * 3;
            }

            // 6. 计算双线性插值结果（三个通道分别计算，加0.5后下取整模拟四舍五入，与OpenCV逻辑一致）
            c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);  // B通道插值结果
            c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);  // G通道插值结果
            c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);  // R通道插值结果
        }

        // 7. 通道顺序调整：若需要交换RB通道（BGR→RGB），适配模型输入格式
        if (norm.channel_type == ChannelType::SwapRB) {
            float t = c2;  // 临时存储R通道值
            c2 = c0;       // R通道 = 原B通道
            c0 = t;        // B通道 = 原R通道（此时c0=R, c1=G, c2=B）
        }

        // 8. 图像归一化：根据配置将uint8值转为模型所需的float格式
        if (norm.type == NormType::MeanStd) {
            // 均值标准差归一化：(像素值 * alpha - 均值) / 标准差
            // alpha通常为1/255，将uint8[0,255]转为float[0,1]
            c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];  // 第一通道（如R）
            c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];  // 第二通道（如G）
            c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];  // 第三通道（如B）
        } else if (norm.type == NormType::AlphaBeta) {
            // 线性缩放归一化：像素值 * alpha + beta（如alpha=1/255，beta=0，仅缩放）
            c0 = c0 * norm.alpha + norm.beta;
            c1 = c1 * norm.alpha + norm.beta;
            c2 = c2 * norm.alpha + norm.beta;
        }

        // 9. 结果写入输出缓冲区（输出格式为CHW：三个通道连续存储，每个通道占dst_width*dst_height个元素）
        int area = dst_width * dst_height;  // 单个通道的像素总数
        float* pdst_c0 = dst + dy * dst_width + dx;  // 第一通道（如R）的当前像素地址
        float* pdst_c1 = pdst_c0 + area;             // 第二通道（如G）的当前像素地址（跳过第一通道）
        float* pdst_c2 = pdst_c1 + area;             // 第三通道（如B）的当前像素地址（跳过前两通道）
        *pdst_c0 = c0;  // 写入第一通道结果
        *pdst_c1 = c1;  // 写入第二通道结果
        *pdst_c2 = c2;  // 写入第三通道结果
    }

    /**
     * @brief 宿主端函数：配置CUDA线程布局并启动核函数（仿射变换+归一化）
     * 作用：作为主机端与GPU核函数的桥梁，负责初始化线程网格/块，传递参数，启动核函数并检查错误
     * 
     * @param src          输入图像GPU指针（uint8_t，BGR格式）
     * @param src_line_size 输入图像每行字节数（含对齐）
     * @param src_width    输入图像宽度
     * @param src_height   输入图像高度
     * @param dst          输出图像GPU指针（float，CHW格式）
     * @param dst_width    输出图像宽度（模型输入宽）
     * @param dst_height   输出图像高度（模型输入高）
     * @param matrix_2_3   仿射矩阵GPU指针（2x3，输出→输入映射）
     * @param const_value  边界填充值（如114）
     * @param norm         归一化配置
     * @param stream       CUDA流（用于异步执行，提升GPU利用率）
     */
    void warp_affine_bilinear_and_normalize_plane(
        uint8_t* src, int src_line_size, int src_width, int src_height,
        float* dst, int dst_width, int dst_height, float* matrix_2_3,
        uint8_t const_value, const Norm& norm, cudaStream_t stream)
    {
        // 1. 配置CUDA线程布局（2D网格+2D线程块，适配图像的2D结构）
        // 线程块大小：32x32（每个块1024个线程，符合CUDA warp size（32）的整数倍，效率最高）
        dim3 block(32, 32);
        // 线程网格大小：按输出图像尺寸向上取整，确保每个输出像素对应一个线程
        // (dst_width + 31)/32：横向块数（31是block.x-1，避免整除时多算一块）
        // (dst_height + 31)/32：纵向块数
        dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);

        // 2. 启动CUDA核函数并检查启动错误
        // <<<grid, block, 0, stream>>>：核函数启动参数（网格、块、共享内存大小、流）
        // checkKernel：自定义宏，检查核函数启动是否成功（如参数错误、资源不足）
        checkKernel(
            warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size, src_width, src_height,
                dst, dst_width, dst_height, const_value,
                matrix_2_3, norm));
    }

}  // namespace tdt_radar
