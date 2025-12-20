#define _HAS_STD_BYTE 0  // 关键：解决 std::byte 冲突
#include <cstdint>       // 关键：解决 uint8_t 未定义
#include "uav.hpp"
#include "BaseInfer.hpp"
#include "NvidiaInterface.hpp"
#include "InferTool.hpp"
#include "yolos.hpp"
#include <vector>
#include <algorithm>     // 确保 std::min 可用
#include <tuple>

namespace uav {

using namespace std;
using namespace tdt_radar;


#define DBG_LBL "\033[1;33m[DEBUG]\033[0m "  // 黄色 [DEBUG]
#define DBG_SUC "\033[1;32m[SUCCESS]\033[0m " // 绿色 [SUCCESS]
#define DBG_ERR "\033[1;31m[ERROR]\033[0m "   // 红色 [ERROR]

// --- 基础常量定义，严格对应后处理内存结构 ---
// [left, top, right, bottom, confidence, label, keep_flag]
const int NUM_BOX_ELEMENT = 7;
const int MAX_IMAGE_BOXES = 1024;

// 仿射变换矩阵结构体（从 yolo.cu 复制）
struct AffineMatrix {
    float i2d[6];  // 正向矩阵（原图→模型输入尺寸）
    float d2i[6];  // 反向矩阵（模型输入尺寸→原图）

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to)
    {
        // 计算缩放比（保持宽高比）
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);

        // 计算正向矩阵（缩放+居中偏移）
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        // 计算反向矩阵（2x3矩阵求逆）
        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];  // 行列式
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        // 赋值反向矩阵
        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

// 内存步长对齐工具（关键：防止 CPU/GPU 读取位置偏移）
inline int upbound(int n, int align = 32) {
    return (n + align - 1) / align * align;
}

/**
 * @brief 坐标变换：将模型输出的 [0, 640] 坐标映射回原图
 */
static __host__ __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}


/**
 * @brief 针对 [5, 33600] 布局修复的解码核函数
 */
static __global__ void decode_kernel_uav(
    float* predict, int num_bboxes, int output_cdim, 
    float confidence_threshold, float* invert_affine_matrix, 
    float* parray, int max_boxes)
{
    // num_bboxes = 33600
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    // --- 核心修复：改“行读取”为“跨平面读取” ---
    // 每一个属性的偏移量都是 num_bboxes (33600)
    float cx         = predict[0 * num_bboxes + position];
    float cy         = predict[1 * num_bboxes + position];
    float w          = predict[2 * num_bboxes + position];
    float h          = predict[3 * num_bboxes + position];
    float confidence = predict[4 * num_bboxes + position]; // 跳过前4个平面，读取第5个平面
    // ----------------------------------------

    // 此时 confidence 才是真正的置信度 (0.0~1.0)
    if (confidence < confidence_threshold) return;

    // 获取存储索引
    int index = atomicAdd((int*)parray, 1);
    if (index >= max_boxes) return;

    // 中心坐标转边界坐标
    float left   = cx - w * 0.5f;
    float top    = cy - h * 0.5f;
    float right  = cx + w * 0.5f;
    float bottom = cy + h * 0.5f;

    // 坐标投影回原图
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // 存入结果数组 (每框7个元素)
    float* pbox = parray + 1 + index * 7; 
    pbox[0] = left;
    pbox[1] = top;
    pbox[2] = right;
    pbox[3] = bottom;
    pbox[4] = confidence;
    pbox[5] = 0;    // 无人机固定类别
    pbox[6] = 1;    // keep_flag
}
/**
 * @brief IOU计算函数
 */
static __device__ float box_iou(float l1, float t1, float r1, float b1, float l2, float t2, float r2, float b2) {
    float inter_l = max(l1, l2), inter_t = max(t1, t2);
    float inter_r = min(r1, r2), inter_b = min(b1, b2);
    if (inter_r <= inter_l || inter_b <= inter_t) return 0;
    float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
    float union_area = (r1 - l1) * (b1 - t1) + (r2 - l2) * (b2 - t2) - inter_area;
    return inter_area / union_area;
}

/**
 * @brief 快速 NMS 核函数
 */
static __global__ void fast_nms_kernel_uav(float* pbox, int max_boxes, float threshold) {
    int count = (int)pbox[0];
    if (count > max_boxes) count = max_boxes;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    float* p_current = pbox + 1 + idx * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float* p_comparison = pbox + 1 + i * NUM_BOX_ELEMENT;
        if (i < idx) { // 只和前面的框比（因为前面的框可能置信度更高或已经排好序）
            if (box_iou(p_current[0], p_current[1], p_current[2], p_current[3],
                        p_comparison[0], p_comparison[1], p_comparison[2], p_comparison[3]) > threshold) {
                p_current[6] = 0; // 置为丢弃
                return;
            }
        }
    }
}


static float cpu_iou(const yolo::Box& a, const yolo::Box& b) {
    float inter_l = std::max(a.left, b.left);
    float inter_t = std::max(a.top, b.top);
    float inter_r = std::min(a.right, b.right);
    float inter_b = std::min(a.bottom, b.bottom);
    if (inter_r <= inter_l || inter_b <= inter_t) return 0;
    float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
    float union_area = (a.right - a.left) * (a.bottom - a.top) + 
                       (b.right - b.left) * (b.bottom - b.top) - inter_area;
    return inter_area / union_area;
}

// CPU 端执行标准 NMS（包含排序）
void cpu_nms(yolo::BoxArray& boxes, float threshold) {
    // 1. 按置信度从高到低排序
    std::sort(boxes.begin(), boxes.end(), [](const yolo::Box& a, const yolo::Box& b) {
        return a.confidence > b.confidence;
    });

    // 2. 迭代抑制
    std::vector<bool> remove_flags(boxes.size(), false);
    for (int i = 0; i < boxes.size(); ++i) {
        if (remove_flags[i]) continue;
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) continue;
            if (cpu_iou(boxes[i], boxes[j]) > threshold) {
                remove_flags[j] = true;
            }
        }
    }

    // 3. 提取结果
    yolo::BoxArray result;
    for (int i = 0; i < boxes.size(); ++i) {
        if (!remove_flags[i]) result.push_back(boxes[i]);
    }
    boxes = std::move(result);
}

// --- 推理实现类 ---
// 使用 trt 命名空间，与 yolo.cu 保持一致
using trt::Memory;

class InferImpl : public Infer<yolo::BoxArray> {
public:
    virtual ~InferImpl() = default;

    bool load(const string& engine_file, float conf, float nms) {
        conf_thr_ = conf;
        nms_thr_  = nms;
        engine_   = trt::load(engine_file);  // 使用 trt::load
        if (!engine_) {
            printf("Failed to load UAV engine from: %s\n", engine_file.c_str());
            return false;
        }
        return true;
    }

    // 批量推理 - 使用 Image 类型（来自 BaseInfer.hpp）
    virtual std::vector<yolo::BoxArray> forwards(const std::vector<Image>& images, void* stream = nullptr) override {
        int num_image = images.size();
        if (num_image == 0) return {};
        cudaStream_t stream_ = (cudaStream_t)stream;

        printf(DBG_LBL "Starting forwards. Batch size: %d\n", num_image); fflush(stdout);

        // 1. 适配批次大小 (动态维度核心修复)
        // 使用 yolo.cu 中的 set_run_dims 接口
        auto input_dims = engine_->static_dims(0); 
        if (input_dims[0] != num_image) {
            input_dims[0] = num_image;
            if (!engine_->set_run_dims(0, input_dims)) {
                printf(DBG_ERR "Failed to set run dims (Dynamic Batch) for: %d\n", num_image);
                return std::vector<yolo::BoxArray>(num_image);
            }
        }

        // 2. 获取维度信息并调整内存
        auto output_dims = engine_->static_dims(1);
        
        /**
         * 严格修正：根据日志 Output Dims: [-1, 5, 33600]
         * output_dims[0] = -1 (Batch)
         * output_dims[1] = 5  (Attribute: cx, cy, w, h, conf) -> 对应 c_dim
         * output_dims[2] = 33600 (Anchors/Boxes) -> 对应 n_bboxes
         */
        int n_bboxes = output_dims[2]; // 33600：决定了 GPU 核函数的线程总数
        int c_dim    = output_dims[1]; // 5：决定了属性平面的数量
        
        int input_c = input_dims[1];
        int input_h = input_dims[2];
        int input_w = input_dims[3];
        
        int input_elem_per_batch = input_c * input_h * input_w;
        int output_elem_per_batch = n_bboxes * c_dim;

        // 调整 GPU 显存空间
        input_buffer_gpu_.gpu(num_image * input_elem_per_batch);
        output_buffer_gpu_.gpu(num_image * output_elem_per_batch);
        affine_matrix_d2i_.gpu(num_image * 6);

        // 处理原始图像缓存（防止异步拷贝冲突）
        size_t total_raw_bytes = 0;
        std::vector<size_t> raw_offsets(num_image);
        for (int i = 0; i < num_image; ++i) {
            raw_offsets[i] = total_raw_bytes;
            total_raw_bytes += images[i].width * images[i].height * 3;
        }
        image_buffer_gpu_.gpu(total_raw_bytes);

        // 3. 批量预处理 (WarpAffine)
        for (int i = 0; i < num_image; ++i) {
            auto& img = images[i];
            if (img.bgrptr == nullptr) continue;

            AffineMatrix am;
            am.compute(std::make_tuple(img.width, img.height), std::make_tuple(input_w, input_h));
            
            checkRuntime(cudaMemcpyAsync(affine_matrix_d2i_.gpu() + i * 6, am.d2i, sizeof(am.d2i), cudaMemcpyHostToDevice, stream_));

            uint8_t* current_raw_ptr = image_buffer_gpu_.gpu() + raw_offsets[i];
            checkRuntime(cudaMemcpyAsync(current_raw_ptr, img.bgrptr, img.width * img.height * 3, cudaMemcpyHostToDevice, stream_));

            warp_affine_bilinear_and_normalize_plane(
                current_raw_ptr, img.width * 3, img.width, img.height,
                input_buffer_gpu_.gpu() + i * input_elem_per_batch,
                input_w, input_h, affine_matrix_d2i_.gpu() + i * 6,
                114, Norm::alpha_beta(1/255.0f, 0.0f, ChannelType::SwapRB), stream_
            );
        }

        // 4. 执行模型推理
        std::vector<void*> bindings = { (void*)input_buffer_gpu_.gpu(), (void*)output_buffer_gpu_.gpu() };
        if (!engine_->forward(bindings, stream_)) {
            printf(DBG_ERR "TensorRT Forward failed!\n");
            return std::vector<yolo::BoxArray>(num_image);
        }

        // 5. 后处理解码
        // 确保 stride 计算使用统一的对齐步长
        int stride = 32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT;
        output_boxarray_.gpu(num_image * stride);
        output_boxarray_.cpu(num_image * stride);
        
        for (int i = 0; i < num_image; ++i) {
            // 每次推理前必须清理计数位（前 4 字节）
            checkRuntime(cudaMemsetAsync(output_boxarray_.gpu() + i * stride, 0, sizeof(int), stream_));

            float* gpu_out_ptr = output_boxarray_.gpu() + i * stride;
            
            // 此时 n_bboxes 为 33600，能够启动足够的并行线程
            decode_kernel_uav<<<(n_bboxes + 255) / 256, 256, 0, stream_>>>(
                output_buffer_gpu_.gpu() + i * output_elem_per_batch,
                n_bboxes, 
                c_dim, 
                conf_thr_, 
                affine_matrix_d2i_.gpu() + i * 6,
                gpu_out_ptr, 
                MAX_IMAGE_BOXES
            );
            
            // // 执行 NMS
            // fast_nms_kernel_uav<<<(MAX_IMAGE_BOXES + 255) / 256, 256, 0, stream_>>>(
            //     gpu_out_ptr, 
            //     MAX_IMAGE_BOXES, 
            //     nms_thr_
            // );
        }

        // 6. 同步并整理结果
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(), output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        std::vector<yolo::BoxArray> res(num_image);
        for (int i = 0; i < num_image; ++i) {
            float* ptr = output_boxarray_.cpu() + i * stride;
            int actual_count = *(int*)ptr; 
            int loop_limit = std::min(actual_count, MAX_IMAGE_BOXES);
            
            auto& output = res[i];
            // 先把所有过阈值的框存入 output，此时先不看 b[6]
            for (int j = 0; j < loop_limit; ++j) {
                float* b = ptr + 1 + j * NUM_BOX_ELEMENT;
                if (b[4] >= conf_thr_) { 
                    output.emplace_back(b[0], b[1], b[2], b[3], b[4], (int)b[5]);
                }
            }

            // --- 核心修复：执行精准的 CPU NMS ---
            if (!output.empty()) {
                cpu_nms(output, nms_thr_); // 使用你在 load 时传入的 nms 阈值
            }
        }
        
        if (!res.empty()) {
            printf(DBG_SUC "UAV Final Post-processed: %zu boxes\n", res[0].size());
        }
        return res;
    }   
    
    // 单张图像推理接口
    virtual yolo::BoxArray forward(const Image& image, void* stream = nullptr) override {
        auto output = forwards({image}, stream);
        return output.empty() ? yolo::BoxArray() : output[0];
    }

private:
    shared_ptr<trt::Infer> engine_;  // 使用 trt::Infer
    float conf_thr_, nms_thr_;
    // 内存管理
    Memory<float> output_boxarray_;
    Memory<float> affine_matrix_d2i_;
    // +++ 新增：管理模型输入和输出的GPU内存 +++
    Memory<float> input_buffer_gpu_;   // 用于存储预处理后的模型输入 (CHW float)
    Memory<uint8_t> image_buffer_gpu_; // 用于临时存放从CPU拷贝来的原始图像 (BGR uint8)
    Memory<float> output_buffer_gpu_;  // 用于存储模型原始输出
};

// 导出函数
std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> load(const string& file, float conf, float nms) {
    auto impl = make_shared<InferImpl>();
    if (!impl->load(file, conf, nms)) return nullptr;
    return impl;
}

} // namespace uav
