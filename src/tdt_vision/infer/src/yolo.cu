// 保持原头文件引用顺序
#include "BaseInfer.hpp"
#include "NvidiaInterface.hpp"
#include "yolos.hpp"
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda_fp16.h>

namespace yolo {

using namespace std;
using namespace tdt_radar;

static const int point_num = 4;
const int NUM_BOX_ELEMENT = 15;
const int MAX_IMAGE_BOXES = 1024;

inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

static __host__ __device__ void affine_project(
    float* matrix, 
    float x, 
    float y, 
    float* ox, 
    float* oy
)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void fp32_to_fp16_kernel(const float* src, __half* dst, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] = __float2half(src[i]);
}

static inline void fp32_to_fp16_async(const float* src, __half* dst, int n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(src, dst, n);
    checkKernel(cudaPeekAtLastError());
}


// 通用解码核函数 (FP32)
static __global__ void decode_kernel_common(
    float* predict, 
    int num_bboxes, 
    int num_classes, 
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + output_cdim * position;
    float objectness = pitem[4];
    if (!isfinite(objectness) || objectness < confidence_threshold) return;

    float* class_confidence = pitem + 5;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; 
}

// V5Face 解码核函数
static __global__ void decode_kernel_v5_face(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + output_cdim * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold) return;

    float* class_confidence = pitem + 5 + 2 * point_num;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    objectness = *pitem++; 
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    float landmark_array[point_num * 2];
    for (int i = 0; i < point_num; i++) {
        landmark_array[2 * i] = *pitem++;
        landmark_array[2 * i + 1] = *pitem++;
    }

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    for (int i = 0; i < point_num; i++) {
        affine_project(invert_affine_matrix, landmark_array[2 * i], landmark_array[2 * i + 1], 
                       &landmark_array[2 * i], &landmark_array[2 * i + 1]);
    }

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; 
    for (int i = 0; i < point_num; i++) {
        *pout_item++ = landmark_array[2 * i];
        *pout_item++ = landmark_array[2 * i + 1];
    }
}

// V8 解码核函数
static __global__ void decode_kernel_v8(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + output_cdim * position;
    float* class_confidence = pitem + 4;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    if (confidence < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; 
    *pout_item++ = position; 
}

// ---------------- V26 Decode Kernels (FP16/FP32 + NXC/CXN) ----------------
static __global__ void decode_kernel_v26_fp16_nxc(
    const __half* predict, int N, float confidence_threshold,
    const float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES
){
    // 只打印一次
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("\n=== [DEBUG V26 FP16 NXC] ===\n");
        printf("half[0]=%f half[4]=%f\n", __half2float(predict[0]), __half2float(predict[4]));
        printf("===========================\n");
    }

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= N) return;

    const __half* pitem = predict + 6 * position;

    float conf = __half2float(pitem[4]);
    if (!isfinite(conf) || conf < confidence_threshold) return;

    float left   = __half2float(pitem[0]);
    float top    = __half2float(pitem[1]);
    float right  = __half2float(pitem[2]);
    float bottom = __half2float(pitem[3]);
    float labelf = __half2float(pitem[5]);

    if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom)) return;

    int index = atomicAdd(parray, 1.0f);
    if (index >= MAX_IMAGE_BOXES) return;

    affine_project((float*)invert_affine_matrix, left, top, &left, &top);
    affine_project((float*)invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout++ = left; *pout++ = top; *pout++ = right; *pout++ = bottom;
    *pout++ = conf;
    *pout++ = (float)((int)labelf);
    *pout++ = 1;
}

static __global__ void decode_kernel_v26_fp16_cxn(
    const __half* predict, int N, float confidence_threshold,
    const float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES
){
    // layout: [6, N]
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("\n=== [DEBUG V26 FP16 CXN] ===\n");
        printf("half[C0,N0]=%f half[C4,N0]=%f\n",
               __half2float(predict[0 * N + 0]),
               __half2float(predict[4 * N + 0]));
        printf("===========================\n");
    }

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= N) return;

    float left   = __half2float(predict[0 * N + position]);
    float top    = __half2float(predict[1 * N + position]);
    float right  = __half2float(predict[2 * N + position]);
    float bottom = __half2float(predict[3 * N + position]);
    float conf   = __half2float(predict[4 * N + position]);
    float labelf = __half2float(predict[5 * N + position]);

    if (!isfinite(conf) || conf < confidence_threshold) return;
    if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom)) return;

    int index = atomicAdd(parray, 1.0f);
    if (index >= MAX_IMAGE_BOXES) return;

    affine_project((float*)invert_affine_matrix, left, top, &left, &top);
    affine_project((float*)invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout++ = left; *pout++ = top; *pout++ = right; *pout++ = bottom;
    *pout++ = conf;
    *pout++ = (float)((int)labelf);
    *pout++ = 1;
}

static __global__ void decode_kernel_v26_fp32_nxc(
    const float* predict, int N, float confidence_threshold,
    const float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES
){
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("\n=== [DEBUG V26 FP32 NXC] ===\n");
        printf("f[0]=%f f[4]=%f\n", predict[0], predict[4]);
        printf("===========================\n");
    }

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= N) return;

    const float* pitem = predict + 6 * position;

    float conf = pitem[4];
    if (!isfinite(conf) || conf < confidence_threshold) return;

    float left = pitem[0], top = pitem[1], right = pitem[2], bottom = pitem[3];
    float labelf = pitem[5];
    if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom)) return;

    int index = atomicAdd(parray, 1.0f);
    if (index >= MAX_IMAGE_BOXES) return;

    affine_project((float*)invert_affine_matrix, left, top, &left, &top);
    affine_project((float*)invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout++ = left; *pout++ = top; *pout++ = right; *pout++ = bottom;
    *pout++ = conf;
    *pout++ = (float)((int)labelf);
    *pout++ = 1;
}

static __global__ void decode_kernel_v26_fp32_cxn(
    const float* predict, int N, float confidence_threshold,
    const float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES
){
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("\n=== [DEBUG V26 FP32 CXN] ===\n");
        printf("f[C0,N0]=%f f[C4,N0]=%f\n", predict[0 * N + 0], predict[4 * N + 0]);
        printf("===========================\n");
    }

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= N) return;

    float left   = predict[0 * N + position];
    float top    = predict[1 * N + position];
    float right  = predict[2 * N + position];
    float bottom = predict[3 * N + position];
    float conf   = predict[4 * N + position];
    float labelf = predict[5 * N + position];

    if (!isfinite(conf) || conf < confidence_threshold) return;
    if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom)) return;

    int index = atomicAdd(parray, 1.0f);
    if (index >= MAX_IMAGE_BOXES) return;

    affine_project((float*)invert_affine_matrix, left, top, &left, &top);
    affine_project((float*)invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout++ = left; *pout++ = top; *pout++ = right; *pout++ = bottom;
    *pout++ = conf;
    *pout++ = (float)((int)labelf);
    *pout++ = 1;
}
// -------------------------------------------------------------------------
static __global__ void probe_v26_dtype_kernel(
    const float* as_float, int sample_half_count, int* out_valid_float, int* out_valid_half
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sample_half_count) return;

    // 1) 按 float 解读（注意 as_float 是 float*，这里按 half 索引会越界，所以只在 idx 为偶数时取 float）
    if ((idx % 2) == 0) {
        float v = as_float[idx / 2];
        if (isfinite(v) && fabsf(v) < 1e6f) atomicAdd(out_valid_float, 1);
    }

    // 2) 按 half 解读：把同一地址 reinterpret 成 half*
    const __half* as_half = (const __half*)as_float;
    float h = __half2float(as_half[idx]);
    if (isfinite(h) && fabsf(h) < 1e6f) atomicAdd(out_valid_half, 1);
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);

    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(
    float* bboxes, 
    int MAX_IMAGE_BOXES, 
    float threshold
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min((int)*bboxes, MAX_IMAGE_BOXES); 
    if (position >= count) return;

    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;

    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );
            if (iou > threshold) {
                pcurrent[6] = 0; 
                return;
            }
        }
    }
}

static dim3 grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs)
{
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

// NOTE: TensorRT bindings are typeless (void*). V26 may output FP16.
// Keep predict as void* and reinterpret with the decided dtype.
static void decode_kernel_invoker(
    void* predict, int num_bboxes, int num_classes, int output_cdim,
    float confidence_threshold, float nms_threshold,
    float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES,
    Type type, cudaStream_t stream,
    bool v26_use_fp16 = false,
    bool v26_layout_cxn = false
){
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    if (type == Type::V8 || type == Type::V8Seg) {
        float* p = (float*)predict;
        decode_kernel_v8<<<grid, block, 0, stream>>>(p, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
    } else if (type == Type::V5Face) {
        float* p = (float*)predict;
        decode_kernel_v5_face<<<grid, block, 0, stream>>>(p, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
    } else if (type == Type::V26) {
        if (v26_use_fp16) {
            const __half* p = (const __half*)predict;
            if (v26_layout_cxn)
                decode_kernel_v26_fp16_cxn<<<grid, block, 0, stream>>>(p, num_bboxes, confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
            else
                decode_kernel_v26_fp16_nxc<<<grid, block, 0, stream>>>(p, num_bboxes, confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
        } else {
            const float* p = (const float*)predict;
            if (v26_layout_cxn)
                decode_kernel_v26_fp32_cxn<<<grid, block, 0, stream>>>(p, num_bboxes, confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
            else
                decode_kernel_v26_fp32_nxc<<<grid, block, 0, stream>>>(p, num_bboxes, confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
        }
    } else {
        float* p = (float*)predict;
        decode_kernel_common<<<grid, block, 0, stream>>>(p, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES);
    }

    auto g2 = grid_dims(MAX_IMAGE_BOXES);
    auto b2 = block_dims(MAX_IMAGE_BOXES);
    fast_nms_kernel<<<g2, b2, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold);
}

static __global__ void decode_single_mask_kernel(
    int left, int top, float* mask_weights, float* mask_predict, 
    int mask_width, int mask_height, unsigned char* mask_out, 
    int mask_dim, int out_width, int out_height
)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height) return;

    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
        mask_out[dy * out_width + dx] = 0; 
        return;
    }

    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) {
        float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    float alpha = 1.0f / (1.0f + exp(-cumprod));
    mask_out[dy * out_width + dx] = alpha * 255;
}

static void decode_single_mask(
    float left, float top, float* mask_weights, float* mask_predict, 
    int mask_width, int mask_height, unsigned char* mask_out, 
    int mask_dim, int out_width, int out_height, cudaStream_t stream
)
{
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
        left, top, mask_weights, mask_predict, mask_width, mask_height,
        mask_out, mask_dim, out_width, out_height
    ));
}

const char* type_name(Type type)
{
    switch (type) {
        case Type::V5:      return "YoloV5";
        case Type::V3:      return "YoloV3";
        case Type::V7:      return "YoloV7";
        case Type::X:       return "YoloX";
        case Type::V8:      return "YoloV8";
        case Type::V8Seg:   return "YoloV8Seg";
        case Type::V5Face:  return "YoloV5Face";
        case Type::V26:     return "YoloV26";
        default:            return "Unknow";
    }
}

struct AffineMatrix {
    float i2d[6]; 
    float d2i[6]; 

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to)
    {
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);

        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

InstanceSegmentMap::InstanceSegmentMap(int width, int height)
{
    this->width = width;
    this->height = height;
    checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap()
{
    if (this->data) {
        checkRuntime(cudaFreeHost(this->data));
        this->data = nullptr;
    }
    this->width = 0;
    this->height = 0;
}

class InferImpl : public Infer<BoxArray> {
public:
    shared_ptr<trt::Infer>                         trt_;
    string                                         engine_file_;
    Type                                           type_;
    float                                          confidence_threshold_;
    float                                          nms_threshold_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float>  input_buffer_, bbox_predict_, output_boxarray_; // input_buffer_ 继续作为 FP32 预处理结果
    trt::Memory<__half> input_buffer_fp16_;                              // 新增：V26 输入 HALF
    trt::Memory<float> segment_predict_;
    int                network_input_width_, network_input_height_;
    Norm               normalize_;
    vector<int>        bbox_head_dims_;
    vector<int>        segment_head_dims_;
    int                num_classes_ = 0;
    bool               has_segment_ = false;
    bool               has_keyPoint = false;
    bool               isdynamic_model_ = false;
    vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;
    bool               v26_layout_cxn_ = false;
    int                v26_C_ = 0;
    int                v26_N_ = 0;
    bool v26_dtype_checked_ = false;
    bool v26_use_fp16_ = false;
    int input_binding_ = 0;
    int bbox_binding_  = 1;



    virtual ~InferImpl() = default;

    void adjust_memory(int batch_size)
    {
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        input_buffer_fp16_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if (has_segment_)
            segment_predict_.gpu(batch_size * segment_head_dims_[1] *
                                 segment_head_dims_[2] * segment_head_dims_[3]);

        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
        }
    }

    void preprocess(
        int ibatch, const Image& image,
        shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
        AffineMatrix& affine, void* stream = nullptr
    )
    {
        affine.compute(
            make_tuple(image.width, image.height),
            make_tuple(network_input_width_, network_input_height_)
        );

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float* input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = image.width * image.height * 3;
        size_t size_matrix = upbound(sizeof(affine.d2i), 32);

        uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float* affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device = gpu_workspace + size_matrix;

        uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float* affine_matrix_host = (float*)cpu_workspace;
        uint8_t* image_host = cpu_workspace + size_matrix;

        cudaStream_t stream_ = (cudaStream_t)stream;

        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));

        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3, image.width, image.height,
            input_device, network_input_width_, network_input_height_,
            affine_matrix_device, 114, normalize_, stream_
        );
    }

    bool load(const string& engine_file, Type type, float confidence_threshold, float nms_threshold)
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr) return false;
        trt_->print();

        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;
        this->engine_file_ = engine_file;

        auto input_dim = trt_->static_dims(0);
        has_segment_ = (type == Type::V8Seg);
        has_keyPoint = (type == Type::V5Face);

        if (has_segment_) {
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        } else {
            bbox_head_dims_ = trt_->static_dims(1);
        }

        // [DEBUG] 打印维度信息，辅助判断 Binding 是否错位
        // 如果 static_dims(0) 不是输入（如4维NCHW），而是3维（输出），则说明 Binding 顺序反了
        int d0 = input_dim.size() > 0 ? input_dim[0] : -1;
        int d1 = input_dim.size() > 1 ? input_dim[1] : -1;
        int d2 = input_dim.size() > 2 ? input_dim[2] : -1;
        int d3 = input_dim.size() > 3 ? input_dim[3] : -1;
        printf("\n[DEBUG] TRT Bindings Check:\n");
        printf("Binding 0 Dims: [%d, %d, %d, %d] (Expected Input NCHW)\n", d0, d1, d2, d3);

        auto dim0 = trt_->static_dims(0);
        auto dim1 = trt_->static_dims(1);

        auto is_input_like = [](const std::vector<int>& d){
            // 典型输入: [N,3,H,W]
            return d.size() == 4 && (d[1] == 3 || d[1] == 1) && d[2] > 16 && d[3] > 16;
        };

        bool b0_is_input = is_input_like(dim0);
        bool b1_is_input = is_input_like(dim1);

        // 判定 binding 索引
        if (b0_is_input && !b1_is_input) {
            input_binding_ = 0; bbox_binding_ = 1;
        } else if (!b0_is_input && b1_is_input) {
            input_binding_ = 1; bbox_binding_ = 0;
        } else {
            // 极端情况：都像 input 或都不像 input，先按老逻辑，但必须打印警告
            input_binding_ = 0; bbox_binding_ = 1;
            printf("[WARN] Cannot uniquely identify input/output bindings by dims.\n");
        }

        printf("[DEBUG] binding decision: input_binding=%d bbox_binding=%d\n", input_binding_, bbox_binding_);
        printf("[DEBUG] dims(input)=[%d,%d,%d,%d]\n",
            trt_->static_dims(input_binding_)[0], trt_->static_dims(input_binding_)[1],
            trt_->static_dims(input_binding_)[2], trt_->static_dims(input_binding_)[3]);

        
        auto input_dim_bind = trt_->static_dims(input_binding_);
        network_input_width_  = input_dim_bind[3];
        network_input_height_ = input_dim_bind[2];

        if (has_segment_) {
            // 这里先不动（你的 seg engine 不是现在的问题）
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        } else {
            bbox_head_dims_ = trt_->static_dims(bbox_binding_);
        }

        isdynamic_model_ = trt_->has_dynamic_dim();

        if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V8) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;
        } else if (type == Type::V8Seg) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
        } else if (type == Type::X) {
            normalize_ = Norm::None();
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V5Face) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5 - 2 * point_num;
        } else if (type == Type::V26) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            int d1 = bbox_head_dims_[1];
            int d2 = bbox_head_dims_[2];
            int C = d2, N = d1;
            bool layout_cxn = false;
            
            // 你的 car_merged_200_fp16.engine: output0 is FLOAT (1,300,6)
            v26_use_fp16_ = false;
            v26_dtype_checked_ = true;   // 跳过 probe，避免误判
            
            if (d2 == 6) { C = 6; N = d1; layout_cxn = false; }
            else if (d1 == 6) { C = 6; N = d2; layout_cxn = true; }
            else {
                if (d1 > 6 && d1 < 512 && d2 > 512) { C = d1; N = d2; layout_cxn = true; }
                else { C = d2; N = d1; layout_cxn = false; }
            }
            
            this->v26_layout_cxn_ = layout_cxn;
            this->v26_C_ = C;
            this->v26_N_ = N;
            
            if (C == 6) num_classes_ = 0;
            else {
                num_classes_ = C - 4;
                if (num_classes_ < 1) num_classes_ = 1;
            }
        } else {
            INFO("Unsupport type %d", type);
            return false;
        }
        return true;
    }

    virtual BoxArray forward(const Image& image, void* stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        return output.empty() ? BoxArray() : output[0];
    }

    virtual vector<BoxArray> forwards(const vector<Image>& images, void* stream = nullptr) override
    {
        int num_image = images.size();
        if (num_image == 0) return {};

        auto input_dims = trt_->static_dims(input_binding_);
        int  infer_batch_size = input_dims[0];

        if (infer_batch_size != num_image) {
            if (isdynamic_model_) {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims)) return {};
            } else {
                if (infer_batch_size < num_image) {
                    INFO("When using static shape model, number of images[%d] must be less than or equal to the maximum batch[%d].", num_image, infer_batch_size);
                    return {};
                }
            }
        }

        adjust_memory(infer_batch_size);
        
        vector<AffineMatrix> affine_matrixs(num_image);
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i) {
            preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);
        }

        // [DEBUG] 检查输入数据是否正常 (防止 Input NaN)
        // 打印第一个图片的第一个通道的前5个值
        // 只需要做一次检查即可
        static bool input_checked = false;
        if (!input_checked) {
            float* debug_input = new float[10];
            cudaMemcpy(debug_input, input_buffer_.gpu(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
            printf("\n[DEBUG] Input Buffer Check (First 5 values): %f, %f, %f, %f, %f\n", 
                   debug_input[0], debug_input[1], debug_input[2], debug_input[3], debug_input[4]);
            delete[] debug_input;
            input_checked = true;
        }

        float* bbox_output_device = bbox_predict_.gpu();

        // === V26: FP32 -> FP16 (engine input is HALF) ===
        if (type_ == Type::V26) {
            int numel = (int)(infer_batch_size * network_input_width_ * network_input_height_ * 3);
            fp32_to_fp16_async(input_buffer_.gpu(), input_buffer_fp16_.gpu(), numel, stream_);
        }

        vector<void*> bindings(2, nullptr);
        bindings[input_binding_] = (type_ == Type::V26) ? (void*)input_buffer_fp16_.gpu() : (void*)input_buffer_.gpu();
        bindings[bbox_binding_]  = (void*)bbox_output_device;

        if (has_segment_) {
            // seg 的情况你先别动逻辑，但 input 也要一致
            void* in_ptr = (type_ == Type::V26) ? (void*)input_buffer_fp16_.gpu() : (void*)input_buffer_.gpu();
            bindings = {in_ptr, segment_predict_.gpu(), bbox_output_device};
        }


        checkRuntime(cudaMemsetAsync(bbox_output_device, 0, bbox_predict_.gpu_bytes(), stream_));


        if (!trt_->forward(bindings, stream)) {
            INFO("Failed to tensorRT forward.");
            return {};
        }

        float host_check[8];
        checkRuntime(cudaMemcpyAsync(host_check, bbox_output_device, sizeof(host_check),
                                    cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));
        printf("[POST-FWD] first floats: %f %f %f %f %f %f %f %f\n",
            host_check[0],host_check[1],host_check[2],host_check[3],
            host_check[4],host_check[5],host_check[6],host_check[7]);


        if (type_ == Type::V26 && !v26_dtype_checked_) {
            // 取 batch0 的输出指针进行探测
            float* ptr = bbox_output_device;

            // device counters
            int* d_valid_float = nullptr;
            int* d_valid_half  = nullptr;
            checkRuntime(cudaMalloc(&d_valid_float, sizeof(int)));
            checkRuntime(cudaMalloc(&d_valid_half,  sizeof(int)));
            checkRuntime(cudaMemsetAsync(d_valid_float, 0, sizeof(int), stream_));
            checkRuntime(cudaMemsetAsync(d_valid_half,  0, sizeof(int), stream_));

            // 采样 2048 个 half（=1024 个 float）
            int sample_half = 2048;
            dim3 blk(256);
            dim3 grd((sample_half + blk.x - 1) / blk.x);
            probe_v26_dtype_kernel<<<grd, blk, 0, stream_>>>(ptr, sample_half, d_valid_float, d_valid_half);
            checkKernel(cudaPeekAtLastError());

            int h_valid_float = 0, h_valid_half = 0;
            checkRuntime(cudaMemcpyAsync(&h_valid_float, d_valid_float, sizeof(int), cudaMemcpyDeviceToHost, stream_));
            checkRuntime(cudaMemcpyAsync(&h_valid_half,  d_valid_half,  sizeof(int), cudaMemcpyDeviceToHost, stream_));
            checkRuntime(cudaStreamSynchronize(stream_));

            checkRuntime(cudaFree(d_valid_float));
            checkRuntime(cudaFree(d_valid_half));

            // float 路径最多只能在 sample_half/2 的线程里计数
            int float_cap = sample_half / 2;
            printf("\n[DEBUG V26 DTYPE PROBE] valid_float=%d/%d valid_half=%d/%d\n",
                h_valid_float, float_cap, h_valid_half, sample_half);

            // 判定：half 的有效率显著更高则认为 FP16
            float rate_f = (float)h_valid_float / (float)float_cap;
            float rate_h = (float)h_valid_half  / (float)sample_half;
            v26_use_fp16_ = (rate_h > rate_f + 0.20f);  // 留个安全 margin
            v26_dtype_checked_ = true;

            printf("[DEBUG V26 DTYPE PROBE] decide: %s\n", v26_use_fp16_ ? "FP16" : "FP32");
        }

        if (type_ == Type::V26 && !v26_dtype_checked_) {
            // 读 FP32 视角
            float fbuf[6];
            checkRuntime(cudaMemcpyAsync(fbuf, bbox_output_device, 6*sizeof(float),
                                        cudaMemcpyDeviceToHost, stream_));
            // 读 FP16 视角：同地址当 half 读
            __half htmp[6];
            checkRuntime(cudaMemcpyAsync(htmp, bbox_output_device, 6*sizeof(__half),
                                        cudaMemcpyDeviceToHost, stream_));
            checkRuntime(cudaStreamSynchronize((cudaStream_t)stream_));

            float hbuf[6];
            for (int i = 0; i < 6; ++i) hbuf[i] = __half2float(htmp[i]);

            printf("\n[V26 PROBE] FP32 f0..f5: %f %f %f %f %f %f\n",
                fbuf[0],fbuf[1],fbuf[2],fbuf[3],fbuf[4],fbuf[5]);
            printf("[V26 PROBE] FP16 h0..h5: %f %f %f %f %f %f\n",
                hbuf[0],hbuf[1],hbuf[2],hbuf[3],hbuf[4],hbuf[5]);

            auto ok_pack = [](const float* p){
                float l=p[0], t=p[1], r=p[2], b=p[3], c=p[4];
                if (!isfinite(l)||!isfinite(t)||!isfinite(r)||!isfinite(b)||!isfinite(c)) return false;
                // 置信度通常在 0~1（留点余量）
                if (c < 0.f || c > 1.5f) return false;
                // bbox 合理性
                if (r < l || b < t) return false;
                return true;
            };

            bool ok32 = ok_pack(fbuf);
            bool ok16 = ok_pack(hbuf);

            // 判定策略：优先相信“结构上合理”的那一个
            if (!ok32 && ok16) v26_use_fp16_ = true;
            else if (ok32 && !ok16) v26_use_fp16_ = false;
            else if (!ok32 && !ok16) {
                // 两边都不合理：先不要强制，打印后续再看（先按 FP32 走）
                v26_use_fp16_ = false;
                printf("[V26 PROBE] both views look invalid, default FP32 for now\n");
            } else {
                // 两边都看起来合理：一般是 output 真是 FP32（half 视角“碰巧”合理概率很低）
                v26_use_fp16_ = false;
            }

            v26_dtype_checked_ = true;
            printf("[V26 PROBE] decide: %s\n", v26_use_fp16_ ? "FP16" : "FP32");
        }



        for (int ib = 0; ib < num_image; ++ib) {
            float* boxarray_device = output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float* affine_matrix_device = (float*)preprocess_buffers_[ib]->gpu();
            size_t out_elems = (size_t)bbox_head_dims_[1] * (size_t)bbox_head_dims_[2];
            size_t elem_size = sizeof(float);
            if (type_ == Type::V26 && v26_use_fp16_) elem_size = sizeof(__half);

            uint8_t* bbox_base = (uint8_t*)bbox_output_device;
            void* image_based_bbox_output = (void*)(bbox_base + (size_t)ib * out_elems * elem_size);
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            int N_for_decode = bbox_head_dims_[1];
            int C_for_decode = bbox_head_dims_[2];
            

            if (type_ == Type::V26) {
                N_for_decode = v26_N_;
                C_for_decode = v26_C_;
            }
            decode_kernel_invoker(
                image_based_bbox_output,
                N_for_decode, num_classes_, C_for_decode,
                confidence_threshold_, nms_threshold_,
                affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES,
                type_, stream_,
                v26_use_fp16_, v26_layout_cxn_
            );
        }

        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(), output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        vector<BoxArray> arrout(num_image);
        int imemory = 0;

        for (int ib = 0; ib < num_image; ++ib) {
            float* parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count = min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray& output = arrout[ib];
            output.reserve(count);

            for (int i = 0; i < count; ++i) {
                float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                float left = pbox[0], top = pbox[1], right = pbox[2], bottom = pbox[3];
                float confidence = pbox[4];
                int   label = pbox[5], keepflag = pbox[6];
                std::vector<float> points;

                for (int j = 1; j < 9; j++) points.push_back(pbox[6 + j]);

                if (keepflag == 1) {
                    Box result_box(left, top, right, bottom, confidence, label, points);

                    if (has_segment_) {
                        int row_index = pbox[7];
                        int mask_dim = segment_head_dims_[1];
                        float* mask_weights = bbox_output_device + (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] + num_classes_ + 4;

                        float* i2d = affine_matrixs[ib].i2d;
                        float mask_left, mask_top, mask_right, mask_bottom;
                        affine_project(i2d, left, top, &mask_left, &mask_top);
                        affine_project(i2d, right, bottom, &mask_right, &mask_bottom);

                        float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
                        float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
                        int mask_out_width = (mask_right - mask_left) * scale_to_predict_x + 0.5f;
                        int mask_out_height = (mask_bottom - mask_top) * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0) {
                            if (imemory >= (int)box_segment_cache_.size()) {
                                box_segment_cache_.push_back(make_shared<trt::Memory<unsigned char>>());
                            }

                            int bytes_of_mask_out = mask_out_width * mask_out_height;
                            auto box_segment_mem = box_segment_cache_[imemory++];
                            result_box.seg = make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
                            unsigned char* mask_out_device = box_segment_mem->gpu(bytes_of_mask_out);
                            unsigned char* mask_out_host = result_box.seg->data;

                            decode_single_mask(
                                mask_left * scale_to_predict_x, mask_top * scale_to_predict_y,
                                mask_weights, segment_predict_.gpu() + ib * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3],
                                segment_head_dims_[3], segment_head_dims_[2],
                                mask_out_device, mask_dim, mask_out_width, mask_out_height, stream_
                            );

                            checkRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device, bytes_of_mask_out, cudaMemcpyDeviceToHost, stream_));
                        }
                    }
                    output.emplace_back(result_box);
                }
            }
        }

        if (has_segment_) checkRuntime(cudaStreamSynchronize(stream_));
        return arrout;
    }
};

Infer<BoxArray>* loadraw(const std::string& engine_file, Type type, float confidence_threshold, float nms_threshold)
{
    InferImpl* impl = new InferImpl();
    if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer<BoxArray>> load(const string& engine_file, Type type, float confidence_threshold, float nms_threshold)
{
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(engine_file, type, confidence_threshold, nms_threshold));
}

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int   h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float       r, g, b;

    switch (h_i) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
        default: r = 1, g = 1, b = 1; break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

}; // namespace yolo