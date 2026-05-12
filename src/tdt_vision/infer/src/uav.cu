#define _HAS_STD_BYTE 0
#include <cstdint>
#include <cuda_fp16.h>

#include "uav.hpp"
#include "BaseInfer.hpp"
#include "InferTool.hpp"
#include "NvidiaInterface.hpp"
#include "yolos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

namespace uav {

using namespace std;
using namespace tdt_radar;
using trt::Memory;

#define DBG_ERR "\033[1;31m[ERROR]\033[0m "

namespace {

constexpr int NUM_BOX_ELEMENT = 7;
constexpr int MAX_IMAGE_BOXES = 1024;
constexpr int MAX_SUPPORTED_ATTR = 128;

enum class OutputLayout {
    kUnknown = 0,
    kCXN = 1,
    kNXC = 2,
};

const char* layoutName(OutputLayout layout)
{
    switch (layout) {
        case OutputLayout::kCXN: return "CXN";
        case OutputLayout::kNXC: return "NXC";
        default: return "Unknown";
    }
}

const char* semanticsName(bool post_nms_xyxy)
{
    return post_nms_xyxy ? "post-nms-xyxy-conf-cls" : "raw-cxcywh-conf";
}

const char* dtypeName(trt::DType dtype)
{
    switch (dtype) {
        case trt::DType::FLOAT: return "float32";
        case trt::DType::HALF: return "float16";
        case trt::DType::INT8: return "int8";
        case trt::DType::INT32: return "int32";
        case trt::DType::BOOL: return "bool";
        case trt::DType::UINT8: return "uint8";
        default: return "unknown";
    }
}

struct UavAffineMatrix {
    float i2d[6];
    float d2i[6];

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to)
    {
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);

        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5f + get<0>(to) * 0.5f + scale * 0.5f - 0.5f;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5f + get<1>(to) * 0.5f + scale * 0.5f - 0.5f;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0.0 ? 1.0 / D : 0.0;
        double A11 = i2d[4] * D;
        double A22 = i2d[0] * D;
        double A12 = -i2d[1] * D;
        double A21 = -i2d[3] * D;
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

static __host__ __device__ void affine_project(
    const float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void fp32_to_fp16_kernel(const float* src, __half* dst, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        dst[index] = __float2half(src[index]);
    }
}

inline void fp32_to_fp16_async(
    const float* src, __half* dst, int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(src, dst, n);
    checkKernel(cudaPeekAtLastError());
}

__device__ __forceinline__ float readTensorValue(float value)
{
    return value;
}

__device__ __forceinline__ float readTensorValue(__half value)
{
    return __half2float(value);
}

template <typename T, OutputLayout kLayout>
__global__ void decode_kernel_uav(
    const T* predict,
    int num_bboxes,
    int output_cdim,
    bool post_nms_xyxy,
    float confidence_threshold,
    const float* invert_affine_matrix,
    float* parray,
    int max_boxes)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) {
        return;
    }

    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float confidence = 0.0f;
    float label = 0.0f;

    if constexpr (kLayout == OutputLayout::kCXN) {
        cx = readTensorValue(predict[0 * num_bboxes + position]);
        cy = readTensorValue(predict[1 * num_bboxes + position]);
        w = readTensorValue(predict[2 * num_bboxes + position]);
        h = readTensorValue(predict[3 * num_bboxes + position]);
        confidence = readTensorValue(predict[4 * num_bboxes + position]);
        if (output_cdim > 5) {
            label = readTensorValue(predict[5 * num_bboxes + position]);
        }
    } else {
        const T* item = predict + position * output_cdim;
        cx = readTensorValue(item[0]);
        cy = readTensorValue(item[1]);
        w = readTensorValue(item[2]);
        h = readTensorValue(item[3]);
        confidence = readTensorValue(item[4]);
        if (output_cdim > 5) {
            label = readTensorValue(item[5]);
        }
    }

    if (!isfinite(confidence) || confidence < confidence_threshold) {
        return;
    }

    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
    if (post_nms_xyxy) {
        left = cx;
        top = cy;
        right = w;
        bottom = h;
    } else {
        left = cx - w * 0.5f;
        top = cy - h * 0.5f;
        right = cx + w * 0.5f;
        bottom = cy + h * 0.5f;
    }

    if (!isfinite(left) || !isfinite(top) ||
        !isfinite(right) || !isfinite(bottom) ||
        right <= left || bottom <= top) {
        return;
    }

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    if (!isfinite(left) || !isfinite(top) ||
        !isfinite(right) || !isfinite(bottom) ||
        right <= left || bottom <= top) {
        return;
    }

    int index = atomicAdd((int*)parray, 1);
    if (index >= max_boxes) {
        return;
    }

    float* pbox = parray + 1 + index * NUM_BOX_ELEMENT;
    pbox[0] = left;
    pbox[1] = top;
    pbox[2] = right;
    pbox[3] = bottom;
    pbox[4] = confidence;
    pbox[5] = label;
    pbox[6] = 1.0f;
}

template <typename T>
void launchDecodeKernel(
    const T* predict,
    OutputLayout layout,
    int num_bboxes,
    int output_cdim,
    bool post_nms_xyxy,
    float confidence_threshold,
    const float* invert_affine_matrix,
    float* parray,
    int max_boxes,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((num_bboxes + block.x - 1) / block.x);

    switch (layout) {
        case OutputLayout::kCXN:
            checkKernel((decode_kernel_uav<T, OutputLayout::kCXN><<<grid, block, 0, stream>>>(
                predict, num_bboxes, output_cdim, post_nms_xyxy, confidence_threshold,
                invert_affine_matrix, parray, max_boxes)));
            return;
        case OutputLayout::kNXC:
            checkKernel((decode_kernel_uav<T, OutputLayout::kNXC><<<grid, block, 0, stream>>>(
                predict, num_bboxes, output_cdim, post_nms_xyxy, confidence_threshold,
                invert_affine_matrix, parray, max_boxes)));
            return;
        default:
            return;
    }
}

static float cpu_iou(const yolo::Box& a, const yolo::Box& b)
{
    float inter_l = std::max(a.left, b.left);
    float inter_t = std::max(a.top, b.top);
    float inter_r = std::min(a.right, b.right);
    float inter_b = std::min(a.bottom, b.bottom);
    if (inter_r <= inter_l || inter_b <= inter_t) {
        return 0.0f;
    }
    float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
    float union_area =
        (a.right - a.left) * (a.bottom - a.top) +
        (b.right - b.left) * (b.bottom - b.top) -
        inter_area;
    return inter_area / union_area;
}

void cpu_nms(yolo::BoxArray& boxes, float threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const yolo::Box& a, const yolo::Box& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> remove_flags(boxes.size(), false);
    for (int i = 0; i < (int)boxes.size(); ++i) {
        if (remove_flags[i]) {
            continue;
        }
        for (int j = i + 1; j < (int)boxes.size(); ++j) {
            if (remove_flags[j]) {
                continue;
            }
            if (cpu_iou(boxes[i], boxes[j]) > threshold) {
                remove_flags[j] = true;
            }
        }
    }

    yolo::BoxArray result;
    for (int i = 0; i < (int)boxes.size(); ++i) {
        if (!remove_flags[i]) {
            result.push_back(boxes[i]);
        }
    }
    boxes = std::move(result);
}

}  // namespace

class InferImpl : public Infer<yolo::BoxArray> {
public:
    virtual ~InferImpl() = default;

    bool load(const string& engine_file, float conf, float nms)
    {
        conf_thr_ = conf;
        nms_thr_ = nms;
        engine_ = trt::load(engine_file);
        if (!engine_) {
            printf("Failed to load UAV engine from: %s\n", engine_file.c_str());
            return false;
        }

        // Noisy TensorRT tensor dump; re-enable only when debugging UAV engine bindings.
        // engine_->print();

        int num_bindings = engine_->num_bindings();
        for (int ib = 0; ib < num_bindings; ++ib) {
            if (engine_->is_input(ib)) {
                if (input_binding_ != -1) {
                    printf(DBG_ERR "UAV engine expects exactly one input tensor.\n");
                    return false;
                }
                input_binding_ = ib;
            } else {
                if (output_binding_ != -1) {
                    printf(DBG_ERR "UAV engine expects exactly one output tensor.\n");
                    return false;
                }
                output_binding_ = ib;
            }
        }

        if (input_binding_ < 0 || output_binding_ < 0) {
            printf(DBG_ERR "UAV engine input/output tensor discovery failed.\n");
            return false;
        }

        input_dtype_ = engine_->dtype(input_binding_);
        output_dtype_ = engine_->dtype(output_binding_);
        if (input_dtype_ != trt::DType::FLOAT && input_dtype_ != trt::DType::HALF) {
            printf(DBG_ERR "Unsupported UAV input dtype: %s\n", dtypeName(input_dtype_));
            return false;
        }
        if (output_dtype_ != trt::DType::FLOAT && output_dtype_ != trt::DType::HALF) {
            printf(DBG_ERR "Unsupported UAV output dtype: %s\n", dtypeName(output_dtype_));
            return false;
        }

        input_is_fp16_ = input_dtype_ == trt::DType::HALF;
        output_is_fp16_ = output_dtype_ == trt::DType::HALF;
        is_dynamic_model_ = engine_->has_dynamic_dim();

        auto input_dims = engine_->static_dims(input_binding_);
        if (input_dims.size() != 4) {
            printf(DBG_ERR "UAV engine input rank must be 4, got {%s}\n",
                   trt::format_shape(input_dims).c_str());
            return false;
        }

        if (input_dims[1] > 0) {
            network_input_channels_ = input_dims[1];
        }
        network_input_height_ = input_dims[2];
        network_input_width_ = input_dims[3];

        if (network_input_channels_ != 3) {
            printf(DBG_ERR "UAV engine expects 3-channel input, got %d\n",
                   network_input_channels_);
            return false;
        }
        if (network_input_height_ <= 0 || network_input_width_ <= 0) {
            printf(DBG_ERR "UAV engine uses dynamic H/W input {%s}; current UAV pipeline expects fixed H/W.\n",
                   trt::format_shape(input_dims).c_str());
            return false;
        }

        if (is_dynamic_model_) {
            input_dims[0] = 1;
            if (!engine_->set_run_dims(input_binding_, input_dims)) {
                printf(DBG_ERR "Failed to prime UAV engine run dims with {%s}\n",
                       trt::format_shape(input_dims).c_str());
                return false;
            }
        }

        if (!refreshOutputDescription()) {
            return false;
        }

        printf(
            "[UAV] engine ready: input=%s output=%s layout=%s input_shape={%d,%d,%d,%d} output_cdim=%d output_boxes=%d\n",
            dtypeName(input_dtype_),
            dtypeName(output_dtype_),
            layoutName(output_layout_),
            1,
            network_input_channels_,
            network_input_height_,
            network_input_width_,
            output_cdim_,
            output_n_bboxes_);
        printf("[UAV] output semantics: %s\n", semanticsName(output_is_post_nms_xyxy_));
        return true;
    }

    virtual std::vector<yolo::BoxArray> forwards(
        const std::vector<Image>& images, void* stream = nullptr) override
    {
        const int num_image = images.size();
        if (num_image == 0) {
            return {};
        }

        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        auto input_dims = engine_->static_dims(input_binding_);
        if (input_dims.size() != 4) {
            printf(DBG_ERR "Unexpected UAV input dims while running.\n");
            return std::vector<yolo::BoxArray>(num_image);
        }

        int infer_batch_size = input_dims[0];
        if (infer_batch_size <= 0) {
            infer_batch_size = num_image;
        }

        if (input_dims[0] != num_image) {
            if (is_dynamic_model_) {
                input_dims[0] = num_image;
                input_dims[1] = network_input_channels_;
                input_dims[2] = network_input_height_;
                input_dims[3] = network_input_width_;
                if (!engine_->set_run_dims(input_binding_, input_dims)) {
                    printf(DBG_ERR "Failed to set UAV run dims to {%s}\n",
                           trt::format_shape(input_dims).c_str());
                    return std::vector<yolo::BoxArray>(num_image);
                }
                infer_batch_size = num_image;
            } else if (infer_batch_size < num_image) {
                printf(DBG_ERR "Static UAV engine batch=%d, but received %d images.\n",
                       infer_batch_size, num_image);
                return std::vector<yolo::BoxArray>(num_image);
            }
        }

        auto run_input_dims = engine_->run_dims(input_binding_);
        if (run_input_dims.size() == 4 && run_input_dims[0] > 0) {
            infer_batch_size = run_input_dims[0];
        }

        if (!refreshOutputDescription()) {
            return std::vector<yolo::BoxArray>(num_image);
        }

        const size_t input_elem_per_batch =
            (size_t)network_input_channels_ *
            (size_t)network_input_height_ *
            (size_t)network_input_width_;
        size_t output_elem_total = engine_->numel(output_binding_);
        if (output_elem_total == 0) {
            output_elem_total =
                (size_t)infer_batch_size *
                (size_t)output_cdim_ *
                (size_t)output_n_bboxes_;
        }
        if (output_elem_total % infer_batch_size != 0) {
            printf(DBG_ERR "UAV output numel=%zu is incompatible with batch=%d.\n",
                   output_elem_total, infer_batch_size);
            return std::vector<yolo::BoxArray>(num_image);
        }
        const size_t output_elem_per_batch = output_elem_total / infer_batch_size;

        input_buffer_gpu_.gpu((size_t)infer_batch_size * input_elem_per_batch);
        if (infer_batch_size > num_image) {
            checkRuntime(cudaMemsetAsync(
                input_buffer_gpu_.gpu() + (size_t)num_image * input_elem_per_batch,
                0,
                ((size_t)infer_batch_size - (size_t)num_image) * input_elem_per_batch * sizeof(float),
                stream_));
        }

        if (input_is_fp16_) {
            input_buffer_half_gpu_.gpu((size_t)infer_batch_size * input_elem_per_batch);
        }
        if (output_is_fp16_) {
            output_buffer_half_gpu_.gpu((size_t)infer_batch_size * output_elem_per_batch);
        } else {
            output_buffer_float_gpu_.gpu((size_t)infer_batch_size * output_elem_per_batch);
        }
        affine_matrix_d2i_.gpu((size_t)num_image * 6);

        size_t total_raw_bytes = 0;
        std::vector<size_t> raw_offsets(num_image, 0);
        for (int i = 0; i < num_image; ++i) {
            raw_offsets[i] = total_raw_bytes;
            total_raw_bytes += (size_t)images[i].width * (size_t)images[i].height * 3;
        }
        image_buffer_gpu_.gpu(total_raw_bytes);

        for (int i = 0; i < num_image; ++i) {
            const auto& img = images[i];
            if (img.bgrptr == nullptr || img.width <= 0 || img.height <= 0) {
                continue;
            }

            UavAffineMatrix affine;
            affine.compute(
                std::make_tuple(img.width, img.height),
                std::make_tuple(network_input_width_, network_input_height_));

            checkRuntime(cudaMemcpyAsync(
                affine_matrix_d2i_.gpu() + (size_t)i * 6,
                affine.d2i,
                sizeof(affine.d2i),
                cudaMemcpyHostToDevice,
                stream_));

            uint8_t* current_raw_ptr = image_buffer_gpu_.gpu() + raw_offsets[i];
            const size_t image_bytes = (size_t)img.width * (size_t)img.height * 3;
            checkRuntime(cudaMemcpyAsync(
                current_raw_ptr,
                img.bgrptr,
                image_bytes,
                cudaMemcpyHostToDevice,
                stream_));

            warp_affine_bilinear_and_normalize_plane(
                current_raw_ptr,
                img.width * 3,
                img.width,
                img.height,
                input_buffer_gpu_.gpu() + (size_t)i * input_elem_per_batch,
                network_input_width_,
                network_input_height_,
                affine_matrix_d2i_.gpu() + (size_t)i * 6,
                114,
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB),
                stream_);
        }

        void* input_binding_ptr = (void*)input_buffer_gpu_.gpu();
        if (input_is_fp16_) {
            fp32_to_fp16_async(
                input_buffer_gpu_.gpu(),
                input_buffer_half_gpu_.gpu(),
                (int)((size_t)infer_batch_size * input_elem_per_batch),
                stream_);
            input_binding_ptr = (void*)input_buffer_half_gpu_.gpu();
        }

        std::vector<void*> bindings(engine_->num_bindings(), nullptr);
        bindings[input_binding_] = input_binding_ptr;
        bindings[output_binding_] = output_is_fp16_
            ? (void*)output_buffer_half_gpu_.gpu()
            : (void*)output_buffer_float_gpu_.gpu();

        if (!engine_->forward(bindings, stream_)) {
            printf(DBG_ERR "TensorRT forward failed for UAV engine.\n");
            return std::vector<yolo::BoxArray>(num_image);
        }

        const int stride = 32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT;
        output_boxarray_.gpu((size_t)num_image * stride);
        output_boxarray_.cpu((size_t)num_image * stride);

        for (int ib = 0; ib < num_image; ++ib) {
            float* gpu_out_ptr = output_boxarray_.gpu() + (size_t)ib * stride;
            checkRuntime(cudaMemsetAsync(gpu_out_ptr, 0, sizeof(int), stream_));

            if (output_is_fp16_) {
                const __half* output_ptr =
                    output_buffer_half_gpu_.gpu() + (size_t)ib * output_elem_per_batch;
                launchDecodeKernel(
                    output_ptr,
                    output_layout_,
                    output_n_bboxes_,
                    output_cdim_,
                    output_is_post_nms_xyxy_,
                    conf_thr_,
                    affine_matrix_d2i_.gpu() + (size_t)ib * 6,
                    gpu_out_ptr,
                    MAX_IMAGE_BOXES,
                    stream_);
            } else {
                const float* output_ptr =
                    output_buffer_float_gpu_.gpu() + (size_t)ib * output_elem_per_batch;
                launchDecodeKernel(
                    output_ptr,
                    output_layout_,
                    output_n_bboxes_,
                    output_cdim_,
                    output_is_post_nms_xyxy_,
                    conf_thr_,
                    affine_matrix_d2i_.gpu() + (size_t)ib * 6,
                    gpu_out_ptr,
                    MAX_IMAGE_BOXES,
                    stream_);
            }
        }

        checkRuntime(cudaMemcpyAsync(
            output_boxarray_.cpu(),
            output_boxarray_.gpu(),
            output_boxarray_.gpu_bytes(),
            cudaMemcpyDeviceToHost,
            stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        std::vector<yolo::BoxArray> result(num_image);
        for (int ib = 0; ib < num_image; ++ib) {
            float* ptr = output_boxarray_.cpu() + (size_t)ib * stride;
            int actual_count = *(int*)ptr;
            int loop_limit = std::min(actual_count, MAX_IMAGE_BOXES);

            auto& output = result[ib];
            for (int i = 0; i < loop_limit; ++i) {
                float* box = ptr + 1 + i * NUM_BOX_ELEMENT;
                if (box[4] >= conf_thr_) {
                    output.emplace_back(
                        box[0], box[1], box[2], box[3], box[4], (int)box[5]);
                }
            }
            if (!output.empty()) {
                cpu_nms(output, nms_thr_);
            }
        }

        return result;
    }

    virtual yolo::BoxArray forward(const Image& image, void* stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        return output.empty() ? yolo::BoxArray() : output[0];
    }

private:
    bool refreshOutputDescription()
    {
        auto output_dims = engine_->run_dims(output_binding_);
        if (output_dims.empty()) {
            output_dims = engine_->static_dims(output_binding_);
        }
        return parseOutputDims(output_dims);
    }

    bool parseOutputDims(const std::vector<int>& output_dims)
    {
        std::vector<int> dims = output_dims;
        if (dims.size() == 3) {
            dims.erase(dims.begin());
        }
        if (dims.size() != 2) {
            printf(DBG_ERR "Unsupported UAV output rank: {%s}\n",
                   trt::format_shape(output_dims).c_str());
            return false;
        }

        int d0 = dims[0];
        int d1 = dims[1];
        if (d0 <= 0 || d1 <= 0) {
            printf(DBG_ERR "UAV output dims unresolved: {%s}\n",
                   trt::format_shape(output_dims).c_str());
            return false;
        }

        OutputLayout layout = OutputLayout::kUnknown;
        int output_cdim = 0;
        int output_n_bboxes = 0;

        if (d0 >= 5 && d0 <= MAX_SUPPORTED_ATTR && d1 > d0) {
            layout = OutputLayout::kCXN;
            output_cdim = d0;
            output_n_bboxes = d1;
        } else if (d1 >= 5 && d1 <= MAX_SUPPORTED_ATTR && d0 > d1) {
            layout = OutputLayout::kNXC;
            output_cdim = d1;
            output_n_bboxes = d0;
        } else if (d0 == 5) {
            layout = OutputLayout::kCXN;
            output_cdim = d0;
            output_n_bboxes = d1;
        } else if (d1 == 5) {
            layout = OutputLayout::kNXC;
            output_cdim = d1;
            output_n_bboxes = d0;
        } else {
            printf(DBG_ERR "Cannot infer UAV output layout from dims {%s}\n",
                   trt::format_shape(output_dims).c_str());
            return false;
        }

        if (output_cdim < 5) {
            printf(DBG_ERR "UAV output attr dim must be >= 5, got %d\n", output_cdim);
            return false;
        }

        output_layout_ = layout;
        output_cdim_ = output_cdim;
        output_n_bboxes_ = output_n_bboxes;
        output_is_post_nms_xyxy_ =
            output_cdim == 6 && output_n_bboxes <= MAX_IMAGE_BOXES;
        return true;
    }

private:
    shared_ptr<trt::Infer> engine_;
    float conf_thr_ = 0.0f;
    float nms_thr_ = 0.0f;
    int input_binding_ = -1;
    int output_binding_ = -1;
    int network_input_channels_ = 3;
    int network_input_height_ = 0;
    int network_input_width_ = 0;
    int output_cdim_ = 0;
    int output_n_bboxes_ = 0;
    bool is_dynamic_model_ = false;
    bool input_is_fp16_ = false;
    bool output_is_fp16_ = false;
    bool output_is_post_nms_xyxy_ = false;
    OutputLayout output_layout_ = OutputLayout::kUnknown;
    trt::DType input_dtype_ = trt::DType::FLOAT;
    trt::DType output_dtype_ = trt::DType::FLOAT;

    Memory<float> output_boxarray_;
    Memory<float> affine_matrix_d2i_;
    Memory<float> input_buffer_gpu_;
    Memory<__half> input_buffer_half_gpu_;
    Memory<uint8_t> image_buffer_gpu_;
    Memory<float> output_buffer_float_gpu_;
    Memory<__half> output_buffer_half_gpu_;
};

std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> load(
    const string& file, float conf, float nms)
{
    auto impl = make_shared<InferImpl>();
    if (!impl->load(file, conf, nms)) {
        return nullptr;
    }
    return impl;
}

}  // namespace uav
