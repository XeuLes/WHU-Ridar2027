#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "NvidiaInterface.hpp"
#include "yolo26.hpp"

namespace yolo {

static const int NUM_BOX_ELEMENT = 15;
static const int GPU_BLOCK_THREADS = 256;

static __host__ __device__ void affine_project(
    float* matrix, float x, float y, float* ox, float* oy
) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
) {
    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(
    float* bboxes,
    int max_boxes,
    float threshold
) {
    int position = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_boxes);
    if (position >= count) return;

    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;

    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || (int)pcurrent[5] != (int)pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );
            if (iou > threshold) {
                pcurrent[6] = 0;  // keepflag = 0
                return;
            }
        }
    }
}

// layout: [N, C]  (等价 [B, N, C])
static __global__ void decode_kernel_yolo26_nxc(
    float* predict,
    int num_bboxes,
    int num_classes,
    int output_cdim,
    float confidence_threshold,
    float* invert_affine_matrix,
    float* parray,
    int max_boxes
) {
    int position = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= num_bboxes) return;

    float* pitem = predict + output_cdim * position;

    // YOLO26 detection（按 V8 逻辑：无 objectness，直接取 max class score）
    float* class_conf = pitem + 4;
    float  best = *class_conf++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_conf) {
        if (*class_conf > best) { best = *class_conf; label = i; }
    }
    if (best < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= max_boxes) return;

    float cx = pitem[0];
    float cy = pitem[1];
    float w  = pitem[2];
    float h  = pitem[3];

    float left   = cx - w * 0.5f;
    float top    = cy - h * 0.5f;
    float right  = cx + w * 0.5f;
    float bottom = cy + h * 0.5f;

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* out = parray + 1 + index * NUM_BOX_ELEMENT;
    out[0] = left;
    out[1] = top;
    out[2] = right;
    out[3] = bottom;
    out[4] = best;
    out[5] = (float)label;
    out[6] = 1.0f;            // keepflag

    out[7] = (float)position; // ✅ 与 yolov8.cu 的 V8 decoder 一致：row_index 用 pbox[7]
    for (int k = 8; k < NUM_BOX_ELEMENT; ++k) out[k] = 0.0f;
}

// layout: [C, N]  (等价 [B, C, N])
static __global__ void decode_kernel_yolo26_cxn(
    float* predict,
    int num_bboxes,    // N
    int num_classes,
    int output_cdim,   // C
    float confidence_threshold,
    float* invert_affine_matrix,
    float* parray,
    int max_boxes
) {
    int position = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= num_bboxes) return;

    // channel-major: predict[c * N + position]
    float cx = predict[0 * num_bboxes + position];
    float cy = predict[1 * num_bboxes + position];
    float w  = predict[2 * num_bboxes + position];
    float h  = predict[3 * num_bboxes + position];

    float best = predict[4 * num_bboxes + position];
    int label = 0;
    for (int i = 1; i < num_classes; ++i) {
        float c = predict[(4 + i) * num_bboxes + position];
        if (c > best) { best = c; label = i; }
    }
    if (best < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= max_boxes) return;

    float left   = cx - w * 0.5f;
    float top    = cy - h * 0.5f;
    float right  = cx + w * 0.5f;
    float bottom = cy + h * 0.5f;

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* out = parray + 1 + index * NUM_BOX_ELEMENT;
    out[0] = left;
    out[1] = top;
    out[2] = right;
    out[3] = bottom;
    out[4] = best;
    out[5] = (float)label;
    out[6] = 1.0f;

    out[7] = (float)position; // ✅ row_index
    for (int k = 8; k < NUM_BOX_ELEMENT; ++k) out[k] = 0.0f;
}

static inline dim3 grid_dims(int jobs) {
    int t = jobs < GPU_BLOCK_THREADS ? jobs : GPU_BLOCK_THREADS;
    int g = (jobs + t - 1) / t;
    return dim3(g, 1, 1);
}
static inline dim3 block_dims(int jobs) {
    int t = jobs < GPU_BLOCK_THREADS ? jobs : GPU_BLOCK_THREADS;
    return dim3(t, 1, 1);
}

void decode_kernel_invoker_yolo26(
    float* predict,
    int dim1,
    int dim2,
    int num_classes,
    float confidence_threshold,
    float nms_threshold,
    float* invert_affine_matrix,
    float* parray,
    int max_image_boxes,
    bool layout_cxn,
    cudaStream_t stream
) {
    // layout_cxn:
    //  true  -> dims [C, N] => dim1=C, dim2=N
    //  false -> dims [N, C] => dim1=N, dim2=C
    int N = layout_cxn ? dim2 : dim1;
    int C = layout_cxn ? dim1 : dim2;

    auto grid = grid_dims(N);
    auto block = block_dims(N);

    if (layout_cxn) {
        checkKernel(decode_kernel_yolo26_cxn<<<grid, block, 0, stream>>>(
            predict, N, num_classes, C,
            confidence_threshold, invert_affine_matrix, parray, max_image_boxes
        ));
    } else {
        checkKernel(decode_kernel_yolo26_nxc<<<grid, block, 0, stream>>>(
            predict, N, num_classes, C,
            confidence_threshold, invert_affine_matrix, parray, max_image_boxes
        ));
    }

    // NMS
    grid = grid_dims(max_image_boxes);
    block = block_dims(max_image_boxes);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(
        parray, max_image_boxes, nms_threshold
    ));
}

}  // namespace yolo
