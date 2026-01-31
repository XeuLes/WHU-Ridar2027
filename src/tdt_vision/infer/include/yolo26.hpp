#pragma once
#include <cuda_runtime.h>

namespace yolo {

/**
 * YOLO26 detection decode + NMS
 *
 * 支持两种输出布局（单 batch 视角）：
 *  1) layout_cxn = false : [N, C]  (等价于 [B, N, C])
 *  2) layout_cxn = true  : [C, N]  (等价于 [B, C, N])
 *
 * C 要求为 4 + num_classes（与当前 yolov8.cu 的 V8 decoder 一致：无 objectness）
 *
 * 输出 parray 的布局与工程一致（NUM_BOX_ELEMENT=15）：
 *  [0]=left [1]=top [2]=right [3]=bottom [4]=conf [5]=label [6]=keepflag
 *  [7]=row_index(=position) 其余填 0，保持与 V8Seg 用 row_index 的方式兼容
 */
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
);

}  // namespace yolo
