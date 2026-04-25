#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

namespace {

__device__ __forceinline__ unsigned char read_bgr(
  const uchar3* img, int w, int h, int step_bytes,
  float x, float y, int c)
{
  x = fminf(fmaxf(x, 0.0f), static_cast<float>(w - 1));
  y = fminf(fmaxf(y, 0.0f), static_cast<float>(h - 1));

  const int x0 = static_cast<int>(floorf(x));
  const int y0 = static_cast<int>(floorf(y));
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const uchar3* row0 = reinterpret_cast<const uchar3*>(
    reinterpret_cast<const unsigned char*>(img) + y0 * step_bytes);
  const uchar3* row1 = reinterpret_cast<const uchar3*>(
    reinterpret_cast<const unsigned char*>(img) + y1 * step_bytes);

  const uchar3 p00 = row0[x0];
  const uchar3 p01 = row0[x1];
  const uchar3 p10 = row1[x0];
  const uchar3 p11 = row1[x1];

  const float v00 = (c == 0 ? p00.x : (c == 1 ? p00.y : p00.z));
  const float v01 = (c == 0 ? p01.x : (c == 1 ? p01.y : p01.z));
  const float v10 = (c == 0 ? p10.x : (c == 1 ? p10.y : p10.z));
  const float v11 = (c == 0 ? p11.x : (c == 1 ? p11.y : p11.z));

  const float v0 = v00 + dx * (v01 - v00);
  const float v1 = v10 + dx * (v11 - v10);
  return static_cast<unsigned char>(v0 + dy * (v1 - v0));
}

__global__ void yolo_letterbox_bgr8_to_half_kernel(
  const uchar3* src,
  int src_w,
  int src_h,
  int src_step_bytes,
  half* dst,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  const float fill = 114.0f / 255.0f;
  const int area = dst_w * dst_h;

  float out_r = fill;
  float out_g = fill;
  float out_b = fill;

  const float src_x = (static_cast<float>(x) - pad_x) / scale;
  const float src_y = (static_cast<float>(y) - pad_y) / scale;

  if (src_x >= 0.0f && src_x <= static_cast<float>(src_w - 1) &&
      src_y >= 0.0f && src_y <= static_cast<float>(src_h - 1)) {
    out_b = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 0)) / 255.0f;
    out_g = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 1)) / 255.0f;
    out_r = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 2)) / 255.0f;
  }

  dst[0 * area + y * dst_w + x] = __float2half(swap_rb ? out_b : out_r);
  dst[1 * area + y * dst_w + x] = __float2half(out_g);
  dst[2 * area + y * dst_w + x] = __float2half(swap_rb ? out_r : out_b);
}

__global__ void yolo_letterbox_bgr8_to_float_kernel(
  const uchar3* src,
  int src_w,
  int src_h,
  int src_step_bytes,
  float* dst,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  const float fill = 114.0f / 255.0f;
  const int area = dst_w * dst_h;

  float out_r = fill;
  float out_g = fill;
  float out_b = fill;

  const float src_x = (static_cast<float>(x) - pad_x) / scale;
  const float src_y = (static_cast<float>(y) - pad_y) / scale;

  if (src_x >= 0.0f && src_x <= static_cast<float>(src_w - 1) &&
      src_y >= 0.0f && src_y <= static_cast<float>(src_h - 1)) {
    out_b = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 0)) / 255.0f;
    out_g = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 1)) / 255.0f;
    out_r = static_cast<float>(read_bgr(src, src_w, src_h, src_step_bytes, src_x, src_y, 2)) / 255.0f;
  }

  dst[0 * area + y * dst_w + x] = swap_rb ? out_b : out_r;
  dst[1 * area + y * dst_w + x] = out_g;
  dst[2 * area + y * dst_w + x] = swap_rb ? out_r : out_b;
}

}  // namespace

extern "C" bool yolo_letterbox_bgr8_to_half(
  const cv::cuda::GpuMat& d_bgr,
  void* d_out,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb,
  cudaStream_t stream)
{
  if (d_bgr.empty() || d_bgr.type() != CV_8UC3 || d_out == nullptr) return false;
  if (dst_w <= 0 || dst_h <= 0) return false;
  if (scale <= 0.0f) return false;

  dim3 block(16, 16);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y);

  yolo_letterbox_bgr8_to_half_kernel<<<grid, block, 0, stream>>>(
    reinterpret_cast<const uchar3*>(d_bgr.ptr<uchar3>()),
    d_bgr.cols,
    d_bgr.rows,
    static_cast<int>(d_bgr.step),
    reinterpret_cast<half*>(d_out),
    dst_w,
    dst_h,
    scale,
    pad_x,
    pad_y,
    swap_rb);

  return cudaGetLastError() == cudaSuccess;
}

extern "C" bool yolo_letterbox_bgr8_to_float(
  const cv::cuda::GpuMat& d_bgr,
  void* d_out,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb,
  cudaStream_t stream)
{
  if (d_bgr.empty() || d_bgr.type() != CV_8UC3 || d_out == nullptr) return false;
  if (dst_w <= 0 || dst_h <= 0) return false;
  if (scale <= 0.0f) return false;

  dim3 block(16, 16);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y);

  yolo_letterbox_bgr8_to_float_kernel<<<grid, block, 0, stream>>>(
    reinterpret_cast<const uchar3*>(d_bgr.ptr<uchar3>()),
    d_bgr.cols,
    d_bgr.rows,
    static_cast<int>(d_bgr.step),
    reinterpret_cast<float*>(d_out),
    dst_w,
    dst_h,
    scale,
    pad_x,
    pad_y,
    swap_rb);

  return cudaGetLastError() == cudaSuccess;
}
