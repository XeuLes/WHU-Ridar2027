#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

struct Float2 { float x, y; };

__device__ __forceinline__ float cross2(const Float2 &a, const Float2 &b, const Float2 &p)
{
  // cross( (b-a), (p-a) )
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

__device__ __forceinline__ bool pointInConvexQuad(const Float2 pts[4], float x, float y)
{
  // 允许边界：>=0 或 <=0（统一符号）
  Float2 p{ x, y };
  float c0 = cross2(pts[0], pts[1], p);
  float c1 = cross2(pts[1], pts[2], p);
  float c2 = cross2(pts[2], pts[3], p);
  float c3 = cross2(pts[3], pts[0], p);

  bool all_pos = (c0 >= 0.f && c1 >= 0.f && c2 >= 0.f && c3 >= 0.f);
  bool all_neg = (c0 <= 0.f && c1 <= 0.f && c2 <= 0.f && c3 <= 0.f);
  return all_pos || all_neg;
}

__global__ void detect_color_kernel(
    const uchar3* __restrict__ bgr, int img_w, int img_h, int img_step_bytes,
    int roi_x, int roi_y, int roi_w, int roi_h,
    Float2 p0, Float2 p1, Float2 p2, Float2 p3,
    int sample_step,
    int* red_sum, int* blue_sum, int* count)
{
  // 每个 thread 对应一个采样点（按 sample_step 下采样）
  int sx = blockIdx.x * blockDim.x + threadIdx.x; // sample grid x
  int sy = blockIdx.y * blockDim.y + threadIdx.y; // sample grid y

  int j = sx * sample_step;
  int i = sy * sample_step;

  if (j >= roi_w || i >= roi_h) return;

  int abs_x = roi_x + j;
  int abs_y = roi_y + i;

  if (abs_x < 0 || abs_x >= img_w || abs_y < 0 || abs_y >= img_h) return;

  Float2 pts[4] = {p0,p1,p2,p3};
  if (!pointInConvexQuad(pts, (float)abs_x, (float)abs_y)) return;

  const uchar3* row = (const uchar3*)((const unsigned char*)bgr + abs_y * img_step_bytes);
  uchar3 pix = row[abs_x]; // pix.x=B pix.y=G pix.z=R

  atomicAdd(red_sum,  (int)pix.z);
  atomicAdd(blue_sum, (int)pix.x);
  atomicAdd(count, 1);
}

extern "C" bool detect_color_cuda(
    const cv::cuda::GpuMat& d_bgr,
    int roi_x, int roi_y, int roi_w, int roi_h,
    const Float2 rect_pts[4],
    int sample_step,
    int* h_red_sum, int* h_blue_sum, int* h_count,
    cudaStream_t stream)
{
  if (d_bgr.empty() || d_bgr.type() != CV_8UC3) return false;
  if (roi_w <= 0 || roi_h <= 0) return false;

  int *d_red=nullptr, *d_blue=nullptr, *d_cnt=nullptr;
  cudaMallocAsync(&d_red,  sizeof(int), stream);
  cudaMallocAsync(&d_blue, sizeof(int), stream);
  cudaMallocAsync(&d_cnt,  sizeof(int), stream);
  cudaMemsetAsync(d_red,  0, sizeof(int), stream);
  cudaMemsetAsync(d_blue, 0, sizeof(int), stream);
  cudaMemsetAsync(d_cnt,  0, sizeof(int), stream);

  dim3 block(16, 16);
  int grid_x = ( (roi_w + sample_step - 1) / sample_step + block.x - 1) / block.x;
  int grid_y = ( (roi_h + sample_step - 1) / sample_step + block.y - 1) / block.y;
  dim3 grid(grid_x, grid_y);

  detect_color_kernel<<<grid, block, 0, stream>>>(
      (const uchar3*)d_bgr.ptr<uchar3>(),
      d_bgr.cols, d_bgr.rows, (int)d_bgr.step,
      roi_x, roi_y, roi_w, roi_h,
      rect_pts[0], rect_pts[1], rect_pts[2], rect_pts[3],
      sample_step,
      d_red, d_blue, d_cnt);

  cudaMemcpyAsync(h_red_sum,  d_red,  sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_blue_sum, d_blue, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_count,    d_cnt,  sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaFreeAsync(d_red, stream);
  cudaFreeAsync(d_blue, stream);
  cudaFreeAsync(d_cnt, stream);

  return true;
}