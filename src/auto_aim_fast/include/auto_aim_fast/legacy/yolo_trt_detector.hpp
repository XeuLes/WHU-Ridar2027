#pragma once

#include "auto_aim_fast/legacy/detector.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace auto_aim_fast
{

class YoloPoseTrtDetector
{
public:
  struct Params
  {
    std::string engine_path;
    int input_w = 640;
    int input_h = 640;
    int max_det = 300;
    int target_class_id = 0;
    float conf_thresh = 0.5f;
    float nms_thresh = 0.45f;
    float kpt_conf_thresh = 0.5f;
    float min_w_ratio = 0.045f;
    float min_h_ratio = 0.055f;
    float min_area_ratio = 0.0025f;
    bool require_outer_corners = true;
    bool use_fp16_input = true;
    bool swap_rb_input = false;
    std::array<int, 4> outer_corner_indices = {0, 4, 7, 3}; // lt, lb, rb, rt from 8 kpts
  };

  YoloPoseTrtDetector();
  ~YoloPoseTrtDetector();

  bool init(const Params& params);
  void reset();
  bool isReady() const { return ready_; }

  void setParams(const Params& params) { params_ = params; }
  const Params& getParams() const { return params_; }

  std::vector<SimpleLightDetector::PairResult> detectDetailed(const cv::Mat& bgr);

private:
  struct TrtDeleter
  {
    template <typename T>
    void operator()(T* p) const
    {
      if (p) {
        delete p;
      }
    }
  };

  struct Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char* msg) noexcept override;
  };

  enum class OutputFormat
  {
    POST_NMS_XYXY = 0,
    RAW_ULTRALYTICS_POSE = 1,
    RAW_ULTRALYTICS_DETECT = 2
  };

  struct LetterboxMeta
  {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int src_w = 0;
    int src_h = 0;
  };

  bool loadEngine();
  bool allocateBuffers();
  void releaseBuffers();
  void prepareLetterboxMeta(int src_w, int src_h, LetterboxMeta& meta) const;

  std::vector<SimpleLightDetector::PairResult> parseOutput(
    const std::vector<float>& host_output,
    const LetterboxMeta& meta) const;

  cv::Point2f undoLetterbox(float x, float y, const LetterboxMeta& meta) const;

private:
  Logger logger_;
  Params params_{};
  bool ready_ = false;

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

  std::string input_name_;
  std::string output_name_;

  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  int output_det_count_ = 0;
  int output_det_stride_ = 0;
  int output_class_count_ = 1;
  OutputFormat output_format_ = OutputFormat::POST_NMS_XYXY;

  void* d_input_ = nullptr;
  void* d_output_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cv::cuda::GpuMat d_bgr_;
};

}  // namespace auto_aim_fast

extern "C" bool yolo_letterbox_bgr8_to_half(
  const cv::cuda::GpuMat& d_bgr,
  void* d_out,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb,
  cudaStream_t stream);

extern "C" bool yolo_letterbox_bgr8_to_float(
  const cv::cuda::GpuMat& d_bgr,
  void* d_out,
  int dst_w,
  int dst_h,
  float scale,
  float pad_x,
  float pad_y,
  bool swap_rb,
  cudaStream_t stream);
