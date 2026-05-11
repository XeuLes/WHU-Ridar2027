#include "auto_aim_fast/legacy/yolo_trt_detector.hpp"

#include <cuda_fp16.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

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

namespace auto_aim_fast
{

namespace
{
constexpr int kExpectedKptNum = 8;
constexpr int kExpectedKptDim = 3;
constexpr int kPerDetBase = 6;
constexpr int kExpectedStride = kPerDetBase + kExpectedKptNum * kExpectedKptDim;  // 30
constexpr int kRawBoxFields = 4;
constexpr int kRawKptFields = kExpectedKptNum * kExpectedKptDim;

float clampf(float v, float lo, float hi)
{
  return std::max(lo, std::min(hi, v));
}

cv::Point2f meanPoints(const std::vector<cv::Point2f>& pts)
{
  if (pts.empty()) return {0.f, 0.f};
  cv::Point2f acc(0.f, 0.f);
  for (const auto& p : pts) acc += p;
  return acc * (1.f / static_cast<float>(pts.size()));
}

const char* dtypeToStr(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return "FLOAT";
    case nvinfer1::DataType::kHALF:  return "HALF";
    case nvinfer1::DataType::kINT8:  return "INT8";
    case nvinfer1::DataType::kINT32: return "INT32";
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
    case nvinfer1::DataType::kBOOL:  return "BOOL";
    case nvinfer1::DataType::kUINT8: return "UINT8";
#endif
    default: return "UNKNOWN";
  }
}

std::string dimsToString(const nvinfer1::Dims& d)
{
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < d.nbDims; ++i) {
    oss << d.d[i];
    if (i + 1 < d.nbDims) oss << ", ";
  }
  oss << "]";
  return oss.str();
}

size_t dataTypeBytes(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
    case nvinfer1::DataType::kBOOL:  return 1;
    case nvinfer1::DataType::kUINT8: return 1;
#endif
    default: return 0;
  }
}

bool isLikelyPostNmsStride(int x)
{
  return x >= kExpectedStride && x <= 256;
}

bool isLikelyPostNmsDetectStride(int x)
{
  return x == 6;
}

int rawPoseClassCount(int stride)
{
  return stride - kRawBoxFields - kRawKptFields;
}

int rawDetectClassCount(int stride)
{
  return stride - kRawBoxFields;
}

bool isLikelyRawPoseStride(int x)
{
  const int class_count = rawPoseClassCount(x);
  return class_count >= 1 && class_count <= 128;
}

bool isLikelyRawDetectStride(int x)
{
  const int class_count = rawDetectClassCount(x);
  return class_count >= 1 && class_count <= 128;
}

bool isLikelyOutputStride(int x)
{
  return isLikelyPostNmsDetectStride(x) || isLikelyPostNmsStride(x) ||
    isLikelyRawPoseStride(x) || isLikelyRawDetectStride(x);
}

const char * outputFormatToString(YoloPoseTrtDetector::OutputFormat format)
{
  switch (format) {
    case YoloPoseTrtDetector::OutputFormat::RAW_ULTRALYTICS_POSE:
      return "raw-ultralytics-pose";
    case YoloPoseTrtDetector::OutputFormat::RAW_ULTRALYTICS_DETECT:
      return "raw-ultralytics-detect";
    case YoloPoseTrtDetector::OutputFormat::POST_NMS_DETECT_XYXY:
      return "post-nms-detect-xyxy";
    case YoloPoseTrtDetector::OutputFormat::POST_NMS_XYXY:
    default:
      return "post-nms-xyxy";
  }
}

float boxIou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
  const float ix1 = std::max(a[0], b[0]);
  const float iy1 = std::max(a[1], b[1]);
  const float ix2 = std::min(a[2], b[2]);
  const float iy2 = std::min(a[3], b[3]);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;

  const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
  const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
  const float denom = area_a + area_b - inter;
  if (denom <= 1e-6f) return 0.0f;
  return inter / denom;
}

}  // namespace

void YoloPoseTrtDetector::Logger::log(Severity severity, const char* msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::fprintf(stderr, "[YoloPoseTrtDetector][TRT] %s\n", msg);
  }
}

YoloPoseTrtDetector::YoloPoseTrtDetector() = default;

YoloPoseTrtDetector::~YoloPoseTrtDetector()
{
  reset();
}

bool YoloPoseTrtDetector::init(const Params& params)
{
  reset();
  params_ = params;

  if (params_.engine_path.empty()) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] engine_path is empty\n");
    return false;
  }

  std::fprintf(stderr,
               "[YoloPoseTrtDetector] init engine=%s input=(%d,%d) conf=%.3f nms=%.3f kpt_conf=%.3f min_ratio=(%.4f,%.4f,%.5f) class_id=%d max_det=%d swap_rb=%d\n",
               params_.engine_path.c_str(),
               params_.input_w, params_.input_h,
               params_.conf_thresh, params_.nms_thresh,
               params_.kpt_conf_thresh,
               params_.min_w_ratio, params_.min_h_ratio, params_.min_area_ratio,
               params_.target_class_id, params_.max_det,
               params_.swap_rb_input ? 1 : 0);

  if (!loadEngine()) return false;
  if (!allocateBuffers()) return false;

  ready_ = true;
  std::fprintf(stderr, "[YoloPoseTrtDetector] init success\n");
  return true;
}

void YoloPoseTrtDetector::reset()
{
  ready_ = false;
  releaseBuffers();
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

bool YoloPoseTrtDetector::loadEngine()
{
  std::ifstream ifs(params_.engine_path, std::ios::binary);
  if (!ifs) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] failed to open engine: %s\n", params_.engine_path.c_str());
    return false;
  }

  std::vector<char> engine_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  if (engine_data.empty()) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] engine file is empty\n");
    return false;
  }

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] createInferRuntime failed\n");
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] deserializeCudaEngine failed\n");
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] createExecutionContext failed\n");
    return false;
  }

  const int nb_io = engine_->getNbIOTensors();
  std::fprintf(stderr, "[YoloPoseTrtDetector] engine nb_io=%d\n", nb_io);

  for (int i = 0; i < nb_io; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (!name) continue;

    const auto mode = engine_->getTensorIOMode(name);
    const auto dtype = engine_->getTensorDataType(name);
    const auto shape = engine_->getTensorShape(name);

    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] tensor[%d] name=%s mode=%s dtype=%s shape=%s\n",
                 i,
                 name,
                 (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT"),
                 dtypeToStr(dtype),
                 dimsToString(shape).c_str());

    if (mode == nvinfer1::TensorIOMode::kINPUT) input_name_ = name;
    if (mode == nvinfer1::TensorIOMode::kOUTPUT) output_name_ = name;
  }

  if (input_name_.empty() || output_name_.empty()) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] failed to resolve input/output tensor names\n");
    return false;
  }

  const auto input_dtype = engine_->getTensorDataType(input_name_.c_str());
  const auto output_dtype = engine_->getTensorDataType(output_name_.c_str());

  std::fprintf(stderr,
               "[YoloPoseTrtDetector] selected input=%s dtype=%s\n",
               input_name_.c_str(), dtypeToStr(input_dtype));
  std::fprintf(stderr,
               "[YoloPoseTrtDetector] selected output=%s dtype=%s\n",
               output_name_.c_str(), dtypeToStr(output_dtype));

  if (input_dtype == nvinfer1::DataType::kHALF) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] preprocess path: BGR8 -> HALF CHW\n");
  } else if (input_dtype == nvinfer1::DataType::kFLOAT) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] preprocess path: BGR8 -> FLOAT CHW\n");
  } else {
    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] unsupported input dtype=%s\n",
                 dtypeToStr(input_dtype));
    return false;
  }

  nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
  if (input_dims.nbDims == 4) {
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = params_.input_h;
    input_dims.d[3] = params_.input_w;
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
      std::fprintf(stderr, "[YoloPoseTrtDetector] setInputShape failed for %s\n", input_name_.c_str());
      return false;
    }
    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] runtime input shape set to %s\n",
                 dimsToString(context_->getTensorShape(input_name_.c_str())).c_str());
  }

  const auto out_dims = context_->getTensorShape(output_name_.c_str());
  std::fprintf(stderr,
               "[YoloPoseTrtDetector] runtime output shape=%s\n",
               dimsToString(out_dims).c_str());

  if (out_dims.nbDims != 3) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] expect output dims=3, got %d\n", out_dims.nbDims);
    return false;
  }

  const int d1 = out_dims.d[1];
  const int d2 = out_dims.d[2];

  bool transposed = false;
  int det_count = 0;
  int det_stride = 0;

  if (isLikelyOutputStride(d2) && d1 > 0) {
    det_count = d1;
    det_stride = d2;
    transposed = false;
  } else if (isLikelyOutputStride(d1) && d2 > 0) {
    det_count = d2;
    det_stride = d1;
    transposed = true;
  } else {
    det_count = std::max(d1, d2);
    det_stride = std::min(d1, d2);
    transposed = (d1 == det_stride);
    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] WARNING: output shape ambiguous, fallback det_count=%d det_stride=%d transposed=%d\n",
                 det_count, det_stride, static_cast<int>(transposed));
  }

  if (det_count <= 0) det_count = params_.max_det;
  if (det_stride <= 0) det_stride = kExpectedStride;

  if (det_stride < kRawBoxFields) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] invalid output stride=%d\n", det_stride);
    return false;
  }

  const int raw_class_count = rawPoseClassCount(det_stride);
  const int raw_detect_class_count = rawDetectClassCount(det_stride);
  const bool raw_possible = isLikelyRawPoseStride(det_stride);
  const bool raw_detect_possible = isLikelyRawDetectStride(det_stride);
  const bool post_nms_possible = isLikelyPostNmsStride(det_stride);
  const bool post_nms_detect_possible = isLikelyPostNmsDetectStride(det_stride);

  constexpr int kRawAnchorCountHint = 1024;
  if (post_nms_detect_possible && det_count <= std::max(1, params_.max_det)) {
    output_format_ = OutputFormat::POST_NMS_DETECT_XYXY;
    output_class_count_ = 1;
  } else if (raw_possible && (det_count > std::max(1, params_.max_det) ||
                       det_count > kRawAnchorCountHint ||
                       !post_nms_possible)) {
    output_format_ = OutputFormat::RAW_ULTRALYTICS_POSE;
    output_class_count_ = raw_class_count;
  } else if (raw_detect_possible && (det_count > std::max(1, params_.max_det) ||
                                     det_count > kRawAnchorCountHint ||
                                     (!post_nms_possible && !raw_possible))) {
    output_format_ = OutputFormat::RAW_ULTRALYTICS_DETECT;
    output_class_count_ = raw_detect_class_count;
  } else if (post_nms_possible) {
    output_format_ = OutputFormat::POST_NMS_XYXY;
    output_class_count_ = std::max(1, raw_class_count);
  } else if (raw_possible) {
    output_format_ = OutputFormat::RAW_ULTRALYTICS_POSE;
    output_class_count_ = raw_class_count;
  } else if (raw_detect_possible) {
    output_format_ = OutputFormat::RAW_ULTRALYTICS_DETECT;
    output_class_count_ = raw_detect_class_count;
  } else {
    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] unsupported output stride=%d for YOLO model\n",
                 det_stride);
    return false;
  }

  output_det_count_ = det_count;
  output_det_stride_ = transposed ? -det_stride : det_stride;

  std::fprintf(stderr,
               "[YoloPoseTrtDetector] parsed layout=%s format=%s det_count=%d det_stride=%d classes=%d\n",
               transposed ? "[1,C,N]" : "[1,N,C]",
               outputFormatToString(output_format_),
               output_det_count_,
               std::abs(output_det_stride_),
               output_class_count_);

  return true;
}

bool YoloPoseTrtDetector::allocateBuffers()
{
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] cudaStreamCreate failed\n");
    return false;
  }

  const auto input_dtype = engine_->getTensorDataType(input_name_.c_str());
  const auto output_dtype = engine_->getTensorDataType(output_name_.c_str());

  const size_t in_elem_size = dataTypeBytes(input_dtype);
  const size_t out_elem_size = dataTypeBytes(output_dtype);

  if (in_elem_size == 0 || out_elem_size == 0) {
    std::fprintf(stderr,
                 "[YoloPoseTrtDetector] unsupported tensor dtype input=%s output=%s\n",
                 dtypeToStr(input_dtype), dtypeToStr(output_dtype));
    return false;
  }

  input_bytes_ =
    static_cast<size_t>(1) * 3 * params_.input_h * params_.input_w * in_elem_size;

  output_bytes_ =
    static_cast<size_t>(1) * output_det_count_ * std::abs(output_det_stride_) * out_elem_size;

  std::fprintf(stderr,
               "[YoloPoseTrtDetector] allocate input_bytes=%zu output_bytes=%zu\n",
               input_bytes_, output_bytes_);

  if (cudaMalloc(&d_input_, input_bytes_) != cudaSuccess) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] cudaMalloc d_input failed\n");
    return false;
  }
  if (cudaMalloc(&d_output_, output_bytes_) != cudaSuccess) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] cudaMalloc d_output failed\n");
    return false;
  }

  if (!context_->setTensorAddress(input_name_.c_str(), d_input_)) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] setTensorAddress input failed\n");
    return false;
  }
  if (!context_->setTensorAddress(output_name_.c_str(), d_output_)) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] setTensorAddress output failed\n");
    return false;
  }

  return true;
}

void YoloPoseTrtDetector::releaseBuffers()
{
  if (d_input_) {
    cudaFree(d_input_);
    d_input_ = nullptr;
  }
  if (d_output_) {
    cudaFree(d_output_);
    d_output_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void YoloPoseTrtDetector::prepareLetterboxMeta(int src_w, int src_h, LetterboxMeta& meta) const
{
  meta.src_w = src_w;
  meta.src_h = src_h;

  const float scale_x = static_cast<float>(params_.input_w) / static_cast<float>(src_w);
  const float scale_y = static_cast<float>(params_.input_h) / static_cast<float>(src_h);
  meta.scale = std::min(scale_x, scale_y);

  const float resized_w = static_cast<float>(src_w) * meta.scale;
  const float resized_h = static_cast<float>(src_h) * meta.scale;

  meta.pad_x = 0.5f * (static_cast<float>(params_.input_w) - resized_w);
  meta.pad_y = 0.5f * (static_cast<float>(params_.input_h) - resized_h);
}

cv::Point2f YoloPoseTrtDetector::undoLetterbox(float x, float y, const LetterboxMeta& meta) const
{
  const float rx = (x - meta.pad_x) / std::max(1e-6f, meta.scale);
  const float ry = (y - meta.pad_y) / std::max(1e-6f, meta.scale);
  return {
    clampf(rx, 0.f, static_cast<float>(meta.src_w - 1)),
    clampf(ry, 0.f, static_cast<float>(meta.src_h - 1))
  };
}

std::vector<SimpleLightDetector::PairResult> YoloPoseTrtDetector::detectDetailed(const cv::Mat& bgr)
{
  std::vector<SimpleLightDetector::PairResult> out;
  if (!ready_ || bgr.empty() || bgr.type() != CV_8UC3) return out;

  static int s_debug_frame = 0;
  ++s_debug_frame;

  LetterboxMeta meta;
  prepareLetterboxMeta(bgr.cols, bgr.rows, meta);

  cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream_);
  d_bgr_.upload(bgr, cv_stream);

  const auto input_dtype = engine_->getTensorDataType(input_name_.c_str());

  bool preprocess_ok = false;
  if (input_dtype == nvinfer1::DataType::kHALF) {
    preprocess_ok = yolo_letterbox_bgr8_to_half(
      d_bgr_, d_input_, params_.input_w, params_.input_h,
      meta.scale, meta.pad_x, meta.pad_y, params_.swap_rb_input, stream_);
  } else if (input_dtype == nvinfer1::DataType::kFLOAT) {
    preprocess_ok = yolo_letterbox_bgr8_to_float(
      d_bgr_, d_input_, params_.input_w, params_.input_h,
      meta.scale, meta.pad_x, meta.pad_y, params_.swap_rb_input, stream_);
  } else {
    std::fprintf(stderr, "[YoloPoseTrtDetector] unsupported input dtype for preprocess\n");
    return out;
  }

  if (!preprocess_ok) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] preprocess kernel failed\n");
    return out;
  }

  if (!context_->enqueueV3(stream_)) {
    std::fprintf(stderr, "[YoloPoseTrtDetector] enqueueV3 failed\n");
    return out;
  }

  const auto output_dtype = engine_->getTensorDataType(output_name_.c_str());
  const size_t elem_count = static_cast<size_t>(output_det_count_) * std::abs(output_det_stride_);
  std::vector<float> host_output(elem_count, 0.0f);

  if (output_dtype == nvinfer1::DataType::kFLOAT) {
    if (cudaMemcpyAsync(host_output.data(), d_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      std::fprintf(stderr, "[YoloPoseTrtDetector] cudaMemcpyAsync FLOAT output failed\n");
      return out;
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      std::fprintf(stderr, "[YoloPoseTrtDetector] cudaStreamSynchronize failed\n");
      return out;
    }
  } else if (output_dtype == nvinfer1::DataType::kHALF) {
    std::vector<__half> host_half(elem_count);
    if (cudaMemcpyAsync(host_half.data(), d_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      std::fprintf(stderr, "[YoloPoseTrtDetector] cudaMemcpyAsync HALF output failed\n");
      return out;
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      std::fprintf(stderr, "[YoloPoseTrtDetector] cudaStreamSynchronize failed\n");
      return out;
    }
    for (size_t i = 0; i < elem_count; ++i) {
      host_output[i] = __half2float(host_half[i]);
    }
  } else {
    std::fprintf(stderr, "[YoloPoseTrtDetector] unsupported output dtype\n");
    return out;
  }

  if (s_debug_frame <= 10) {
    const int stride_abs = std::abs(output_det_stride_);
    const bool transposed = output_det_stride_ < 0;

    float raw_max = -std::numeric_limits<float>::infinity();
    for (float v : host_output) raw_max = std::max(raw_max, v);

    std::fprintf(stderr,
                 "[YoloPoseTrtDetector][DBG] frame=%d src=%dx%d scale=%.6f pad=(%.3f, %.3f) layout=%s format=%s det_count=%d stride=%d output_dtype=%s raw_max=%.6f\n",
                 s_debug_frame,
                 bgr.cols, bgr.rows,
                 meta.scale, meta.pad_x, meta.pad_y,
                 transposed ? "[1,C,N]" : "[1,N,C]",
                 outputFormatToString(output_format_),
                 output_det_count_, stride_abs,
                 dtypeToStr(output_dtype),
                 raw_max);

    const int print_n = std::min<int>(16, static_cast<int>(host_output.size()));
    std::fprintf(stderr, "[YoloPoseTrtDetector][DBG] raw first %d vals:", print_n);
    for (int i = 0; i < print_n; ++i) {
      std::fprintf(stderr, " %.6f", host_output[i]);
    }
    std::fprintf(stderr, "\n");

    auto getDetValDbg = [&](int det_i, int field_j) -> float {
      if (!transposed) {
        return host_output[static_cast<size_t>(det_i) * stride_abs + field_j];
      } else {
        return host_output[static_cast<size_t>(field_j) * output_det_count_ + det_i];
      }
    };

    const int det_print = std::min(output_det_count_, 3);
    for (int i = 0; i < det_print; ++i) {
      if (output_format_ == OutputFormat::RAW_ULTRALYTICS_POSE ||
          output_format_ == OutputFormat::RAW_ULTRALYTICS_DETECT) {
        std::fprintf(stderr,
                     "[YoloPoseTrtDetector][DBG] raw det[%d] cx=%.3f cy=%.3f w=%.3f h=%.3f class0=%.6f\n",
                     i,
                     getDetValDbg(i, 0), getDetValDbg(i, 1),
                     getDetValDbg(i, 2), getDetValDbg(i, 3),
                     getDetValDbg(i, 4));
      } else if (output_format_ == OutputFormat::POST_NMS_XYXY ||
                 output_format_ == OutputFormat::POST_NMS_DETECT_XYXY) {
        std::fprintf(stderr,
                     "[YoloPoseTrtDetector][DBG] det[%d] x1=%.3f y1=%.3f x2=%.3f y2=%.3f score=%.6f cls=%.3f\n",
                     i,
                     getDetValDbg(i, 0), getDetValDbg(i, 1),
                     getDetValDbg(i, 2), getDetValDbg(i, 3),
                     getDetValDbg(i, 4), getDetValDbg(i, 5));
      }
    }
  }

  return parseOutput(host_output, meta);
}

std::vector<SimpleLightDetector::PairResult> YoloPoseTrtDetector::parseOutput(
  const std::vector<float>& host_output,
  const LetterboxMeta& meta) const
{
  std::vector<SimpleLightDetector::PairResult> out;

  const bool transposed = output_det_stride_ < 0;
  const int stride_abs = std::abs(output_det_stride_);

  auto getDetVal = [&](int det_i, int field_j) -> float {
    if (!transposed) {
      return host_output[static_cast<size_t>(det_i) * stride_abs + field_j];
    } else {
      return host_output[static_cast<size_t>(field_j) * output_det_count_ + det_i];
    }
  };

  struct DecodedDetection
  {
    std::array<float, 4> box_net{};  // x1, y1, x2, y2 in network input coordinates
    float score = 0.0f;
    int cls = 0;
    bool has_keypoints = false;
    std::array<cv::Point2f, kExpectedKptNum> kpts_net{};
    std::array<float, kExpectedKptNum> kpt_scores{};
  };

  auto makePair = [&](const DecodedDetection& det, int det_i) -> SimpleLightDetector::PairResult {
    SimpleLightDetector::PairResult pr;
    pr.pair_type = 200;
    pr.confidence = det.score;

    const cv::Point2f lt = undoLetterbox(det.box_net[0], det.box_net[1], meta);
    const cv::Point2f rb = undoLetterbox(det.box_net[2], det.box_net[3], meta);

    pr.center.x = 0.5f * (lt.x + rb.x);
    pr.center.y = 0.5f * (lt.y + rb.y);
    pr.width = std::max(1.f, rb.x - lt.x);
    pr.height = std::max(1.f, rb.y - lt.y);
    pr.angle = 0.0f;

    pr.has_keypoints = false;
    pr.has_precise_corners = false;

    int valid_kpt = 0;
    std::vector<cv::Point2f> top_pts;
    std::vector<cv::Point2f> bottom_pts;

    if (det.has_keypoints) {
      for (int k = 0; k < kExpectedKptNum; ++k) {
        const cv::Point2f p = undoLetterbox(det.kpts_net[k].x, det.kpts_net[k].y, meta);
        const float kv = det.kpt_scores[k];

        pr.keypoints[k] = p;
        pr.keypoint_scores[k] = kv;

        if (kv >= params_.kpt_conf_thresh) {
          ++valid_kpt;
          if (k < 4) top_pts.push_back(p);
          else bottom_pts.push_back(p);
        }
      }
    }

    pr.has_keypoints = (valid_kpt > 0);

    const auto& idx = params_.outer_corner_indices;
    bool outer_idx_valid = true;
    bool outer_score_valid = true;
    for (int id : idx) {
      if (id < 0 || id >= kExpectedKptNum) {
        outer_idx_valid = false;
        outer_score_valid = false;
        break;
      }
      if (pr.keypoint_scores[static_cast<size_t>(id)] < params_.kpt_conf_thresh) {
        outer_score_valid = false;
      }
    }

    if (det.has_keypoints && outer_idx_valid && outer_score_valid) {
      pr.precise_left_top = pr.keypoints[static_cast<size_t>(idx[0])];
      pr.precise_left_bottom = pr.keypoints[static_cast<size_t>(idx[1])];
      pr.precise_right_bottom = pr.keypoints[static_cast<size_t>(idx[2])];
      pr.precise_right_top = pr.keypoints[static_cast<size_t>(idx[3])];
      pr.has_precise_corners = true;

      pr.center = (pr.precise_left_top + pr.precise_left_bottom +
                   pr.precise_right_bottom + pr.precise_right_top) * 0.25f;
    } else if (det.has_keypoints && params_.require_outer_corners) {
      if (det_i >= 0) {
        static int s_outer_reject_dbg = 0;
        if (s_outer_reject_dbg < 20) {
          ++s_outer_reject_dbg;
          std::fprintf(stderr,
                       "[YoloPoseTrtDetector][DBG] reject det[%d]: outer corners invalid idx=%d score=%d\n",
                       det_i, outer_idx_valid ? 1 : 0, outer_score_valid ? 1 : 0);
        }
      }
      pr.confidence = -1.0f;
      return pr;
    }

    if (!top_pts.empty()) {
      pr.light1_center = meanPoints(top_pts);
    } else {
      pr.light1_center = cv::Point2f(pr.center.x, pr.center.y - 0.25f * pr.height);
    }

    if (!bottom_pts.empty()) {
      pr.light2_center = meanPoints(bottom_pts);
    } else {
      pr.light2_center = cv::Point2f(pr.center.x, pr.center.y + 0.25f * pr.height);
    }

    return pr;
  };

  float max_score_seen = -1.0f;
  static int s_accept_dbg = 0;
  static int s_reject_dbg = 0;
  // static int s_size_reject_dbg = 0;

  auto passesSizeGate = [&](const DecodedDetection& det, int det_i) -> bool {
    (void)det_i;
    const cv::Point2f p1 = undoLetterbox(det.box_net[0], det.box_net[1], meta);
    const cv::Point2f p2 = undoLetterbox(det.box_net[2], det.box_net[3], meta);

    const float w = std::abs(p2.x - p1.x);
    const float h = std::abs(p2.y - p1.y);
    const float area = w * h;

    const float min_w = std::max(0.0f, params_.min_w_ratio) * static_cast<float>(meta.src_w);
    const float min_h = std::max(0.0f, params_.min_h_ratio) * static_cast<float>(meta.src_h);
    const float min_area =
      std::max(0.0f, params_.min_area_ratio) *
      static_cast<float>(meta.src_w) * static_cast<float>(meta.src_h);

    const bool ok = w >= min_w && h >= min_h && area >= min_area;
    // if (!ok && s_size_reject_dbg < 30) {
    //   ++s_size_reject_dbg;
    //   std::fprintf(stderr,
    //                "[YoloPoseTrtDetector][DBG] reject det[%d]: size gate wh=(%.1f,%.1f) area=%.1f min=(%.1f,%.1f,%.1f)\n",
    //                det_i, w, h, area, min_w, min_h, min_area);
    // }
    return ok;
  };

  if (output_format_ == OutputFormat::RAW_ULTRALYTICS_POSE ||
      output_format_ == OutputFormat::RAW_ULTRALYTICS_DETECT) {
    const bool has_pose = output_format_ == OutputFormat::RAW_ULTRALYTICS_POSE;
    const int class_count = output_class_count_;
    const int kpt_offset = kRawBoxFields + class_count;

    if (has_pose && stride_abs < kpt_offset + kRawKptFields) {
      std::fprintf(stderr,
                   "[YoloPoseTrtDetector] raw pose stride too small: stride=%d classes=%d\n",
                   stride_abs, class_count);
      return out;
    }

    const int max_det = std::max(1, params_.max_det);
    std::vector<DecodedDetection> candidates;
    candidates.reserve(static_cast<size_t>(std::min(output_det_count_, max_det * 8)));

    for (int i = 0; i < output_det_count_; ++i) {
      int cls = 0;
      float score = -1.0f;

      if (params_.target_class_id >= 0 && params_.target_class_id < class_count) {
        cls = params_.target_class_id;
        score = getDetVal(i, kRawBoxFields + cls);
      } else {
        for (int c = 0; c < class_count; ++c) {
          const float cls_score = getDetVal(i, kRawBoxFields + c);
          if (cls_score > score) {
            score = cls_score;
            cls = c;
          }
        }
      }

      max_score_seen = std::max(max_score_seen, score);
      if (score < params_.conf_thresh) continue;

      const float cx = getDetVal(i, 0);
      const float cy = getDetVal(i, 1);
      const float w = getDetVal(i, 2);
      const float h = getDetVal(i, 3);
      if (w <= 0.0f || h <= 0.0f) continue;

      DecodedDetection det;
      det.score = score;
      det.cls = cls;
      det.has_keypoints = has_pose;
      det.box_net = {
        cx - 0.5f * w,
        cy - 0.5f * h,
        cx + 0.5f * w,
        cy + 0.5f * h
      };

      if (!passesSizeGate(det, i)) continue;

      if (has_pose) {
        for (int k = 0; k < kExpectedKptNum; ++k) {
          det.kpts_net[static_cast<size_t>(k)] = {
            getDetVal(i, kpt_offset + k * 3 + 0),
            getDetVal(i, kpt_offset + k * 3 + 1)
          };
          det.kpt_scores[static_cast<size_t>(k)] =
            getDetVal(i, kpt_offset + k * 3 + 2);
        }
      }

      candidates.push_back(det);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    const float nms_thresh = clampf(params_.nms_thresh, 0.0f, 1.0f);
    std::vector<DecodedDetection> kept;
    kept.reserve(static_cast<size_t>(max_det));

    for (const auto& cand : candidates) {
      bool suppressed = false;
      for (const auto& prev : kept) {
        if (cand.cls != prev.cls) continue;
        if (boxIou(cand.box_net, prev.box_net) > nms_thresh) {
          suppressed = true;
          break;
        }
      }
      if (suppressed) continue;

      kept.push_back(cand);
      if (static_cast<int>(kept.size()) >= max_det) break;
    }

    for (size_t i = 0; i < kept.size(); ++i) {
      auto pr = makePair(kept[i], static_cast<int>(i));
      if (pr.confidence < 0.0f) continue;

      if (s_accept_dbg < 30) {
        ++s_accept_dbg;
        std::fprintf(stderr,
                     "[YoloPoseTrtDetector][DBG] accept raw %s[%zu]: score=%.6f cls=%d center=(%.1f,%.1f) wh=(%.1f,%.1f) precise=%d\n",
                     has_pose ? "pose" : "detect",
                     i, pr.confidence, kept[i].cls, pr.center.x, pr.center.y,
                     pr.width, pr.height, pr.has_precise_corners ? 1 : 0);
      }

      out.push_back(pr);
    }
  } else {
    const int kpt_offset = kPerDetBase;

    if (output_format_ == OutputFormat::POST_NMS_XYXY && stride_abs < kExpectedStride) {
      std::fprintf(stderr,
                   "[YoloPoseTrtDetector] post-NMS output stride too small for pose: %d\n",
                   stride_abs);
      return out;
    }
    if (output_format_ == OutputFormat::POST_NMS_DETECT_XYXY && stride_abs < 6) {
      std::fprintf(stderr,
                   "[YoloPoseTrtDetector] post-NMS detect output stride too small: %d\n",
                   stride_abs);
      return out;
    }

    for (int i = 0; i < output_det_count_; ++i) {
      const float score = getDetVal(i, 4);
      const int cls = static_cast<int>(std::round(getDetVal(i, 5)));
      max_score_seen = std::max(max_score_seen, score);

      if (score < params_.conf_thresh) {
        continue;
      }

      if (params_.target_class_id >= 0 && cls != params_.target_class_id) {
        if (s_reject_dbg < 20) {
          ++s_reject_dbg;
          std::fprintf(stderr,
                       "[YoloPoseTrtDetector][DBG] reject det[%d]: class=%d expect=%d score=%.6f\n",
                       i, cls, params_.target_class_id, score);
        }
        continue;
      }

      DecodedDetection det;
      det.score = score;
      det.cls = cls;
      det.has_keypoints = (output_format_ == OutputFormat::POST_NMS_XYXY);
      det.box_net = {
        getDetVal(i, 0),
        getDetVal(i, 1),
        getDetVal(i, 2),
        getDetVal(i, 3)
      };

      if (!passesSizeGate(det, i)) continue;

      if (det.has_keypoints) {
        for (int k = 0; k < kExpectedKptNum; ++k) {
          det.kpts_net[static_cast<size_t>(k)] = {
            getDetVal(i, kpt_offset + k * 3 + 0),
            getDetVal(i, kpt_offset + k * 3 + 1)
          };
          det.kpt_scores[static_cast<size_t>(k)] =
            getDetVal(i, kpt_offset + k * 3 + 2);
        }
      }

      auto pr = makePair(det, i);
      if (pr.confidence < 0.0f) continue;

      if (s_accept_dbg < 30) {
        ++s_accept_dbg;
        std::fprintf(stderr,
                     "[YoloPoseTrtDetector][DBG] accept det[%d]: score=%.6f cls=%d center=(%.1f,%.1f) wh=(%.1f,%.1f) precise=%d\n",
                     i, score, cls, pr.center.x, pr.center.y,
                     pr.width, pr.height, pr.has_precise_corners ? 1 : 0);
      }

      out.push_back(pr);
    }
  }

  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.confidence > b.confidence;
  });

  if (out.empty()) {
    static int s_empty_print = 0;
    if (s_empty_print < 30) {
      ++s_empty_print;
      std::fprintf(stderr,
                   "[YoloPoseTrtDetector][DBG] parseOutput no detections, max_score_seen=%.6f conf_thresh=%.6f class_id=%d\n",
                   max_score_seen, params_.conf_thresh, params_.target_class_id);
    }
  } else {
    static int s_hit_print = 0;
    if (s_hit_print < 30) {
      ++s_hit_print;
      std::fprintf(stderr,
                   "[YoloPoseTrtDetector][DBG] parseOutput detections=%zu top_score=%.6f\n",
                   out.size(), out.front().confidence);
    }
  }

  return out;
}

}  // namespace auto_aim_fast
