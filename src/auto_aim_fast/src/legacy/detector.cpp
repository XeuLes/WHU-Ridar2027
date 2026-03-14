#include "auto_aim_fast/legacy/detector.hpp" // 确保包含正确的头文件
#include <algorithm>
#include <cmath>
#include <numeric> // For std::accumulate if needed
#include <iostream> // For std::cout debug logging

// detector.cpp 顶部追加
#ifdef AUTO_AIM_FAST_USE_CUDA
  #include <opencv2/core/cuda.hpp>
  #include <opencv2/cudaimgproc.hpp>
  #include <opencv2/cudaarithm.hpp>
  #include <opencv2/cudafilters.hpp>
#endif

#ifdef AUTO_AIM_FAST_USE_CUDA
  #include <cuda_runtime.h>
  #include <opencv2/core/cuda_stream_accessor.hpp>

  struct Float2 { float x, y; };

  extern "C" bool detect_color_cuda(
      const cv::cuda::GpuMat& d_bgr,
      int roi_x, int roi_y, int roi_w, int roi_h,
      const Float2 rect_pts[4],
      int sample_step,
      int* h_red_sum, int* h_blue_sum, int* h_count,
      cudaStream_t stream);

  struct FrameCudaCache {
    bool valid = false;
    cv::cuda::GpuMat d_bgr;
    cv::cuda::Stream stream;
    int w = 0, h = 0;
  };

  static thread_local FrameCudaCache g_cuda_frame;
#endif


// 静态成员初始化
const cv::Scalar SimpleLightDetector::RED_COLOR = cv::Scalar(0, 0, 255);
const cv::Scalar SimpleLightDetector::BLUE_COLOR = cv::Scalar(255, 0, 0);
const cv::Scalar SimpleLightDetector::GREEN_COLOR = cv::Scalar(0, 255, 0);
const cv::Scalar SimpleLightDetector::YELLOW_COLOR = cv::Scalar(0, 255, 255);


// ==================== SimpleLightDetector 方法实现 ====================

// 构造函数
SimpleLightDetector::SimpleLightDetector() {
    // 使用默认参数
}

SimpleLightDetector::SimpleLightDetector(const Params& params)
    : params_(params) {
}

// 设置参数
void SimpleLightDetector::setParams(const Params& params) {
    params_ = params;
}

// 获取参数
SimpleLightDetector::Params SimpleLightDetector::getParams() const {
    return params_;
}

// 主检测函数
std::vector<cv::Point2f> SimpleLightDetector::detect(const cv::Mat& image) {
    last_spots_.clear();
    last_pairs_.clear();

#ifdef AUTO_AIM_FAST_USE_CUDA
    if (!image.empty() && image.type() == CV_8UC3 &&
        cv::cuda::getCudaEnabledDeviceCount() > 0) {
        g_cuda_frame.valid = true;
        g_cuda_frame.w = image.cols;
        g_cuda_frame.h = image.rows;
        g_cuda_frame.d_bgr.upload(image, g_cuda_frame.stream);
    } else {
        g_cuda_frame.valid = false;
    }
#endif

    last_binary_ = preprocess(image);
    last_spots_ = detectSpots(image, last_binary_);
    last_pairs_ = groupSpotsToPatterns(last_spots_);

    std::vector<cv::Point2f> centers;
    centers.reserve(last_pairs_.size());
    for (const auto& pair : last_pairs_) {
        centers.push_back(pair.center);
    }

#ifdef AUTO_AIM_FAST_USE_CUDA
    g_cuda_frame.valid = false;
#endif

    return centers;
}
// 扩展检测函数
std::vector<SimpleLightDetector::PairResult> SimpleLightDetector::detectDetailed(const cv::Mat& image) {
    // 先执行普通检测
    detect(image);
    return last_pairs_;
}

// 单次检测
bool SimpleLightDetector::detectSingle(const cv::Mat& image, cv::Point2f& center) {
    // 执行检测
    auto pairs = detectDetailed(image);

    if (pairs.empty()) {
        return false;
    }

    // 找到置信度最高的配对
    auto best_pair = std::max_element(pairs.begin(), pairs.end(),
        [](const PairResult& a, const PairResult& b) {
            return a.confidence < b.confidence;
        });

    center = best_pair->center;
    return true;
}

// 绘制结果
// 注意：移除了 const 限定符
void SimpleLightDetector::drawResults(cv::Mat& image, bool draw_lights, bool draw_centers) {
    createDebugImage(image, last_binary_);

    if (params_.show_contours) {
        cv::Mat contours_img = image.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(last_binary_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(contours_img, contours, -1, cv::Scalar(255, 255, 255), 1);
        cv::imshow("Contours", contours_img);
    }

    // 画 spot
    if (draw_lights) {
        for (const auto& spot : last_spots_) {
            cv::Scalar color = (spot.color == 0) ? RED_COLOR : BLUE_COLOR;

            cv::Point2f vertices[4];
            spot.rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], color, 1);
            }

            cv::circle(image, spot.center, 2, YELLOW_COLOR, -1);
        }
    }

    // 画 pattern
    if (draw_centers && params_.show_pairs) {
        for (const auto& pair : last_pairs_) {
            cv::circle(image, pair.center, 6, YELLOW_COLOR, 2);
            cv::line(image, pair.light1_center, pair.light2_center, GREEN_COLOR, 2);

            const float half_w = pair.width * 0.5f;
            const float half_h = pair.height * 0.5f;
            const float c = std::cos(pair.angle);
            const float s = std::sin(pair.angle);

            auto rot = [&](float x, float y) -> cv::Point2f {
                return cv::Point2f(pair.center.x + c * x - s * y,
                                   pair.center.y + s * x + c * y);
            };

            cv::Point2f lt = rot(-half_w, -half_h);
            cv::Point2f rt = rot( half_w, -half_h);
            cv::Point2f rb = rot( half_w,  half_h);
            cv::Point2f lb = rot(-half_w,  half_h);

            cv::line(image, lt, rt, YELLOW_COLOR, 1);
            cv::line(image, rt, rb, YELLOW_COLOR, 1);
            cv::line(image, rb, lb, YELLOW_COLOR, 1);
            cv::line(image, lb, lt, YELLOW_COLOR, 1);

            cv::circle(image, pair.light1_center, 4, GREEN_COLOR, -1);  // 上行中心
            cv::circle(image, pair.light2_center, 4, BLUE_COLOR, -1);   // 下行中心

            std::string conf_text = "C:" + std::to_string(int(pair.confidence * 100)) + "%";
            cv::putText(image, conf_text, pair.center + cv::Point2f(0, pair.height * 0.25f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1);
        }
    }

    std::string spot_text = "Spots: " + std::to_string(last_spots_.size());
    cv::putText(image, spot_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, GREEN_COLOR, 2);

    std::string pair_text = "Patterns: " + std::to_string(last_pairs_.size());
    cv::putText(image, pair_text, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, YELLOW_COLOR, 2);
}
// 获取最后检测到的亮斑
const std::vector<SimpleLightDetector::Spot>& SimpleLightDetector::getLastSpots() const {
    return last_spots_;
}

// 获取最后的配对结果
const std::vector<SimpleLightDetector::PairResult>& SimpleLightDetector::getLastPairs() const {
    return last_pairs_;
}

// 获取调试图像
cv::Mat SimpleLightDetector::getBinaryImage() const {
    cv::Mat color_binary;
    cv::cvtColor(last_binary_, color_binary, cv::COLOR_GRAY2BGR);
    return color_binary;
}

cv::Mat SimpleLightDetector::getDebugImage() const {
    return last_debug_.clone();
}

// ==================== 私有方法实现 ====================

// 预处理
cv::Mat SimpleLightDetector::preprocess(const cv::Mat& image) const
{
#ifdef AUTO_AIM_FAST_USE_CUDA
    // 运行时兜底：没 CUDA 设备 / 输入不对就退回 CPU
    if (!image.empty() &&
        image.type() == CV_8UC3 &&
        cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        // 线程内缓存：避免每帧反复创建 Filter / 反复申请显存
        struct CudaCache {
            bool inited = false;
            cv::cuda::GpuMat d_bgr, d_gray, d_bin, d_tmp;
            // cv::Ptr<cv::cuda::Filter> close_f;
            cv::Ptr<cv::cuda::Filter> open_f;
            cv::cuda::Stream stream;
        };
        static thread_local CudaCache cache;

        if (!cache.inited) {
            // kernel 与 CPU 版完全一致：3x3 MORPH_RECT
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            // cache.close_f = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel);
            cache.open_f  = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN,  CV_8UC1, kernel);
            cache.inited = true;
        }

        // upload -> cuda preprocess -> download
        cache.d_bgr.upload(image, cache.stream);
        cv::cuda::cvtColor(cache.d_bgr, cache.d_gray, cv::COLOR_BGR2GRAY, 0, cache.stream);

        // 阈值逻辑保持不变：THRESH_BINARY + params_.binary_thresh
        double thr = params_.binary_thresh;

        if (params_.use_sigma_thresh) {
            cv::Mat gray_cpu;
            cache.d_gray.download(gray_cpu, cache.stream);
            cache.stream.waitForCompletion();

            cv::Scalar mean_val, std_val;
            cv::meanStdDev(gray_cpu, mean_val, std_val);

            const double mu = mean_val[0];
            const double sigma = std_val[0];

            thr = mu + static_cast<double>(params_.sigma_k) * sigma;
            thr = std::max<double>(params_.min_dynamic_thresh,
                std::min<double>(thr, params_.max_dynamic_thresh));
        }

        cv::cuda::threshold(
            cache.d_gray, cache.d_bin,
            thr, 255.0,
            cv::THRESH_BINARY, cache.stream);

        cache.open_f->apply(cache.d_bin, cache.d_tmp, cache.stream);

        cv::Mat binary;
        cache.d_tmp.download(binary, cache.stream);
        cache.stream.waitForCompletion();
        return binary;
    }
#endif

    // ========= CPU fallback（保持你原来的逻辑不变）=========
    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    double thr = params_.binary_thresh;

    if (params_.use_sigma_thresh) {
        cv::Scalar mean_val, std_val;
        cv::meanStdDev(gray, mean_val, std_val);

        const double mu = mean_val[0];
        const double sigma = std_val[0];

        thr = mu + static_cast<double>(params_.sigma_k) * sigma;

        // 钳制到合理范围
        thr = std::max<double>(params_.min_dynamic_thresh,
            std::min<double>(thr, params_.max_dynamic_thresh));
    }

    cv::threshold(gray, binary, thr, 255, cv::THRESH_BINARY);

    cv::medianBlur(binary, binary, 3);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    return binary;
}

std::vector<SimpleLightDetector::Spot>
SimpleLightDetector::detectSpots(const cv::Mat& color_img, const cv::Mat& binary_img)
{
    std::vector<Spot> spots;
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int pass_area = 0;
    int pass_shape = 0;
    int pass_color = 0;

    std::cout << "[detectSpots] contours=" << contours.size() << std::endl;

    for (const auto& contour : contours) {
        if (contour.size() < 3) continue;

        const double area = cv::contourArea(contour);
        // if (area < params_.spot_min_area || area > params_.spot_max_area) {
        //     continue;
        // }
        pass_area++;

        cv::RotatedRect rect = cv::minAreaRect(contour);
        Spot spot(rect);

        if (spot.width < 1.0f || spot.height < 1.0f) {
            continue;
        }

        if (!isValidSpot(spot)) {
            std::cout << "[detectSpots] reject shape:"
                      << " area=" << spot.area
                      << " w=" << spot.width
                      << " h=" << spot.height
                      << " ratio=" << (std::min(spot.width, spot.height) / std::max(spot.width, spot.height))
                      << std::endl;
            continue;
        }
        pass_shape++;

        spot.color = detectColorFromRect(spot.rect, color_img);

        std::cout << "[detectSpots] color check:"
                  << " area=" << spot.area
                  << " center=(" << spot.center.x << "," << spot.center.y << ")"
                  << " color=" << spot.color
                  << " target=" << params_.target_color
                  << std::endl;

        if (spot.color != params_.target_color) {
            continue;
        }
        pass_color++;

        spots.push_back(spot);
    }

    std::cout << "[detectSpots] pass_area=" << pass_area
              << " pass_shape=" << pass_shape
              << " pass_color=" << pass_color
              << " final_spots=" << spots.size()
              << std::endl;

    return spots;
}

// 判断是否为有效灯条
bool SimpleLightDetector::isValidSpot(const Spot& spot) const {
    std::cout << "[isValidSpot FORCE PASS] "
              << "area=" << spot.area
              << " w=" << spot.width
              << " h=" << spot.height
              << std::endl;
    return true;
}


// 检测灯条颜色
int SimpleLightDetector::detectColorFromRect(const cv::RotatedRect& rect_r,
                                             const cv::Mat& color_img) const
{
    cv::Rect rect = rect_r.boundingRect();

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width  = std::min(rect.width,  color_img.cols - rect.x);
    rect.height = std::min(rect.height, color_img.rows - rect.y);

    if (rect.width <= 0 || rect.height <= 0) {
        return -1;
    }

#ifdef AUTO_AIM_FAST_USE_CUDA
    if (g_cuda_frame.valid && !g_cuda_frame.d_bgr.empty() &&
        g_cuda_frame.d_bgr.cols == color_img.cols &&
        g_cuda_frame.d_bgr.rows == color_img.rows &&
        cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::vector<cv::Point2f> pts(4);
        rect_r.points(pts.data());

        Float2 fpts[4] = {
            {pts[0].x, pts[0].y},
            {pts[1].x, pts[1].y},
            {pts[2].x, pts[2].y},
            {pts[3].x, pts[3].y}
        };

        int red_sum = 0, blue_sum = 0, pixel_count = 0;
        const int sample_step = 2;

        cudaStream_t s = cv::cuda::StreamAccessor::getStream(g_cuda_frame.stream);
        bool ok = detect_color_cuda(
            g_cuda_frame.d_bgr,
            rect.x, rect.y, rect.width, rect.height,
            fpts,
            sample_step,
            &red_sum, &blue_sum, &pixel_count,
            s);

        g_cuda_frame.stream.waitForCompletion();

        if (ok && pixel_count > 0) {
            if (red_sum > blue_sum + params_.color_threshold) return 0;
            if (blue_sum > red_sum + params_.color_threshold) return 1;
            return -1;
        }
    }
#endif

    cv::Mat roi = color_img(rect);

    int red_sum = 0, blue_sum = 0;
    int pixel_count = 0;

    std::vector<cv::Point2f> rect_points(4);
    rect_r.points(rect_points.data());
    std::vector<cv::Point> polygon_points(rect_points.begin(), rect_points.end());

    for (int i = 0; i < roi.rows; i += 2) {
        const cv::Vec3b* row = roi.ptr<cv::Vec3b>(i);
        for (int j = 0; j < roi.cols; j += 2) {
            cv::Point abs_pt(j + rect.x, i + rect.y);
            double test_result = cv::pointPolygonTest(polygon_points, abs_pt, false);
            if (test_result >= 0) {
                red_sum += row[j][2];
                blue_sum += row[j][0];
                pixel_count++;
            }
        }
    }

    if (pixel_count == 0) return -1;

    if (red_sum > blue_sum + params_.color_threshold) return 0;
    if (blue_sum > red_sum + params_.color_threshold) return 1;
    return -1;
}

float SimpleLightDetector::calculatePatternConfidence(const std::vector<Spot>& top,
                                                      const std::vector<Spot>& bottom,
                                                      float width_px,
                                                      float height_px) const
{
    if (top.empty() || bottom.empty() || width_px <= 1.f || height_px <= 1.f) {
        return 0.0f;
    }

    auto mean_center = [](const std::vector<Spot>& row) {
        cv::Point2f c(0.f, 0.f);
        for (const auto& s : row) c += s.center;
        c *= (1.f / std::max<size_t>(1, row.size()));
        return c;
    };

    const cv::Point2f top_center = mean_center(top);
    const cv::Point2f bottom_center = mean_center(bottom);

    float count_score = 1.0f;
    const int nt = static_cast<int>(top.size());
    const int nb = static_cast<int>(bottom.size());
    if (nt < params_.min_row_spots || nb < params_.min_row_spots) {
        return 0.0f;
    }
    if (nt != nb) {
        count_score *= 0.85f;
    }
    if ((nt != 3 && nt != 4) || (nb != 3 && nb != 4)) {
        count_score *= 0.85f;
    }

    const float center_x_offset = std::abs(top_center.x - bottom_center.x);
    const float align_den =
        std::max(1.0f, width_px * static_cast<float>(params_.row_center_x_tol_ratio));
    float align_score = 1.0f - center_x_offset / align_den;
    align_score = std::max(0.0f, std::min(1.0f, align_score));

    const float top_width = top.back().center.x - top.front().center.x;
    const float bottom_width = bottom.back().center.x - bottom.front().center.x;
    const float width_diff = std::abs(top_width - bottom_width);

    const float width_den =
        std::max(1.0f, width_px * static_cast<float>(params_.row_width_similarity_ratio));
    float width_score = 1.0f - width_diff / width_den;
    width_score = std::max(0.0f, std::min(1.0f, width_score));

    const float aspect = width_px / std::max(1.0f, height_px);
    const float aspect_mid =
        0.5f * (static_cast<float>(params_.pattern_min_aspect) +
                static_cast<float>(params_.pattern_max_aspect));
    const float aspect_half =
        0.5f * (static_cast<float>(params_.pattern_max_aspect) -
                static_cast<float>(params_.pattern_min_aspect));

    float aspect_score = 1.0f;
    if (aspect_half > 1e-6f) {
        aspect_score = 1.0f - std::abs(aspect - aspect_mid) / aspect_half;
        aspect_score = std::max(0.0f, std::min(1.0f, aspect_score));
    }

    auto row_angle_abs = [](const std::vector<Spot>& row) {
        if (row.size() < 2) return 0.0f;
        cv::Point2f d = row.back().center - row.front().center;
        return std::abs(std::atan2(d.y, d.x));
    };

    const float top_angle = row_angle_abs(top);
    const float bottom_angle = row_angle_abs(bottom);
    const float mean_angle = 0.5f * (top_angle + bottom_angle);

    const float max_angle = 20.0f * static_cast<float>(CV_PI) / 180.0f;
    float angle_score = 1.0f - mean_angle / std::max(1e-6f, max_angle);
    angle_score = std::max(0.0f, std::min(1.0f, angle_score));

    float confidence =
        count_score *
        align_score *
        width_score *
        aspect_score *
        angle_score;

    confidence = std::max(0.0f, std::min(1.0f, confidence));
    return confidence;
}

std::vector<SimpleLightDetector::PairResult>
SimpleLightDetector::groupSpotsToPatterns(const std::vector<Spot>& spots)
{
    std::vector<PairResult> out;

    if (spots.size() < 4) {
        std::cout << "[pattern] reject: spots.size()=" << spots.size() << " < 4" << std::endl;
        return out;
    }

    std::vector<Spot> sorted = spots;
    std::sort(sorted.begin(), sorted.end(),
              [](const Spot& a, const Spot& b) {
                  return a.center.y < b.center.y;
              });

    std::cout << "[pattern] input spots=" << sorted.size() << std::endl;
    for (size_t i = 0; i < sorted.size(); ++i) {
        std::cout << "  [pattern] spot[" << i << "]"
                  << " center=(" << sorted[i].center.x << "," << sorted[i].center.y << ")"
                  << " area=" << sorted[i].area
                  << " w=" << sorted[i].width
                  << " h=" << sorted[i].height
                  << " color=" << sorted[i].color
                  << std::endl;
    }

    // 按 y 排序后，找最大相邻间隔作为上下两排分界
    int split_idx = -1;
    float max_gap = -1.0f;
    for (int i = 0; i + 1 < static_cast<int>(sorted.size()); ++i) {
        const float gap = sorted[i + 1].center.y - sorted[i].center.y;
        std::cout << "[pattern] gap[" << i << "->" << (i + 1) << "]=" << gap << std::endl;
        if (gap > max_gap) {
            max_gap = gap;
            split_idx = i;
        }
    }

    if (split_idx < 0) {
        std::cout << "[pattern] reject: split_idx < 0" << std::endl;
        return out;
    }

    std::vector<Spot> top, bottom;
    for (int i = 0; i <= split_idx; ++i) top.push_back(sorted[i]);
    for (int i = split_idx + 1; i < static_cast<int>(sorted.size()); ++i) bottom.push_back(sorted[i]);

    std::cout << "[pattern] split_idx=" << split_idx
              << " max_gap=" << max_gap
              << " top.size=" << top.size()
              << " bottom.size=" << bottom.size()
              << std::endl;

    if (static_cast<int>(top.size()) < params_.min_row_spots ||
        static_cast<int>(bottom.size()) < params_.min_row_spots) {
        std::cout << "[pattern] reject: row spot count too small, "
                  << "top=" << top.size()
                  << " bottom=" << bottom.size()
                  << " min_row_spots=" << params_.min_row_spots
                  << std::endl;
        return out;
    }

    auto sort_x = [](const Spot& a, const Spot& b) {
        return a.center.x < b.center.x;
    };
    std::sort(top.begin(), top.end(), sort_x);
    std::sort(bottom.begin(), bottom.end(), sort_x);

    std::cout << "[pattern] top row after sort_x:" << std::endl;
    for (size_t i = 0; i < top.size(); ++i) {
        std::cout << "  top[" << i << "] center=("
                  << top[i].center.x << "," << top[i].center.y << ")" << std::endl;
    }
    std::cout << "[pattern] bottom row after sort_x:" << std::endl;
    for (size_t i = 0; i < bottom.size(); ++i) {
        std::cout << "  bottom[" << i << "] center=("
                  << bottom[i].center.x << "," << bottom[i].center.y << ")" << std::endl;
    }

    auto mean_center = [](const std::vector<Spot>& row) {
        cv::Point2f c(0.f, 0.f);
        for (const auto& s : row) c += s.center;
        c *= (1.f / std::max<size_t>(1, row.size()));
        return c;
    };

    const cv::Point2f top_center = mean_center(top);
    const cv::Point2f bottom_center = mean_center(bottom);
    const cv::Point2f target_center = 0.5f * (top_center + bottom_center);

    const float top_width = top.back().center.x - top.front().center.x;
    const float bottom_width = bottom.back().center.x - bottom.front().center.x;
    const float width_px = 0.5f * (top_width + bottom_width);
    const float height_px = std::abs(bottom_center.y - top_center.y);

    std::cout << "[pattern] centers:"
              << " top_center=(" << top_center.x << "," << top_center.y << ")"
              << " bottom_center=(" << bottom_center.x << "," << bottom_center.y << ")"
              << " target_center=(" << target_center.x << "," << target_center.y << ")"
              << std::endl;

    std::cout << "[pattern] sizes:"
              << " top_width=" << top_width
              << " bottom_width=" << bottom_width
              << " width_px=" << width_px
              << " height_px=" << height_px
              << std::endl;

    if (width_px <= 1.f || height_px <= 1.f) {
        std::cout << "[pattern] reject: width_px<=1 or height_px<=1" << std::endl;
        return out;
    }

    auto row_y_span = [](const std::vector<Spot>& row) {
        float y_min = row.front().center.y;
        float y_max = row.front().center.y;
        for (const auto& s : row) {
            y_min = std::min(y_min, s.center.y);
            y_max = std::max(y_max, s.center.y);
        }
        return y_max - y_min;
    };

    const float top_y_span = row_y_span(top);
    const float bottom_y_span = row_y_span(bottom);

    // 方案 A：一排缺边缘点时，允许行内离散稍大一点
    float y_span_limit = height_px * 0.35f;
    if (top.size() != bottom.size()) {
        y_span_limit *= 1.5f;
    }

    std::cout << "[pattern] y_span:"
              << " top_y_span=" << top_y_span
              << " bottom_y_span=" << bottom_y_span
              << " limit=" << y_span_limit
              << std::endl;

    if (top_y_span > y_span_limit || bottom_y_span > y_span_limit) {
        std::cout << "[pattern] reject: row_y_span too large" << std::endl;
        return out;
    }

    const float aspect = width_px / height_px;
    std::cout << "[pattern] aspect=" << aspect
              << " range=[" << params_.pattern_min_aspect
              << ", " << params_.pattern_max_aspect << "]"
              << std::endl;

    if (aspect < params_.pattern_min_aspect ||
        aspect > params_.pattern_max_aspect) {
        std::cout << "[pattern] reject: aspect out of range" << std::endl;
        return out;
    }

    // 方案 A：当上下两排点数不等时，放宽中心 x 偏移约束
    const float center_x_offset = std::abs(top_center.x - bottom_center.x);
    float center_x_limit = width_px * params_.row_center_x_tol_ratio;
    if (top.size() != bottom.size()) {
        center_x_limit *= 1.8f;
    }

    std::cout << "[pattern] center_x_offset=" << center_x_offset
              << " limit=" << center_x_limit
              << std::endl;

    if (center_x_offset > center_x_limit) {
        std::cout << "[pattern] reject: center_x_offset too large" << std::endl;
        return out;
    }

    // 方案 A：当上下两排点数不等时，弱化宽度一致性约束
    const float width_diff = std::abs(top_width - bottom_width);
    float width_diff_limit = width_px * params_.row_width_similarity_ratio;
    if (top.size() != bottom.size()) {
        width_diff_limit *= 2.0f;
    }

    std::cout << "[pattern] width_diff=" << width_diff
              << " limit=" << width_diff_limit
              << std::endl;

    if (width_diff > width_diff_limit) {
        std::cout << "[pattern] reject: width_diff too large" << std::endl;
        return out;
    }

    // 目标顶边方向角（弧度）
    cv::Point2f top_dir = top.back().center - top.front().center;
    const float angle = std::atan2(top_dir.y, top_dir.x);

    PairResult pr;
    pr.center = target_center;         // 目标整体中心
    pr.light1_center = top_center;     // 上排中心
    pr.light2_center = bottom_center;  // 下排中心
    pr.width = width_px;
    pr.height = height_px;
    pr.angle = angle;
    pr.pair_type = 100;
    pr.confidence = calculatePatternConfidence(top, bottom, width_px, height_px);

    std::cout << "[pattern] ACCEPT:"
              << " center=(" << pr.center.x << "," << pr.center.y << ")"
              << " width=" << pr.width
              << " height=" << pr.height
              << " angle=" << pr.angle
              << " conf=" << pr.confidence
              << std::endl;

    out.push_back(pr);
    return out;
}
// 创建调试图像
// 注意：添加了 const 限定符
void SimpleLightDetector::createDebugImage(const cv::Mat& color_img, const cv::Mat& binary_img) const {
    // 创建三通道的调试图像
    cv::Mat binary_color;
    cv::cvtColor(binary_img, binary_color, cv::COLOR_GRAY2BGR);

    // 水平拼接原图和二值图
    // 由于 last_debug_ 是 mutable，const 函数可以修改它
    cv::hconcat(color_img, binary_color, last_debug_);
    // 如果不想用 mutable，可以用 const_cast，但这不推荐：
    // cv::hconcat(color_img, binary_color, const_cast<cv::Mat&>(last_debug_));
}

// 辅助函数：计算两点距离
float SimpleLightDetector::distance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 辅助函数：计算角度差
float SimpleLightDetector::angleDifference(float a1, float a2) const {
    float diff = std::abs(a1 - a2);
    if (diff > 90.0f) {
        diff = 180.0f - diff;
    }
    return diff;
}