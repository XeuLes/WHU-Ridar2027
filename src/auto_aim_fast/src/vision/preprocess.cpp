#include "auto_aim_fast/vision/preprocess.hpp"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

namespace auto_aim_fast::vision
{

static cv::Rect clampRoi(const cv::Rect & r, int w, int h)
{
  cv::Rect img(0, 0, w, h);
  cv::Rect rr = r & img;
  if (rr.width <= 0 || rr.height <= 0) return cv::Rect();
  return rr;
}

PreprocessResult Preprocess::run(const auto_aim_fast::pipeline::Frame & frame) const
{
  PreprocessResult r;
  if (!frame.image) {
    r.ok = false;
    return r;
  }

  try {
    auto cv_ptr = cv_bridge::toCvShare(frame.image, sensor_msgs::image_encodings::BGR8);
    const cv::Mat full = cv_ptr->image;
    if (full.empty()) {
      r.ok = false;
      return r;
    }
    r.full_bgr = full;

    // ROI view (zero-copy) if valid
    if (frame.roi.valid) {
      cv::Rect want(frame.roi.x, frame.roi.y, frame.roi.w, frame.roi.h);
      cv::Rect rr = clampRoi(want, full.cols, full.rows);
      if (rr.area() > 0) {
        r.bgr = full(rr);
        r.roi_used = true;
        r.roi_rect = rr;
      } else {
        r.bgr = full;
      }
    } else {
      r.bgr = full;
    }

    r.ok = !r.bgr.empty();
    return r;
  } catch (...) {
    r.ok = false;
    return r;
  }
}

}  // namespace auto_aim_fast::vision
