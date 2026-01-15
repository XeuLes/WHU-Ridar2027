#pragma once
#include "auto_aim_fast/vision/preprocess.hpp"
#include <vector>

namespace auto_aim_fast::vision
{
struct LightBar
{
  // placeholder for rotated rect / endpoints
  float x{0}, y{0};
  float angle{0};
  float length{0};
};

class LightExtractor
{
public:
  LightExtractor() = default;
  std::vector<LightBar> extract(const PreprocessResult & in) const;
};
}  // namespace auto_aim_fast::vision
