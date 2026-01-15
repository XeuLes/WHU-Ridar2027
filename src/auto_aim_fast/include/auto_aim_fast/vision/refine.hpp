#pragma once
#include "auto_aim_fast/vision/pairing.hpp"

namespace auto_aim_fast::vision
{
class Refine
{
public:
  Refine() = default;
  Armor2D refine(const Armor2D & a) const;
};
}  // namespace auto_aim_fast::vision
