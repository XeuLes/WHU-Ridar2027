#pragma once
#include <cstdint>
#include <string>

namespace auto_aim_fast::util
{
int64_t nowSteadyNs();

struct StageTimer
{
  void tic();
  int64_t tocNs() const;
  double tocMs() const;

private:
  int64_t t0_{0};
};

std::string formatNs(int64_t ns);
}  // namespace auto_aim_fast::util
