#include "auto_aim_fast/util/time.hpp"
#include <chrono>
#include <sstream>

namespace auto_aim_fast::util
{
int64_t nowSteadyNs()
{
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

void StageTimer::tic()
{
  t0_ = nowSteadyNs();
}

int64_t StageTimer::tocNs() const
{
  const int64_t t1 = nowSteadyNs();
  return (t0_ > 0) ? (t1 - t0_) : 0;
}

double StageTimer::tocMs() const
{
  return static_cast<double>(tocNs()) / 1e6;
}

std::string formatNs(int64_t ns)
{
  std::ostringstream oss;
  if (ns < 1000) {
    oss << ns << " ns";
  } else if (ns < 1000000) {
    oss << (ns / 1000.0) << " us";
  } else if (ns < 1000000000) {
    oss << (ns / 1000000.0) << " ms";
  } else {
    oss << (ns / 1000000000.0) << " s";
  }
  return oss.str();
}
}  // namespace auto_aim_fast::util
