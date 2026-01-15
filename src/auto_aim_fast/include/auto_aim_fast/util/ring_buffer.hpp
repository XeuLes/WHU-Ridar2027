#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace auto_aim_fast::util
{
// A simple thread-safe ring buffer placeholder.
// Later we can replace with a lock-free SPSC queue for 200FPS.
template <typename T>
class RingBuffer
{
public:
  explicit RingBuffer(size_t capacity)
  : buf_(capacity), cap_(capacity) {}

  bool push(const T & v)
  {
    std::lock_guard<std::mutex> lk(m_);
    if (cap_ == 0) return false;
    buf_[head_] = v;
    head_ = (head_ + 1) % cap_;
    if (size_ < cap_) {
      size_++;
    } else {
      // overwrite oldest
      tail_ = (tail_ + 1) % cap_;
    }
    return true;
  }

  std::optional<T> pop()
  {
    std::lock_guard<std::mutex> lk(m_);
    if (size_ == 0) return std::nullopt;
    T v = buf_[tail_];
    tail_ = (tail_ + 1) % cap_;
    size_--;
    return v;
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lk(m_);
    return size_;
  }

private:
  mutable std::mutex m_;
  std::vector<T> buf_;
  size_t cap_{0};
  size_t head_{0};
  size_t tail_{0};
  size_t size_{0};
};
}  // namespace auto_aim_fast::util
