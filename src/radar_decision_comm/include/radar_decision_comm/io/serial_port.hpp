#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace radar_decision_comm
{

class SerialPort
{
public:
  SerialPort() = default;
  ~SerialPort();

  bool open(const std::string & device, int baudrate, int timeout_ms = 2);
  void close();
  bool is_open() const;

  std::size_t write(const std::vector<uint8_t> & data);
  std::vector<uint8_t> read_some(std::size_t max_bytes = 512);

  std::string device() const;

private:
  int fd_{-1};
  int timeout_ms_{2};
  std::string device_;
  mutable std::mutex mutex_;
};

}  // namespace radar_decision_comm
