#include "radar_decision_comm/io/serial_port.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace radar_decision_comm
{

namespace
{

speed_t baud_to_termios(int baudrate)
{
  switch (baudrate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default: return B115200;
  }
}

}  // namespace

SerialPort::~SerialPort()
{
  close();
}

bool SerialPort::open(const std::string & device, int baudrate, int timeout_ms)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }

  timeout_ms_ = timeout_ms;
  device_ = device;

  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    return false;
  }

  struct termios tty {};
  if (tcgetattr(fd_, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  cfmakeraw(&tty);

  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~CRTSCTS;

  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_oflag = 0;
  tty.c_lflag = 0;

  const speed_t speed = baud_to_termios(baudrate);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);

  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = static_cast<cc_t>((timeout_ms_ + 99) / 100);

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  tcflush(fd_, TCIOFLUSH);
  return true;
}

void SerialPort::close()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialPort::is_open() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return fd_ >= 0;
}

std::size_t SerialPort::write(const std::vector<uint8_t> & data)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd_ < 0 || data.empty()) {
    return 0;
  }

  std::size_t total_written = 0;
  while (total_written < data.size()) {
    const auto n = ::write(
      fd_,
      data.data() + total_written,
      data.size() - total_written);
    if (n <= 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    total_written += static_cast<std::size_t>(n);
  }
  return total_written;
}

std::vector<uint8_t> SerialPort::read_some(std::size_t max_bytes)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<uint8_t> out;
  if (fd_ < 0 || max_bytes == 0) {
    return out;
  }

  out.resize(max_bytes);
  const auto n = ::read(fd_, out.data(), max_bytes);
  if (n <= 0) {
    out.clear();
    return out;
  }

  out.resize(static_cast<std::size_t>(n));
  return out;
}

std::string SerialPort::device() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return device_;
}

}  // namespace radar_decision_comm
