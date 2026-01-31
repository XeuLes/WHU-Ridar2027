#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <deque>
#include <utility>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <poll.h>

using namespace std::chrono_literals;

namespace {
#pragma pack(push, 1)
struct GimbalToVision {
    uint8_t  head[2];  // 'S','P'
    uint8_t  mode;     // {0,1}
    float    q[4];     // ignored
    float    yaw;
    float    yaw_vel;
    float    pitch;
    float    pitch_vel;
    float    bullet_speed;  // ignored
    uint16_t bullet_count;  // ignored
    uint16_t crc16;
};

struct VisionToGimbal {
    uint8_t  head[2];  // 'S','P'
    uint8_t  mode;     // {0,1}
    float    yaw;
    float    yaw_vel;
    float    yaw_acc;
    float    pitch;
    float    pitch_vel;
    float    pitch_acc;
    uint16_t crc16;
};
#pragma pack(pop)

static std::string hex_dump(const uint8_t* data, size_t len)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<int>(data[i]);
        if (i + 1 != len) oss << ' ';
    }
    return oss.str();
}

// CRC16: poly=0x1021 reflected -> 0x8408, init=0xFFFF, xorout=0x0000, refin=true
static uint16_t crc16_ccitt_reflect_ffff(const uint8_t* data, size_t len)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]);
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x0001) crc = (crc >> 1) ^ 0x8408;
            else crc >>= 1;
        }
    }
    return crc;
}

static bool set_baudrate(termios& tty, int baud)
{
    speed_t spd = B115200;
    switch (baud) {
    case 9600: spd = B9600; break;
    case 19200: spd = B19200; break;
    case 38400: spd = B38400; break;
    case 57600: spd = B57600; break;
    case 115200: spd = B115200; break;
    case 230400: spd = B230400; break;
    case 460800: spd = B460800; break;
    case 921600: spd = B921600; break;
    default: return false;
    }
    cfsetispeed(&tty, spd);
    cfsetospeed(&tty, spd);
    return true;
}

static int open_serial(const std::string& dev, int baud, std::string& err_out)
{
    err_out.clear();
    int fd = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        err_out = "open() failed: errno=" + std::to_string(errno) + " " + std::string(std::strerror(errno));
        return -1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        err_out = "tcgetattr() failed: errno=" + std::to_string(errno) + " " + std::string(std::strerror(errno));
        ::close(fd);
        return -1;
    }

    cfmakeraw(&tty);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (!set_baudrate(tty, baud)) {
        err_out = "unsupported baud: " + std::to_string(baud);
        ::close(fd);
        return -1;
    }

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        err_out = "tcsetattr() failed: errno=" + std::to_string(errno) + " " + std::string(std::strerror(errno));
        ::close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    return fd;
}

static bool write_all_nonblock(int fd, const uint8_t* buf, size_t n)
{
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = ::write(fd, buf + sent, n - sent);
        if (w > 0) { sent += static_cast<size_t>(w); continue; }
        if (w < 0 && (errno == EAGAIN || errno == EINTR)) {
            std::this_thread::yield();
            continue;
        }
        return false;
    }
    return true;
}

// triangle wave in [-1,1] (continuous)
static double triangle_wave(double t, double period, double phase01)
{
    if (period <= 1e-9) return 0.0;
    double p = std::fmod(t / period + phase01, 1.0);
    if (p < 0) p += 1.0;
    return 4.0 * std::abs(p - 0.5) - 1.0;
}

// sine wave in [-1,1] (continuous)
static double sine_wave(double t, double period, double phase01)
{
    if (period <= 1e-9) return 0.0;
    const double phase = 2.0 * M_PI * (t / period + phase01);
    return std::sin(phase);
}

// step wave in [-1,1] (discrete, sample-hold by time interval)
// - step_dt: time interval between level changes
// - levels: number of discrete levels between [-1,1], >= 2
// - phase01: phase shift in [0,1) applied as shift in "step index" within a cycle
static double step_wave(double t, double step_dt, int levels, double phase01)
{
    if (step_dt <= 1e-9) return 0.0;
    if (levels < 2) levels = 2;

    // number of steps for one back-and-forth cycle
    const int up_steps = levels - 1;                 // from -1 to +1 inclusive needs levels-1 increments
    const int cycle_steps = 2 * up_steps;            // up then down

    // step index by time
    long long k = static_cast<long long>(std::floor(t / step_dt));
    long long phase_shift = static_cast<long long>(std::llround(phase01 * static_cast<double>(cycle_steps)));
    long long s = (k + phase_shift) % cycle_steps;
    if (s < 0) s += cycle_steps;

    // triangle in step domain: 0..up_steps..0
    int idx = 0;
    if (s <= up_steps) idx = static_cast<int>(s);
    else idx = static_cast<int>(cycle_steps - s);

    // map idx in [0, up_steps] to value in [-1, +1]
    double v = -1.0 + 2.0 * (static_cast<double>(idx) / static_cast<double>(up_steps));
    return std::max(-1.0, std::min(1.0, v));
}

static inline void set_pixel_rgb(std::vector<uint8_t>& img, int w, int h, int x, int y,
                                 uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    const size_t idx = static_cast<size_t>((y * w + x) * 3);
    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

static void draw_line(std::vector<uint8_t>& img, int w, int h,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b)
{
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        set_pixel_rgb(img, w, h, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

} // namespace

class GimbalSerialBridge final : public rclcpp::Node {
public:
    explicit GimbalSerialBridge(const rclcpp::NodeOptions& opts)
        : rclcpp::Node("gimbal_serial_bridge", opts)
    {
        declareParams();
        loadParams();

        start_tp_ = std::chrono::steady_clock::now();

        pub_angles_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            angles_topic_, rclcpp::SensorDataQoS());
        pub_mode_ = this->create_publisher<std_msgs::msg::UInt8>(
            mode_up_topic_, rclcpp::QoS(10));

        sub_cmd_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            cmd_topic_, rclcpp::QoS(10),
            std::bind(&GimbalSerialBridge::onCmd, this, std::placeholders::_1));

        sub_enable_ = this->create_subscription<std_msgs::msg::Bool>(
            enable_topic_, rclcpp::QoS(10),
            std::bind(&GimbalSerialBridge::onEnable, this, std::placeholders::_1));

        if (lissajous_enable_) {
            pub_lissajous_img_ = this->create_publisher<sensor_msgs::msg::Image>(
                lissajous_topic_, rclcpp::QoS(1));
            const double hz = std::max(1.0, lissajous_rate_hz_);
            lissajous_timer_ = this->create_wall_timer(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / hz)),
                std::bind(&GimbalSerialBridge::publishLissajousImage, this));
        }

        rx_thread_ = std::thread(&GimbalSerialBridge::rxLoop, this);
        tx_thread_ = std::thread(&GimbalSerialBridge::txLoop, this);

        RCLCPP_INFO(
            this->get_logger(),
            "Started. baud=%d angles_topic=%s cmd_topic=%s enable_topic=%s serial=%s",
            baud_, angles_topic_.c_str(), cmd_topic_.c_str(),
            enable_topic_.c_str(), serial_port_.c_str());
    }

    ~GimbalSerialBridge() override
    {
        quit_.store(true);
        { std::lock_guard<std::mutex> lk(tx_cv_mtx_); tx_req_seq_++; }
        tx_cv_.notify_all();

        if (rx_thread_.joinable()) rx_thread_.join();
        if (tx_thread_.joinable()) tx_thread_.join();
        closeFd();
    }

private:
    void declareParams()
    {
        this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
        this->declare_parameter<std::vector<std::string>>("serial_ports", std::vector<std::string>{});
        this->declare_parameter<int>("baud", 115200);

        this->declare_parameter<std::string>("angles_topic", "/gimbal/angles");
        this->declare_parameter<std::string>("mode_up_topic", "/gimbal/mode_up");
        this->declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
        this->declare_parameter<std::string>("enable_topic", "/gimbal/enable");

        this->declare_parameter<std::string>("angle_unit", "rad"); // 仅标记
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);

        this->declare_parameter<bool>("default_enable", false);

        // ---- yaw sweep ----
        this->declare_parameter<bool>("yaw_sweep_enable", true);
        this->declare_parameter<bool>("yaw_sweep_add_to_cmd", false);
        this->declare_parameter<double>("yaw_sweep_center", 0.0);
        this->declare_parameter<double>("yaw_sweep_amplitude", 0.6);
        this->declare_parameter<double>("yaw_sweep_period", 2.0);
        this->declare_parameter<double>("yaw_sweep_phase01", 0.0);
        this->declare_parameter<std::string>("yaw_sweep_wave", "sine"); // "sine" | "triangle" | "step"
        this->declare_parameter<double>("yaw_step_dt", 0.05);           // step模式：每隔多少秒跳到下一个离散级
        this->declare_parameter<int>("yaw_step_levels", 9);             // step模式：离散级数(>=2)

        // ---- pitch sweep ----
        this->declare_parameter<bool>("pitch_sweep_enable", true);
        this->declare_parameter<bool>("pitch_sweep_add_to_cmd", false);
        this->declare_parameter<double>("pitch_sweep_center", 0.0);
        this->declare_parameter<double>("pitch_sweep_amplitude", 0.3);
        this->declare_parameter<double>("pitch_sweep_period", 3.0);
        this->declare_parameter<double>("pitch_sweep_phase01", 0.25);
        this->declare_parameter<std::string>("pitch_sweep_wave", "sine"); // "sine" | "triangle" | "step"
        this->declare_parameter<double>("pitch_step_dt", 0.05);
        this->declare_parameter<int>("pitch_step_levels", 9);

        this->declare_parameter<double>("tx_period_ms", 10.0);

        // ---- Lissajous visualization ----
        this->declare_parameter<bool>("lissajous_enable", true);
        this->declare_parameter<std::string>("lissajous_topic", "/gimbal/lissajous_image");
        this->declare_parameter<int>("lissajous_width", 640);
        this->declare_parameter<int>("lissajous_height", 640);
        this->declare_parameter<int>("lissajous_history", 2000);
        this->declare_parameter<double>("lissajous_rate_hz", 30.0);
        this->declare_parameter<double>("lissajous_yaw_range", 0.0);
        this->declare_parameter<double>("lissajous_pitch_range", 0.0);
        this->declare_parameter<bool>("lissajous_draw_axes", true);
    }

    void loadParams()
    {
        serial_port_ = this->get_parameter("serial_port").as_string();
        serial_ports_ = this->get_parameter("serial_ports").as_string_array();
        baud_ = this->get_parameter("baud").as_int();

        angles_topic_ = this->get_parameter("angles_topic").as_string();
        mode_up_topic_ = this->get_parameter("mode_up_topic").as_string();
        cmd_topic_ = this->get_parameter("cmd_topic").as_string();
        enable_topic_ = this->get_parameter("enable_topic").as_string();

        angle_unit_ = this->get_parameter("angle_unit").as_string();
        if (angle_unit_ != "deg") angle_unit_ = "rad";

        yaw_index_ = this->get_parameter("yaw_index").as_int();
        pitch_index_ = this->get_parameter("pitch_index").as_int();

        enable_.store(this->get_parameter("default_enable").as_bool());

        // yaw sweep
        yaw_sweep_enable_ = this->get_parameter("yaw_sweep_enable").as_bool();
        yaw_sweep_add_to_cmd_ = this->get_parameter("yaw_sweep_add_to_cmd").as_bool();
        yaw_sweep_center_ = this->get_parameter("yaw_sweep_center").as_double();
        yaw_sweep_amplitude_ = std::abs(this->get_parameter("yaw_sweep_amplitude").as_double());
        yaw_sweep_period_ = this->get_parameter("yaw_sweep_period").as_double();
        yaw_sweep_phase01_ = this->get_parameter("yaw_sweep_phase01").as_double();
        yaw_sweep_wave_ = this->get_parameter("yaw_sweep_wave").as_string();
        yaw_step_dt_ = this->get_parameter("yaw_step_dt").as_double();
        yaw_step_levels_ = this->get_parameter("yaw_step_levels").as_int();

        // pitch sweep
        pitch_sweep_enable_ = this->get_parameter("pitch_sweep_enable").as_bool();
        pitch_sweep_add_to_cmd_ = this->get_parameter("pitch_sweep_add_to_cmd").as_bool();
        pitch_sweep_center_ = this->get_parameter("pitch_sweep_center").as_double();
        pitch_sweep_amplitude_ = std::abs(this->get_parameter("pitch_sweep_amplitude").as_double());
        pitch_sweep_period_ = this->get_parameter("pitch_sweep_period").as_double();
        pitch_sweep_phase01_ = this->get_parameter("pitch_sweep_phase01").as_double();
        pitch_sweep_wave_ = this->get_parameter("pitch_sweep_wave").as_string();
        pitch_step_dt_ = this->get_parameter("pitch_step_dt").as_double();
        pitch_step_levels_ = this->get_parameter("pitch_step_levels").as_int();

        tx_period_ms_ = this->get_parameter("tx_period_ms").as_double();
        if (tx_period_ms_ < 0.0) tx_period_ms_ = 0.0;

        auto norm_phase = [](double& p) {
            if (p < 0.0) p = 0.0;
            if (p >= 1.0) p = std::fmod(p, 1.0);
        };
        if (yaw_sweep_period_ <= 1e-6) yaw_sweep_period_ = 1.0;
        if (pitch_sweep_period_ <= 1e-6) pitch_sweep_period_ = 1.0;
        norm_phase(yaw_sweep_phase01_);
        norm_phase(pitch_sweep_phase01_);

        if (yaw_step_dt_ <= 1e-6) yaw_step_dt_ = 0.05;
        if (pitch_step_dt_ <= 1e-6) pitch_step_dt_ = 0.05;
        if (yaw_step_levels_ < 2) yaw_step_levels_ = 2;
        if (pitch_step_levels_ < 2) pitch_step_levels_ = 2;

        // lissajous
        lissajous_enable_ = this->get_parameter("lissajous_enable").as_bool();
        lissajous_topic_ = this->get_parameter("lissajous_topic").as_string();
        lissajous_width_ = this->get_parameter("lissajous_width").as_int();
        lissajous_height_ = this->get_parameter("lissajous_height").as_int();
        lissajous_history_ = this->get_parameter("lissajous_history").as_int();
        lissajous_rate_hz_ = this->get_parameter("lissajous_rate_hz").as_double();
        lissajous_yaw_range_ = this->get_parameter("lissajous_yaw_range").as_double();
        lissajous_pitch_range_ = this->get_parameter("lissajous_pitch_range").as_double();
        lissajous_draw_axes_ = this->get_parameter("lissajous_draw_axes").as_bool();

        if (lissajous_width_ < 64) lissajous_width_ = 64;
        if (lissajous_height_ < 64) lissajous_height_ = 64;
        if (lissajous_history_ < 100) lissajous_history_ = 100;
        if (lissajous_rate_hz_ < 1.0) lissajous_rate_hz_ = 1.0;

        if (lissajous_yaw_range_ <= 1e-9) {
            lissajous_yaw_range_ = std::max(1e-3, std::abs(yaw_sweep_center_) + yaw_sweep_amplitude_);
        }
        if (lissajous_pitch_range_ <= 1e-9) {
            lissajous_pitch_range_ = std::max(1e-3, std::abs(pitch_sweep_center_) + pitch_sweep_amplitude_);
        }
    }

    void closeFd()
    {
        std::lock_guard<std::mutex> lk(fd_mutex_);
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        opened_dev_.clear();
    }

    bool ensureOpen()
    {
        std::lock_guard<std::mutex> lk(fd_mutex_);
        if (fd_ >= 0) return true;

        std::vector<std::string> candidates = serial_ports_;
        if (candidates.empty()) candidates.push_back(serial_port_);

        std::string last_err;
        for (const auto& dev : candidates) {
            std::string err;
            int fd = open_serial(dev, baud_, err);
            if (fd >= 0) {
                fd_ = fd;
                opened_dev_ = dev;
                RCLCPP_INFO(this->get_logger(), "Serial opened: %s (baud=%d)", opened_dev_.c_str(), baud_);
                return true;
            }
            last_err = dev + " -> " + err;
        }

        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "Failed to open serial. last_error: %s",
                             last_err.c_str());
        return false;
    }

    void requestTx()
    {
        { std::lock_guard<std::mutex> lk(tx_cv_mtx_); tx_req_seq_++; }
        tx_cv_.notify_one();
    }

    void onEnable(const std_msgs::msg::Bool::SharedPtr msg)
    {
        enable_.store(msg->data);
        requestTx();
    }

    void onCmd(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2) return;
        const int n = static_cast<int>(msg->data.size());
        if (yaw_index_ < 0 || yaw_index_ >= n || pitch_index_ < 0 || pitch_index_ >= n) return;

        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            last_cmd_yaw_   = static_cast<float>(msg->data[static_cast<size_t>(yaw_index_)]);
            last_cmd_pitch_ = static_cast<float>(msg->data[static_cast<size_t>(pitch_index_)]);
            has_cmd_ = true;
        }
        requestTx();
    }

    // unified wave evaluator: returns value in [-1,1]
    double eval_wave(double t,
                     double period, double phase01, const std::string& kind,
                     double step_dt, int step_levels) const
    {
        if (kind == "triangle") return triangle_wave(t, period, phase01);
        if (kind == "step")     return step_wave(t, step_dt, step_levels, phase01);
        return sine_wave(t, period, phase01); // default sine
    }

    void rxLoop()
    {
        std::vector<uint8_t> rxbuf;
        rxbuf.reserve(2048);
        const size_t frame_len = sizeof(GimbalToVision);

        while (!quit_.load()) {
            if (!ensureOpen()) { std::this_thread::sleep_for(200ms); continue; }

            int fd_local = -1;
            { std::lock_guard<std::mutex> lk(fd_mutex_); fd_local = fd_; }
            if (fd_local < 0) continue;

            pollfd pfd{};
            pfd.fd = fd_local;
            pfd.events = POLLIN;

            int pr = ::poll(&pfd, 1, 50);
            if (pr < 0) {
                if (errno == EINTR) continue;
                closeFd();
                continue;
            }
            if (pr == 0) continue;
            if (!(pfd.revents & POLLIN)) continue;

            uint8_t tmp[512];
            ssize_t r = ::read(fd_local, tmp, sizeof(tmp));
            if (r < 0) {
                if (errno == EAGAIN || errno == EINTR) continue;
                closeFd();
                continue;
            }
            if (r == 0) continue;

            rxbuf.insert(rxbuf.end(), tmp, tmp + r);

            while (rxbuf.size() >= 2) {
                size_t pos = 0;
                while (pos + 1 < rxbuf.size()) {
                    if (rxbuf[pos] == 'S' && rxbuf[pos + 1] == 'P') break;
                    pos++;
                }
                if (pos > 0) rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(pos));
                if (rxbuf.size() < frame_len) break;

                uint8_t frame_bytes[sizeof(GimbalToVision)];
                std::memcpy(frame_bytes, rxbuf.data(), frame_len);

                uint16_t crc_calc = crc16_ccitt_reflect_ffff(frame_bytes, frame_len - sizeof(uint16_t));
                uint16_t crc_rx = 0;
                std::memcpy(&crc_rx, frame_bytes + frame_len - sizeof(uint16_t), sizeof(uint16_t));

                if (crc_calc != crc_rx) {
                    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + 1);
                    continue;
                }

                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                     "[RX] %zu bytes: %s",
                                     frame_len, hex_dump(frame_bytes, frame_len).c_str());

                GimbalToVision rx{};
                std::memcpy(&rx, frame_bytes, sizeof(rx));
                rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(frame_len));

                std_msgs::msg::Float64MultiArray angles;
                angles.data.resize(2);
                angles.data[0] = static_cast<double>(rx.yaw);
                angles.data[1] = static_cast<double>(rx.pitch);
                pub_angles_->publish(angles);

                std_msgs::msg::UInt8 m;
                m.data = rx.mode;
                pub_mode_->publish(m);

                requestTx();
            }

            if (rxbuf.size() > 8192) rxbuf.clear();
        }
    }

    void txLoop()
    {
        uint64_t last_seq = 0;
        auto next_tick = std::chrono::steady_clock::now();

        while (!quit_.load()) {
            {
                std::unique_lock<std::mutex> lk(tx_cv_mtx_);
                if (tx_period_ms_ > 0.0) {
                    auto now = std::chrono::steady_clock::now();
                    if (now < next_tick) {
                        tx_cv_.wait_until(lk, next_tick, [&] { return quit_.load() || tx_req_seq_ != last_seq; });
                    }
                } else {
                    tx_cv_.wait_for(lk, 10ms, [&] { return quit_.load() || tx_req_seq_ != last_seq; });
                }
                if (quit_.load()) break;
                if (tx_req_seq_ != last_seq) last_seq = tx_req_seq_;
            }

            if (tx_period_ms_ > 0.0) {
                auto now = std::chrono::steady_clock::now();
                auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double, std::milli>(tx_period_ms_));
                if (period <= 1ms) period = 1ms;
                if (next_tick < now) next_tick = now;
                next_tick += period;
            }

            if (!ensureOpen()) { std::this_thread::sleep_for(200ms); continue; }

            int fd_local = -1;
            { std::lock_guard<std::mutex> lk(fd_mutex_); fd_local = fd_; }
            if (fd_local < 0) continue;

            VisionToGimbal tx{};
            tx.head[0] = 'S';
            tx.head[1] = 'P';
            tx.mode = enable_.load() ? 1 : 0;

            float cmd_yaw = 0.0f, cmd_pitch = 0.0f;
            bool has_cmd = false;
            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                has_cmd = has_cmd_;
                cmd_yaw = last_cmd_yaw_;
                cmd_pitch = last_cmd_pitch_;
            }

            const double t = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_tp_).count();
            const bool enabled = (tx.mode == 1);

            double sweep_yaw = 0.0;
            if (enabled && yaw_sweep_enable_) {
                const double w = eval_wave(t, yaw_sweep_period_, yaw_sweep_phase01_, yaw_sweep_wave_,
                                           yaw_step_dt_, yaw_step_levels_);
                sweep_yaw = yaw_sweep_center_ + yaw_sweep_amplitude_ * w;
            }

            double sweep_pitch = 0.0;
            if (enabled && pitch_sweep_enable_) {
                const double w = eval_wave(t, pitch_sweep_period_, pitch_sweep_phase01_, pitch_sweep_wave_,
                                           pitch_step_dt_, pitch_step_levels_);
                sweep_pitch = pitch_sweep_center_ + pitch_sweep_amplitude_ * w;
            }

            float out_yaw = 0.0f;
            float out_pitch = 0.0f;

            if (enabled) {
                // yaw
                if (yaw_sweep_enable_) {
                    if (has_cmd && yaw_sweep_add_to_cmd_) out_yaw = static_cast<float>(static_cast<double>(cmd_yaw) + sweep_yaw);
                    else                                 out_yaw = static_cast<float>(sweep_yaw);
                } else {
                    out_yaw = has_cmd ? cmd_yaw : 0.0f;
                }

                // pitch
                if (pitch_sweep_enable_) {
                    if (has_cmd && pitch_sweep_add_to_cmd_) out_pitch = static_cast<float>(static_cast<double>(cmd_pitch) + sweep_pitch);
                    else                                    out_pitch = static_cast<float>(sweep_pitch);
                } else {
                    out_pitch = has_cmd ? cmd_pitch : 0.0f;
                }
            } else {
                out_yaw = 0.0f;
                out_pitch = 0.0f;
            }

            tx.yaw = out_yaw;
            tx.pitch = out_pitch;
            tx.yaw_vel = 0.0f;
            tx.yaw_acc = 0.0f;
            tx.pitch_vel = 0.0f;
            tx.pitch_acc = 0.0f;

            // record for lissajous (TX)
            if (lissajous_enable_ && enabled) {
                std::lock_guard<std::mutex> lk(lissajous_mtx_);
                lissajous_pts_.emplace_back(static_cast<double>(out_yaw), static_cast<double>(out_pitch));
                while (static_cast<int>(lissajous_pts_.size()) > lissajous_history_) lissajous_pts_.pop_front();
            }

            tx.crc16 = 0;
            uint8_t tx_bytes[sizeof(VisionToGimbal)];
            std::memcpy(tx_bytes, &tx, sizeof(tx));
            uint16_t tx_crc = crc16_ccitt_reflect_ffff(tx_bytes, sizeof(VisionToGimbal) - sizeof(uint16_t));
            std::memcpy(tx_bytes + sizeof(VisionToGimbal) - sizeof(uint16_t), &tx_crc, sizeof(uint16_t));

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                 "[TX] %zu bytes: %s",
                                 sizeof(tx_bytes), hex_dump(tx_bytes, sizeof(tx_bytes)).c_str());

            if (!write_all_nonblock(fd_local, tx_bytes, sizeof(tx_bytes))) {
                closeFd();
                continue;
            }
        }
    }

    void publishLissajousImage()
    {
        if (!pub_lissajous_img_) return;

        std::deque<std::pair<double,double>> pts;
        {
            std::lock_guard<std::mutex> lk(lissajous_mtx_);
            pts = lissajous_pts_;
        }
        if (pts.size() < 2) return;

        const int w = lissajous_width_;
        const int h = lissajous_height_;
        std::vector<uint8_t> rgb(static_cast<size_t>(w*h*3), 0);

        const double xr = std::max(1e-6, lissajous_yaw_range_);
        const double yr = std::max(1e-6, lissajous_pitch_range_);

        auto to_px = [&](double yaw, double pitch) -> std::pair<int,int> {
            double nx = yaw / xr;
            double ny = pitch / yr;
            nx = std::max(-1.0, std::min(1.0, nx));
            ny = std::max(-1.0, std::min(1.0, ny));
            int x = static_cast<int>(std::lround((nx * 0.5 + 0.5) * (w - 1)));
            int y = static_cast<int>(std::lround(((-ny) * 0.5 + 0.5) * (h - 1)));
            return {x,y};
        };

        if (lissajous_draw_axes_) {
            const int cx = w / 2;
            const int cy = h / 2;
            draw_line(rgb, w, h, 0, cy, w-1, cy, 80, 80, 80);
            draw_line(rgb, w, h, cx, 0, cx, h-1, 80, 80, 80);
        }

        auto [x0,y0] = to_px(pts.front().first, pts.front().second);
        for (size_t i = 1; i < pts.size(); ++i) {
            auto [x1,y1] = to_px(pts[i].first, pts[i].second);
            draw_line(rgb, w, h, x0, y0, x1, y1, 0, 255, 0);
            x0 = x1; y0 = y1;
        }

        auto [xl,yl] = to_px(pts.back().first, pts.back().second);
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                set_pixel_rgb(rgb, w, h, xl+dx, yl+dy, 255, 0, 0);

        sensor_msgs::msg::Image msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "gimbal";
        msg.height = static_cast<uint32_t>(h);
        msg.width  = static_cast<uint32_t>(w);
        msg.encoding = "rgb8";
        msg.is_bigendian = false;
        msg.step = static_cast<uint32_t>(w * 3);
        msg.data = std::move(rgb);

        pub_lissajous_img_->publish(std::move(msg));
    }

private:
    // serial
    std::string              serial_port_;
    std::vector<std::string> serial_ports_;
    int                      baud_{115200};
    int                      fd_{-1};
    std::string              opened_dev_;
    std::mutex               fd_mutex_;

    std::thread              rx_thread_;
    std::thread              tx_thread_;
    std::atomic<bool>        quit_{false};

    // topics
    std::string angles_topic_;
    std::string mode_up_topic_;
    std::string cmd_topic_;
    std::string enable_topic_;

    std::string angle_unit_{"rad"};
    int         yaw_index_{0};
    int         pitch_index_{1};

    std::atomic<bool> enable_{false};
    std::mutex        cmd_mutex_;
    bool              has_cmd_{false};
    float             last_cmd_yaw_{0.0f};
    float             last_cmd_pitch_{0.0f};

    // yaw sweep
    bool   yaw_sweep_enable_{true};
    bool   yaw_sweep_add_to_cmd_{false};
    double yaw_sweep_center_{0.0};
    double yaw_sweep_amplitude_{0.6};
    double yaw_sweep_period_{2.0};
    double yaw_sweep_phase01_{0.0};
    std::string yaw_sweep_wave_{"sine"};
    double yaw_step_dt_{0.05};
    int    yaw_step_levels_{9};

    // pitch sweep
    bool   pitch_sweep_enable_{true};
    bool   pitch_sweep_add_to_cmd_{false};
    double pitch_sweep_center_{0.0};
    double pitch_sweep_amplitude_{0.3};
    double pitch_sweep_period_{3.0};
    double pitch_sweep_phase01_{0.25};
    std::string pitch_sweep_wave_{"sine"};
    double pitch_step_dt_{0.05};
    int    pitch_step_levels_{9};

    double tx_period_ms_{10.0};
    std::chrono::steady_clock::time_point start_tp_{std::chrono::steady_clock::now()};

    // TX wakeup
    std::condition_variable tx_cv_;
    std::mutex              tx_cv_mtx_;
    uint64_t                tx_req_seq_{0};

    // ROS pubs/subs
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_angles_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr            pub_mode_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              sub_enable_;

    // lissajous
    bool lissajous_enable_{true};
    std::string lissajous_topic_{"/gimbal/lissajous_image"};
    int lissajous_width_{640};
    int lissajous_height_{640};
    int lissajous_history_{2000};
    double lissajous_rate_hz_{30.0};
    double lissajous_yaw_range_{0.0};
    double lissajous_pitch_range_{0.0};
    bool lissajous_draw_axes_{true};

    std::mutex lissajous_mtx_;
    std::deque<std::pair<double,double>> lissajous_pts_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_lissajous_img_;
    rclcpp::TimerBase::SharedPtr lissajous_timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalSerialBridge>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
