#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>

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

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <poll.h>
#include <iomanip>
#include <sstream>

using namespace std::chrono_literals;

namespace {
#pragma pack(push, 1)
struct GimbalToVision {
    uint8_t  head[2];  // 'S','P'
    uint8_t  mode;     // {0,1} in laser project
    float    q[4];     // wxyz (ignored)
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
    uint8_t  mode;     // {0,1} in laser project
    float    yaw;
    float    yaw_vel;
    float    yaw_acc;
    float    pitch;
    float    pitch_vel;
    float    pitch_acc;
    uint16_t crc16;
};
#pragma pack(pop)

static_assert(sizeof(GimbalToVision) <= 64, "GimbalToVision too large");
static_assert(sizeof(VisionToGimbal) <= 64, "VisionToGimbal too large");

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

    // 低延迟建议：不要 O_SYNC；使用 NONBLOCK + poll
    int fd = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        err_out = "open() failed: errno=" + std::to_string(errno) + " " +
                  std::string(std::strerror(errno));
        return -1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        err_out = "tcgetattr() failed: errno=" + std::to_string(errno) + " " +
                  std::string(std::strerror(errno));
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

    // 非阻塞读：交给 poll() 控制等待
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (!set_baudrate(tty, baud)) {
        err_out = "unsupported baud: " + std::to_string(baud);
        ::close(fd);
        return -1;
    }

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        err_out = "tcsetattr() failed: errno=" + std::to_string(errno) + " " +
                  std::string(std::strerror(errno));
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
        if (w > 0) {
            sent += static_cast<size_t>(w);
            continue;
        }
        if (w < 0 && (errno == EAGAIN || errno == EINTR)) {
            // 让出一点点调度，但不 sleep 很久
            std::this_thread::yield();
            continue;
        }
        return false;
    }
    return true;
}

}  // namespace

class GimbalSerialBridge final : public rclcpp::Node {
public:
    explicit GimbalSerialBridge(const rclcpp::NodeOptions& opts)
        : rclcpp::Node("gimbal_serial_bridge", opts)
    {
        declareParams();
        loadParams();

        pub_angles_ =
            this->create_publisher<std_msgs::msg::Float64MultiArray>(
                angles_topic_, rclcpp::SensorDataQoS());
        pub_mode_ = this->create_publisher<std_msgs::msg::UInt8>(
            mode_up_topic_, rclcpp::QoS(10));

        sub_cmd_ =
            this->create_subscription<std_msgs::msg::Float64MultiArray>(
                cmd_topic_, rclcpp::QoS(10),
                std::bind(&GimbalSerialBridge::onCmd, this,
                          std::placeholders::_1));

        sub_enable_ = this->create_subscription<std_msgs::msg::Bool>(
            enable_topic_, rclcpp::QoS(10),
            std::bind(&GimbalSerialBridge::onEnable, this,
                      std::placeholders::_1));

        rx_thread_ = std::thread(&GimbalSerialBridge::rxLoop, this);
        tx_thread_ = std::thread(&GimbalSerialBridge::txLoop, this);

        RCLCPP_INFO(
            this->get_logger(),
            "Started. baud=%d angles_topic=%s cmd_topic=%s enable_topic=%s",
            baud_, angles_topic_.c_str(), cmd_topic_.c_str(),
            enable_topic_.c_str());
    }

    ~GimbalSerialBridge() override
    {
        quit_.store(true);

        // 唤醒 TX 线程退出
        {
            std::lock_guard<std::mutex> lk(tx_cv_mtx_);
            tx_req_seq_++;
        }
        tx_cv_.notify_all();

        if (rx_thread_.joinable()) rx_thread_.join();
        if (tx_thread_.joinable()) tx_thread_.join();

        closeFd();
    }

private:
    void declareParams()
    {
        this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
        this->declare_parameter<std::vector<std::string>>(
            "serial_ports", std::vector<std::string>{});
        this->declare_parameter<int>("baud", 115200);

        this->declare_parameter<std::string>("angles_topic", "/gimbal/angles");
        this->declare_parameter<std::string>("mode_up_topic", "/gimbal/mode_up");
        this->declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
        this->declare_parameter<std::string>("enable_topic", "/gimbal/enable");

        this->declare_parameter<std::string>("angle_unit", "rad"); // rad/deg
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);

        this->declare_parameter<bool>("default_enable", false);
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
    }

    void closeFd()
    {
        std::lock_guard<std::mutex> lk(fd_mutex_);
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
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
                RCLCPP_INFO(this->get_logger(),
                            "Serial opened: %s (baud=%d)",
                            opened_dev_.c_str(), baud_);
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
        {
            std::lock_guard<std::mutex> lk(tx_cv_mtx_);
            tx_req_seq_++;
        }
        tx_cv_.notify_one();
    }

    void onEnable(const std_msgs::msg::Bool::SharedPtr msg)
    {
        enable_.store(msg->data);
        requestTx();  // enable 变化，立即发一帧
    }

    void onCmd(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2) return;
        const int n = static_cast<int>(msg->data.size());
        if (yaw_index_ < 0 || yaw_index_ >= n || pitch_index_ < 0 || pitch_index_ >= n)
            return;

        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            last_cmd_yaw_ = static_cast<float>(msg->data[static_cast<size_t>(yaw_index_)]);
            last_cmd_pitch_ = static_cast<float>(msg->data[static_cast<size_t>(pitch_index_)]);
            has_cmd_ = true;
        }
        requestTx();  // 新命令立刻触发 TX
    }

    // -----------------------------
    // RX：只读+解析+发布（不写串口）
    // -----------------------------
    void rxLoop()
    {
        std::vector<uint8_t> rxbuf;
        rxbuf.reserve(2048);

        const size_t frame_len = sizeof(GimbalToVision);

        while (!quit_.load()) {
            if (!ensureOpen()) {
                std::this_thread::sleep_for(200ms);
                continue;
            }

            int fd_local = -1;
            {
                std::lock_guard<std::mutex> lk(fd_mutex_);
                fd_local = fd_;
            }
            if (fd_local < 0) continue;

            pollfd pfd{};
            pfd.fd = fd_local;
            pfd.events = POLLIN;

            int pr = ::poll(&pfd, 1, 50);  // 50ms 最大等待
            if (pr < 0) {
                if (errno == EINTR) continue;
                closeFd();
                continue;
            }
            if (pr == 0) continue;  // timeout
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

            // 流解析：扫描头 "S P"
            while (rxbuf.size() >= 2) {
                size_t pos = 0;
                while (pos + 1 < rxbuf.size()) {
                    if (rxbuf[pos] == 'S' && rxbuf[pos + 1] == 'P') break;
                    pos++;
                }
                if (pos > 0) {
                    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(pos));
                }
                if (rxbuf.size() < frame_len) break;

                // 取一帧
                uint8_t frame_bytes[sizeof(GimbalToVision)];
                std::memcpy(frame_bytes, rxbuf.data(), frame_len);

                // CRC 校验
                uint16_t crc_calc = crc16_ccitt_reflect_ffff(
                    frame_bytes, frame_len - sizeof(uint16_t));

                uint16_t crc_rx = 0;
                std::memcpy(&crc_rx,
                            frame_bytes + frame_len - sizeof(uint16_t),
                            sizeof(uint16_t));

                if (crc_calc != crc_rx) {
                    // CRC 错：丢一个字节继续重同步（非常关键）
                    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + 1);
                    continue;
                }

                // ✅ CRC OK：先打印 RX 原始帧，再解析
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(), *this->get_clock(), 500 /*ms*/,
                    "[RX] %zu bytes: %s",
                    frame_len,
                    hex_dump(frame_bytes, frame_len).c_str());

                // CRC OK：解析并消费
                GimbalToVision rx{};
                std::memcpy(&rx, frame_bytes, sizeof(rx));
                rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(frame_len));

                // publish angles (yaw,pitch)
                std_msgs::msg::Float64MultiArray angles;
                angles.data.resize(2);
                angles.data[0] = static_cast<double>(rx.yaw);
                angles.data[1] = static_cast<double>(rx.pitch);
                pub_angles_->publish(angles);

                // publish mode_up
                std_msgs::msg::UInt8 m;
                m.data = rx.mode;
                pub_mode_->publish(m);

                // 收到有效帧，也触发 TX（保证链路连续）
                requestTx();
            }

            // 防止无限增长（异常情况）
            if (rxbuf.size() > 8192) {
                rxbuf.clear();
            }
        }
    }

    // -----------------------------
    // TX：只组帧+发送（不读串口）
    // -----------------------------
    void txLoop()
    {
        uint64_t last_seq = 0;

        while (!quit_.load()) {
            // 等待触发（命令/enable变化/收到RX有效帧）
            {
                std::unique_lock<std::mutex> lk(tx_cv_mtx_);
                tx_cv_.wait_for(lk, 10ms, [&] {
                    return quit_.load() || tx_req_seq_ != last_seq;
                });
                if (quit_.load()) break;
                last_seq = tx_req_seq_;
            }

            if (!ensureOpen()) {
                std::this_thread::sleep_for(200ms);
                continue;
            }

            int fd_local = -1;
            {
                std::lock_guard<std::mutex> lk(fd_mutex_);
                fd_local = fd_;
            }
            if (fd_local < 0) continue;

            // 组帧：永远发“最新状态”，不等待 RX
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

            if (has_cmd && tx.mode == 1) {
                tx.yaw = cmd_yaw;
                tx.pitch = cmd_pitch;
            } else {
                tx.yaw = 0.0f;
                tx.pitch = 0.0f;
            }

            tx.yaw_vel = 0.0f;
            tx.yaw_acc = 0.0f;
            tx.pitch_vel = 0.0f;
            tx.pitch_acc = 0.0f;

            // CRC
            tx.crc16 = 0;
            uint8_t tx_bytes[sizeof(VisionToGimbal)];
            std::memcpy(tx_bytes, &tx, sizeof(tx));
            uint16_t tx_crc = crc16_ccitt_reflect_ffff(
                tx_bytes, sizeof(VisionToGimbal) - sizeof(uint16_t));
            std::memcpy(tx_bytes + sizeof(VisionToGimbal) - sizeof(uint16_t),
                        &tx_crc, sizeof(uint16_t));

            // 低频日志，避免影响实时性
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 500 /*ms*/,
                "[TX] %zu bytes: %s",
                sizeof(tx_bytes),
                hex_dump(tx_bytes, sizeof(tx_bytes)).c_str());

            if (!write_all_nonblock(fd_local, tx_bytes, sizeof(tx_bytes))) {
                closeFd();
                continue;
            }
        }
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

    // TX wakeup
    std::condition_variable tx_cv_;
    std::mutex              tx_cv_mtx_;
    uint64_t                tx_req_seq_{0};

    // ROS
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_angles_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr            pub_mode_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              sub_enable_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalSerialBridge>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
