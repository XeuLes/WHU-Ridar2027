#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>

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
#include <cmath>
#include <limits>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <poll.h>
#include <iomanip>
#include <sstream>

using namespace std::chrono_literals;

namespace {

struct QuatWXYZ {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

#pragma pack(push, 1)
struct GimbalToVision {
    uint8_t  head[2];
    uint8_t  mode;
    float    q[4];            // wxyz, drifted/raw quaternion from MCU
    float    yaw;             // drifted/raw yaw from MCU
    float    yaw_vel;
    float    pitch;
    float    pitch_vel;
    float    bullet_speed;
    uint16_t bullet_count;
    uint32_t running_time;    // ms
    uint16_t crc16;
};

struct VisionToGimbal {
    uint8_t  head[2];
    uint8_t  mode;
    float    yaw;             // drifted yaw command sent to MCU
    float    yaw_vel;
    float    yaw_acc;
    float    pitch;
    float    pitch_vel;
    float    pitch_acc;
    uint16_t crc16;           // unchanged
};
#pragma pack(pop)

static_assert(sizeof(GimbalToVision) <= 64, "GimbalToVision too large");
static_assert(sizeof(VisionToGimbal) <= 64, "VisionToGimbal too large");

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

static QuatWXYZ quat_from_euler(double roll, double pitch, double yaw)
{
    const double cr = std::cos(roll * 0.5);
    const double sr = std::sin(roll * 0.5);
    const double cp = std::cos(pitch * 0.5);
    const double sp = std::sin(pitch * 0.5);
    const double cy = std::cos(yaw * 0.5);
    const double sy = std::sin(yaw * 0.5);

    QuatWXYZ q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return q;
}

static void quat_to_euler(const QuatWXYZ& q, double& roll, double& pitch, double& yaw)
{
    roll = std::atan2(
        2.0 * (q.w * q.x + q.y * q.z),
        1.0 - 2.0 * (q.x * q.x + q.y * q.y));

    double s = 2.0 * (q.w * q.y - q.z * q.x);
    if (s > 1.0) s = 1.0;
    if (s < -1.0) s = -1.0;
    pitch = std::asin(s);

    yaw = std::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));
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

        pub_angles_raw_ =
            this->create_publisher<std_msgs::msg::Float64MultiArray>(
                angles_raw_topic_, rclcpp::SensorDataQoS());
        pub_angles_true_ =
            this->create_publisher<std_msgs::msg::Float64MultiArray>(
                angles_true_topic_, rclcpp::SensorDataQoS());

        pub_quaternion_raw_ =
            this->create_publisher<geometry_msgs::msg::QuaternionStamped>(
                quaternion_raw_topic_, rclcpp::SensorDataQoS());
        pub_quaternion_true_ =
            this->create_publisher<geometry_msgs::msg::QuaternionStamped>(
                quaternion_true_topic_, rclcpp::SensorDataQoS());

        pub_mode_ = this->create_publisher<std_msgs::msg::UInt8>(
            mode_up_topic_, rclcpp::QoS(10));

        sub_cmd_ =
            this->create_subscription<std_msgs::msg::Float64MultiArray>(
                cmd_topic_, rclcpp::QoS(10),
                std::bind(&GimbalSerialBridge::onCmd, this, std::placeholders::_1));

        sub_enable_ = this->create_subscription<std_msgs::msg::Bool>(
            enable_topic_, rclcpp::QoS(10),
            std::bind(&GimbalSerialBridge::onEnable, this, std::placeholders::_1));

        rx_thread_ = std::thread(&GimbalSerialBridge::rxLoop, this);
        tx_thread_ = std::thread(&GimbalSerialBridge::txLoop, this);

        RCLCPP_INFO(
            this->get_logger(),
            "Started. baud=%d cmd_topic=%s enable_topic=%s yaw_drift_rate=%.9f",
            baud_, cmd_topic_.c_str(), enable_topic_.c_str(), yaw_drift_rate_);
        RCLCPP_INFO(
            this->get_logger(),
            "Publish topics: angles_raw=%s angles_true=%s quaternion_raw=%s quaternion_true=%s mode=%s",
            angles_raw_topic_.c_str(),
            angles_true_topic_.c_str(),
            quaternion_raw_topic_.c_str(),
            quaternion_true_topic_.c_str(),
            mode_up_topic_.c_str());
    }

    ~GimbalSerialBridge() override
    {
        quit_.store(true);

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
        this->declare_parameter<std::string>("serial_port", "/dev/ttyACM0");
        this->declare_parameter<std::vector<std::string>>(
            "serial_ports", std::vector<std::string>{});
        this->declare_parameter<int>("baud", 115200);

        this->declare_parameter<std::string>("angles_raw_topic", "/gimbal/angles_raw");
        this->declare_parameter<std::string>("angles_true_topic", "/gimbal/angles_true");
        this->declare_parameter<std::string>("quaternion_raw_topic", "/gimbal/quaternion_raw");
        this->declare_parameter<std::string>("quaternion_true_topic", "/gimbal/quaternion_true");
        this->declare_parameter<std::string>("mode_up_topic", "/gimbal/mode_up");

        this->declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
        this->declare_parameter<std::string>("enable_topic", "/gimbal/enable");

        this->declare_parameter<std::string>("angle_unit", "rad");
        this->declare_parameter<int>("yaw_index", 0);
        this->declare_parameter<int>("pitch_index", 1);

        this->declare_parameter<bool>("default_enable", true);

        // same unit as yaw per second
        this->declare_parameter<double>("yaw_drift_rate", 0.05);
    }

    void loadParams()
    {
        serial_port_ = this->get_parameter("serial_port").as_string();
        serial_ports_ = this->get_parameter("serial_ports").as_string_array();
        baud_ = this->get_parameter("baud").as_int();

        angles_raw_topic_ = this->get_parameter("angles_raw_topic").as_string();
        angles_true_topic_ = this->get_parameter("angles_true_topic").as_string();
        quaternion_raw_topic_ = this->get_parameter("quaternion_raw_topic").as_string();
        quaternion_true_topic_ = this->get_parameter("quaternion_true_topic").as_string();
        mode_up_topic_ = this->get_parameter("mode_up_topic").as_string();

        cmd_topic_ = this->get_parameter("cmd_topic").as_string();
        enable_topic_ = this->get_parameter("enable_topic").as_string();

        angle_unit_ = this->get_parameter("angle_unit").as_string();
        if (angle_unit_ != "deg") angle_unit_ = "rad";

        yaw_index_ = this->get_parameter("yaw_index").as_int();
        pitch_index_ = this->get_parameter("pitch_index").as_int();

        enable_.store(this->get_parameter("default_enable").as_bool());
        yaw_drift_rate_ = this->get_parameter("yaw_drift_rate").as_double();
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
        requestTx();
    }

    void onCmd(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2) return;
        const int n = static_cast<int>(msg->data.size());
        if (yaw_index_ < 0 || yaw_index_ >= n || pitch_index_ < 0 || pitch_index_ >= n) {
            return;
        }

        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            last_cmd_yaw_true_ =
                static_cast<float>(msg->data[static_cast<size_t>(yaw_index_)]);
            last_cmd_pitch_ =
                static_cast<float>(msg->data[static_cast<size_t>(pitch_index_)]);
            has_cmd_ = true;
        }
        requestTx();
    }

    void updateLastRxTiming(uint32_t running_time_ms)
    {
        std::lock_guard<std::mutex> lk(rx_state_mutex_);
        last_mcu_running_time_ms_ = running_time_ms;
        last_rx_host_tp_ = std::chrono::steady_clock::now();
        has_last_mcu_time_ = true;
    }

    double computeTxDriftSeconds()
    {
        std::lock_guard<std::mutex> lk(rx_state_mutex_);
        if (!has_last_mcu_time_) {
            return 0.0;
        }

        const auto now = std::chrono::steady_clock::now();
        const double t_mcu = static_cast<double>(last_mcu_running_time_ms_) / 1000.0;
        const double t_local_delay =
            std::chrono::duration<double>(now - last_rx_host_tp_).count();

        return t_mcu + t_local_delay;
    }

    void publishRawAndTrue(
        const GimbalToVision& rx,
        double yaw_true,
        const QuatWXYZ& q_true)
    {
        // raw angles
        std_msgs::msg::Float64MultiArray angles_raw;
        angles_raw.data.resize(2);
        angles_raw.data[0] = static_cast<double>(rx.yaw);
        angles_raw.data[1] = static_cast<double>(rx.pitch);
        pub_angles_raw_->publish(angles_raw);

        // true angles
        std_msgs::msg::Float64MultiArray angles_true;
        angles_true.data.resize(2);
        angles_true.data[0] = yaw_true;
        angles_true.data[1] = static_cast<double>(rx.pitch);
        pub_angles_true_->publish(angles_true);

        // raw quaternion
        geometry_msgs::msg::QuaternionStamped qmsg_raw;
        qmsg_raw.header.stamp = this->now();
        qmsg_raw.header.frame_id = "gimbal_frame";
        qmsg_raw.quaternion.w = static_cast<double>(rx.q[0]);
        qmsg_raw.quaternion.x = static_cast<double>(rx.q[1]);
        qmsg_raw.quaternion.y = static_cast<double>(rx.q[2]);
        qmsg_raw.quaternion.z = static_cast<double>(rx.q[3]);
        pub_quaternion_raw_->publish(qmsg_raw);

        // true quaternion
        geometry_msgs::msg::QuaternionStamped qmsg_true;
        qmsg_true.header.stamp = qmsg_raw.header.stamp;
        qmsg_true.header.frame_id = "gimbal_frame";
        qmsg_true.quaternion.w = q_true.w;
        qmsg_true.quaternion.x = q_true.x;
        qmsg_true.quaternion.y = q_true.y;
        qmsg_true.quaternion.z = q_true.z;
        pub_quaternion_true_->publish(qmsg_true);
    }

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
                if (pos > 0) {
                    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(pos));
                }
                if (rxbuf.size() < frame_len) break;

                uint8_t frame_bytes[sizeof(GimbalToVision)];
                std::memcpy(frame_bytes, rxbuf.data(), frame_len);

                uint16_t crc_calc = crc16_ccitt_reflect_ffff(
                    frame_bytes, frame_len - sizeof(uint16_t));

                uint16_t crc_rx = 0;
                std::memcpy(
                    &crc_rx,
                    frame_bytes + frame_len - sizeof(uint16_t),
                    sizeof(uint16_t));

                if (crc_calc != crc_rx) {
                    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + 1);
                    continue;
                }

                GimbalToVision rx{};
                std::memcpy(&rx, frame_bytes, sizeof(rx));
                rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<long>(frame_len));

                updateLastRxTiming(rx.running_time);

                // raw/drifted yaw from MCU -> true yaw on host
                const double t_mcu = static_cast<double>(rx.running_time) / 1000.0;
                const double yaw_true =
                    static_cast<double>(rx.yaw) - yaw_drift_rate_ * t_mcu;

                // raw quaternion -> true quaternion by replacing yaw
                QuatWXYZ q_raw;
                q_raw.w = static_cast<double>(rx.q[0]);
                q_raw.x = static_cast<double>(rx.q[1]);
                q_raw.y = static_cast<double>(rx.q[2]);
                q_raw.z = static_cast<double>(rx.q[3]);

                // 如果协议中真正可信的双轴角度来源是 rx.yaw / rx.pitch，
                // 那 true quaternion 应直接由补偿后的 yaw_true 和 rx.pitch 重建
                const double roll_true = 0.0;
                const double pitch_true = static_cast<double>(rx.pitch);
                QuatWXYZ q_true = quat_from_euler(roll_true, pitch_true, yaw_true);

                publishRawAndTrue(rx, yaw_true, q_true);

                std_msgs::msg::UInt8 m;
                m.data = rx.mode;
                pub_mode_->publish(m);

                // RCLCPP_INFO_THROTTLE(
                //     this->get_logger(), *this->get_clock(), 500,
                //     "RX mode=%u yaw_raw=%.6f yaw_true=%.6f pitch=%.6f running_time=%u ms",
                //     static_cast<unsigned>(rx.mode),
                //     static_cast<double>(rx.yaw),
                //     yaw_true,
                //     static_cast<double>(rx.pitch),
                //     static_cast<unsigned>(rx.running_time));

                requestTx();
            }

            if (rxbuf.size() > 8192) {
                rxbuf.clear();
            }
        }
    }

    void txLoop()
    {
        uint64_t last_seq = 0;

        while (!quit_.load()) {
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

            VisionToGimbal tx{};
            tx.head[0] = 'S';
            tx.head[1] = 'P';
            tx.mode = enable_.load() ? 1 : 0;

            float cmd_yaw_true = 0.0f;
            float cmd_pitch = 0.0f;
            bool has_cmd = false;
            {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                has_cmd = has_cmd_;
                cmd_yaw_true = last_cmd_yaw_true_;
                cmd_pitch = last_cmd_pitch_;
            }

            const float base_yaw_true = has_cmd ? cmd_yaw_true : 0.0f;
            const float base_pitch = has_cmd ? cmd_pitch : 0.0f;

            // host true yaw -> drifted yaw for MCU
            const double t_total_sec = computeTxDriftSeconds();
            const float yaw_drift =
                static_cast<float>(yaw_drift_rate_ * t_total_sec);

            if (tx.mode == 1) {
                tx.yaw = base_yaw_true + yaw_drift;
                tx.pitch = base_pitch;
            } else {
                tx.yaw = 0.0f;
                tx.pitch = 0.0f;
            }

            tx.yaw_vel = 0.0f;
            tx.yaw_acc = 0.0f;
            tx.pitch_vel = 0.0f;
            tx.pitch_acc = 0.0f;

            tx.crc16 = 0;
            uint8_t tx_bytes[sizeof(VisionToGimbal)];
            std::memcpy(tx_bytes, &tx, sizeof(tx));

            const uint16_t tx_crc = crc16_ccitt_reflect_ffff(
                tx_bytes, sizeof(VisionToGimbal) - sizeof(uint16_t));
            std::memcpy(
                tx_bytes + sizeof(VisionToGimbal) - sizeof(uint16_t),
                &tx_crc, sizeof(uint16_t));

            if (!write_all_nonblock(fd_local, tx_bytes, sizeof(tx_bytes))) {
                closeFd();
                continue;
            }

            // RCLCPP_INFO_THROTTLE(
            //     this->get_logger(), *this->get_clock(), 500,
            //     "TX mode=%u yaw_cmd_true=%.6f yaw_tx_drifted=%.6f pitch=%.6f t_total=%.6f",
            //     static_cast<unsigned>(tx.mode),
            //     static_cast<double>(base_yaw_true),
            //     static_cast<double>(tx.yaw),
            //     static_cast<double>(tx.pitch),
            //     t_total_sec);
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
    std::string angles_raw_topic_;
    std::string angles_true_topic_;
    std::string quaternion_raw_topic_;
    std::string quaternion_true_topic_;
    std::string mode_up_topic_;
    std::string cmd_topic_;
    std::string enable_topic_;

    std::string angle_unit_{"rad"};
    int         yaw_index_{0};
    int         pitch_index_{1};

    std::atomic<bool> enable_{false};
    std::mutex        cmd_mutex_;
    bool              has_cmd_{false};
    float             last_cmd_yaw_true_{0.0f};
    float             last_cmd_pitch_{0.0f};

    // latest RX timing state for TX reverse compensation
    std::mutex rx_state_mutex_;
    uint32_t last_mcu_running_time_ms_{0};
    bool has_last_mcu_time_{false};
    std::chrono::steady_clock::time_point last_rx_host_tp_{};

    // TX wakeup
    std::condition_variable tx_cv_;
    std::mutex              tx_cv_mtx_;
    uint64_t                tx_req_seq_{0};

    // same unit as yaw per second
    double yaw_drift_rate_{0.0};

    // ROS publishers/subscribers
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_angles_raw_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_angles_true_;
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub_quaternion_raw_;
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub_quaternion_true_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_mode_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_enable_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalSerialBridge>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}