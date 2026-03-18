#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <termios.h>
#include <unistd.h>
#include <poll.h>

using namespace std::chrono_literals;

class ManualAngleCmdNode final : public rclcpp::Node
{
public:
    explicit ManualAngleCmdNode(const rclcpp::NodeOptions & options)
        : Node("manual_angle_cmd_node", options)
    {
        // 改为手动累计量输出话题，不再直发最终控制话题
        declare_parameter<std::string>("cmd_topic", "/gimbal/command_angles");
        declare_parameter<std::string>("enable_topic", "/gimbal/enable");

        declare_parameter<double>("yaw_deg", 0.0);
        declare_parameter<double>("pitch_deg", 0.0);
        declare_parameter<double>("step_deg", 0.1);

        declare_parameter<int>("yaw_index", 0);
        declare_parameter<int>("pitch_index", 1);
        declare_parameter<int>("publish_dim", 2);

        declare_parameter<bool>("publish_enable", true);
        declare_parameter<bool>("enable_value", true);
        declare_parameter<double>("publish_hz", 20.0);

        cmd_topic_ = get_parameter("cmd_topic").as_string();
        enable_topic_ = get_parameter("enable_topic").as_string();

        yaw_deg_ = get_parameter("yaw_deg").as_double();
        pitch_deg_ = get_parameter("pitch_deg").as_double();
        step_deg_ = get_parameter("step_deg").as_double();

        yaw_index_ = get_parameter("yaw_index").as_int();
        pitch_index_ = get_parameter("pitch_index").as_int();
        publish_dim_ = get_parameter("publish_dim").as_int();

        publish_enable_ = get_parameter("publish_enable").as_bool();
        enable_value_ = get_parameter("enable_value").as_bool();

        const double publish_hz = get_parameter("publish_hz").as_double();

        if (publish_dim_ < 2) {
            publish_dim_ = 2;
        }
        if (yaw_index_ < 0) yaw_index_ = 0;
        if (pitch_index_ < 0) pitch_index_ = 1;
        if (yaw_index_ >= publish_dim_ || pitch_index_ >= publish_dim_) {
            RCLCPP_FATAL(
                get_logger(),
                "Invalid yaw_index/pitch_index: yaw_index=%d pitch_index=%d publish_dim=%d",
                yaw_index_, pitch_index_, publish_dim_);
            throw std::runtime_error("invalid index configuration");
        }
        if (step_deg_ <= 0.0) {
            step_deg_ = 0.1;
        }

        pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic_, rclcpp::QoS(10));
        if (publish_enable_) {
            pub_enable_ = create_publisher<std_msgs::msg::Bool>(enable_topic_, rclcpp::QoS(10));
        }

        publishOnce();

        const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_hz));
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(period),
            std::bind(&ManualAngleCmdNode::onTimer, this));

        kb_thread_ = std::thread(&ManualAngleCmdNode::keyboardLoop, this);

        RCLCPP_INFO(
            get_logger(),
            "manual_angle_cmd_node started. cmd_topic=%s enable_topic=%s yaw=%.3f deg pitch=%.3f deg step=%.3f deg",
            cmd_topic_.c_str(), enable_topic_.c_str(), yaw_deg_, pitch_deg_, step_deg_);

        RCLCPP_INFO(
            get_logger(),
            "Keyboard control enabled in current terminal: W=+pitch S=-pitch A=-yaw D=+yaw (STRICT uppercase only), Q=quit keyboard thread");
    }

    ~ManualAngleCmdNode() override
    {
        quit_.store(true);
        if (kb_thread_.joinable()) {
            kb_thread_.join();
        }
    }

private:
    void onTimer()
    {
        publishOnce();
    }

    void publishOnce()
    {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(static_cast<size_t>(publish_dim_), 0.0);

        double yaw = 0.0;
        double pitch = 0.0;
        {
            std::lock_guard<std::mutex> lk(angle_mutex_);
            yaw = yaw_deg_;
            pitch = pitch_deg_;
        }

        msg.data[static_cast<size_t>(yaw_index_)] = yaw;
        msg.data[static_cast<size_t>(pitch_index_)] = pitch;
        pub_cmd_->publish(msg);

        if (publish_enable_ && pub_enable_) {
            std_msgs::msg::Bool en;
            en.data = enable_value_;
            pub_enable_->publish(en);
        }
    }

    void keyboardLoop()
    {
        if (!::isatty(STDIN_FILENO)) {
            RCLCPP_WARN(
                get_logger(),
                "STDIN is not a TTY. Keyboard AWSD control is disabled. Start this node in a foreground terminal.");
            return;
        }

        termios oldt {};
        if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
            RCLCPP_WARN(get_logger(), "tcgetattr failed, keyboard control disabled.");
            return;
        }

        termios newt = oldt;
        newt.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
        newt.c_cc[VMIN] = 0;
        newt.c_cc[VTIME] = 0;

        if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) != 0) {
            RCLCPP_WARN(get_logger(), "tcsetattr failed, keyboard control disabled.");
            return;
        }

        auto restore = [&]() {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        };

        while (!quit_.load()) {
            pollfd pfd {};
            pfd.fd = STDIN_FILENO;
            pfd.events = POLLIN;

            const int pr = ::poll(&pfd, 1, 100);
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            if (pr == 0) {
                continue;
            }
            if (!(pfd.revents & POLLIN)) {
                continue;
            }

            char ch = 0;
            const ssize_t n = ::read(STDIN_FILENO, &ch, 1);
            if (n <= 0) {
                continue;
            }

            bool changed = false;
            double yaw = 0.0;
            double pitch = 0.0;

            {
                std::lock_guard<std::mutex> lk(angle_mutex_);
                switch (ch) {
                    case 'A':
                        yaw_deg_ += step_deg_;
                        changed = true;
                        break;
                    case 'D':
                        yaw_deg_ -= step_deg_;
                        changed = true;
                        break;
                    case 'W':
                        pitch_deg_ += step_deg_;
                        changed = true;
                        break;
                    case 'S':
                        pitch_deg_ -= step_deg_;
                        changed = true;
                        break;
                    case 'Q':
                        RCLCPP_INFO(get_logger(), "Keyboard thread exit requested by 'Q'.");
                        restore();
                        return;
                    default:
                        break;
                }
                yaw = yaw_deg_;
                pitch = pitch_deg_;
            }

            if (changed) {
                publishOnce();
                RCLCPP_INFO(
                    get_logger(),
                    "Updated by key '%c': yaw=%.3f deg, pitch=%.3f deg",
                    ch, yaw, pitch);
            }
        }

        restore();
    }

private:
    std::string cmd_topic_;
    std::string enable_topic_;

    double yaw_deg_{0.0};
    double pitch_deg_{0.0};
    double step_deg_{0.1};

    int yaw_index_{0};
    int pitch_index_{1};
    int publish_dim_{2};

    bool publish_enable_{true};
    bool enable_value_{true};

    std::mutex angle_mutex_;
    std::atomic<bool> quit_{false};

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_enable_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::thread kb_thread_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ManualAngleCmdNode>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}