#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace hik_camera_ros2_driver {

class ImageEnhancerNode : public rclcpp::Node {
public:
    ImageEnhancerNode() : Node("image_enhancer_node")
    {
        // 启用OpenCV多线程加速（关键优化）
        cv::setUseOptimized(true);
        cv::setNumThreads(cv::getNumberOfCPUs());  // 使用全部CPU核心

        // 声明参数（精简并优化默认值）
        this->declare_parameter<std::string>("input_topic",
                                             "/camera/image");
        this->declare_parameter<std::string>("output_topic",
                                             "/camera/enhance");
        this->declare_parameter<double>("clahe_clip_limit", 2.5);
        this->declare_parameter<int>("clahe_grid_size",
                                     16);  // 减小网格降低计算量
        this->declare_parameter<double>("gamma", 0.7);
        this->declare_parameter<double>("global_gain",
                                        2.0);  // 降低增益减少溢出处理
        this->declare_parameter<int>("denoise_strength",
                                     5);  // 降低降噪强度

        // 获取参数
        std::string input_topic =
            this->get_parameter("input_topic").as_string();
        std::string output_topic =
            this->get_parameter("output_topic").as_string();
        clahe_clip_limit_ =
            this->get_parameter("clahe_clip_limit").as_double();
        int grid_size = this->get_parameter("clahe_grid_size").as_int();
        clahe_grid_size_ = cv::Size(grid_size, grid_size);
        gamma_ = this->get_parameter("gamma").as_double();
        global_gain_ = this->get_parameter("global_gain").as_double();
        denoise_strength_ =
            this->get_parameter("denoise_strength").as_int();

        // 预生成Gamma查找表（避免实时计算）
        precomputeGammaLUT();

        // 预初始化CLAHE（避免重复创建）
        clahe_ = cv::createCLAHE(clahe_clip_limit_, clahe_grid_size_);

        // QoS配置优化（减少缓冲延迟）
        auto qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        qos_profile.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = 3;  // 减少缓冲区大小，降低延迟
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            output_topic,
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile)));

        // 订阅者使用最佳-effort减少延迟
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic, rclcpp::SensorDataQoS(),
            std::bind(&ImageEnhancerNode::imageCallback, this,
                      std::placeholders::_1));

        // 动态参数回调
        params_callback_ = this->add_on_set_parameters_callback(
            std::bind(&ImageEnhancerNode::onParamsChanged, this,
                      std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "High-performance enhancer started (Input: %s, Output: %s)",
            input_topic.c_str(), output_topic.c_str());
    }

private:
    // 预计算Gamma查找表（核心优化：避免实时pow计算）
    void precomputeGammaLUT()
    {
        gamma_lut_ = cv::Mat(1, 256, CV_8U);
        uchar* lut_data = gamma_lut_.ptr();
        for (int i = 0; i < 256; ++i) {
            lut_data[i] = cv::saturate_cast<uchar>(
                pow(static_cast<double>(i) / 255.0, gamma_) * 255.0);
        }
    }

    rcl_interfaces::msg::SetParametersResult
    onParamsChanged(const std::vector<rclcpp::Parameter>& params)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        bool gamma_updated = false;

        for (const auto& param : params) {
            if (param.get_name() == "clahe_clip_limit") {
                clahe_clip_limit_ = param.as_double();
                clahe_->setClipLimit(clahe_clip_limit_);
            } else if (param.get_name() == "clahe_grid_size") {
                int grid = param.as_int();
                clahe_grid_size_ = cv::Size(grid, grid);
                clahe_->setTilesGridSize(clahe_grid_size_);
            } else if (param.get_name() == "gamma") {
                gamma_ = param.as_double();
                gamma_updated = true;  // 标记Gamma更新
            } else if (param.get_name() == "global_gain") {
                global_gain_ = param.as_double();
            } else if (param.get_name() == "denoise_strength") {
                denoise_strength_ = param.as_int();
            }
        }

        // 仅在Gamma变化时重新计算LUT（减少开销）
        if (gamma_updated) {
            precomputeGammaLUT();
        }
        return result;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 转换为RGB8格式（使用toCvShare减少内存拷贝）
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(
                msg, sensor_msgs::image_encodings::RGB8);
            const cv::Mat& image = cv_ptr->image;

            if (image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Received empty image");
                return;
            }

            // 增强处理（优化后流程）
            cv::Mat enhanced_image = enhanceDarkImage(image);

            // 发布消息（复用header，减少内存操作）
            sensor_msgs::msg::Image::SharedPtr enhanced_msg =
                cv_bridge::CvImage(msg->header,
                                   sensor_msgs::image_encodings::RGB8,
                                   enhanced_image)
                    .toImageMsg();
            enhanced_msg->step = enhanced_msg->width * 3;
            image_pub_->publish(*enhanced_msg);

            // 性能监控（每100帧输出一次耗时）
            static size_t frame_count = 0;
            static auto   last_time = this->now();
            if (++frame_count % 100 == 0) {
                auto   now = this->now();
                double fps = 100.0 / (now - last_time).seconds();
                RCLCPP_INFO(this->get_logger(), "Processing FPS: %.1f",
                            fps);
                last_time = now;
            }
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s",
                         e.what());
        }
        catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s",
                         e.what());
        }
    }

    // 优化后的增强流程（减少计算量）
    cv::Mat enhanceDarkImage(const cv::Mat& rgb_image)
    {
        // 1. 转换到YCrCb（仅提取亮度通道）
        cv::Mat y_channel;
        {
            cv::Mat ycrcb;
            cv::cvtColor(rgb_image, ycrcb, cv::COLOR_RGB2YCrCb);
            std::vector<cv::Mat> channels;
            cv::split(ycrcb, channels);
            y_channel = channels[0];  // 仅保留亮度通道
        }

        // 2. 全局增益（使用整数运算优化）
        y_channel.convertTo(y_channel, -1, global_gain_, 0);

        // 3. CLAHE增强（复用预初始化的clahe_对象）
        clahe_->apply(y_channel, y_channel);

        // 4. Gamma校正（使用预计算的LUT）
        cv::LUT(y_channel, gamma_lut_, y_channel);

        // 5. 合并通道并转回RGB（减少中间变量）
        cv::Mat enhanced_rgb;
        {
            cv::Mat              ycrcb;
            std::vector<cv::Mat> channels(3);
            // 复用原始图像的色度通道（减少计算）
            cv::cvtColor(rgb_image, ycrcb, cv::COLOR_RGB2YCrCb);
            cv::split(ycrcb, channels);
            channels[0] = y_channel;  // 替换亮度通道
            cv::merge(channels, ycrcb);
            cv::cvtColor(ycrcb, enhanced_rgb, cv::COLOR_YCrCb2RGB);
        }

        // 6. 轻量降噪（替换为快速中值滤波）
        if (denoise_strength_ > 0) {
            cv::medianBlur(enhanced_rgb, enhanced_rgb,
                           3);  // 3x3中值滤波，速度远快于非局部均值
        }

        return enhanced_rgb;
    }

    // 成员变量（避免重复创建）
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr    image_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        params_callback_;

    // 预初始化的处理对象
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Mat            gamma_lut_;  // 预计算的Gamma查找表

    // 优化后的参数
    double   clahe_clip_limit_;
    cv::Size clahe_grid_size_;
    double   gamma_;
    double   global_gain_;
    int      denoise_strength_;
};

}  // namespace hik_camera_ros2_driver

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(
        std::make_shared<hik_camera_ros2_driver::ImageEnhancerNode>());
    rclcpp::shutdown();
    return 0;
}
