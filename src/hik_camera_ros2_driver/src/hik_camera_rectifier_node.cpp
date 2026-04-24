#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace hik_camera_ros2_driver {

class HikCameraRectifierNode : public rclcpp::Node {
public:
    explicit HikCameraRectifierNode(const rclcpp::NodeOptions& options)
        : Node("hik_camera_rectifier", options)
    {
        input_image_topic_ = this->declare_parameter<std::string>(
            "input_image_topic", "/camera0/image");
        input_camera_info_topic_ = this->declare_parameter<std::string>(
            "input_camera_info_topic", "/camera0/camera_info");
        output_image_topic_ = this->declare_parameter<std::string>(
            "output_image_topic", "/camera0_rect/image");
        output_camera_info_topic_ = this->declare_parameter<std::string>(
            "output_camera_info_topic", "/camera0_rect/camera_info");
        interpolation_ = this->declare_parameter<int>(
            "interpolation", cv::INTER_LINEAR);
        use_sensor_data_qos_ = this->declare_parameter<bool>(
            "use_sensor_data_qos", true);

        const auto image_qos =
            use_sensor_data_qos_ ? rclcpp::SensorDataQoS()
                                 : rclcpp::QoS(rclcpp::KeepLast(10));
        const auto info_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            output_image_topic_, image_qos);
        camera_info_pub_ =
            this->create_publisher<sensor_msgs::msg::CameraInfo>(
                output_camera_info_topic_, info_qos);

        camera_info_sub_ =
            this->create_subscription<sensor_msgs::msg::CameraInfo>(
                input_camera_info_topic_,
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
                std::bind(&HikCameraRectifierNode::onCameraInfo, this,
                          std::placeholders::_1));

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_image_topic_, image_qos,
            std::bind(&HikCameraRectifierNode::onImage, this,
                      std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "HikCameraRectifierNode started. input_image=%s input_info=%s output_image=%s output_info=%s",
            input_image_topic_.c_str(), input_camera_info_topic_.c_str(),
            output_image_topic_.c_str(), output_camera_info_topic_.c_str());
    }

private:
    struct RectifyCache {
        sensor_msgs::msg::CameraInfo source_info;
        sensor_msgs::msg::CameraInfo rectified_info;
        cv::Mat                      map1;
        cv::Mat                      map2;
        bool                         valid = false;
        bool                         zero_distortion = true;
    };

    static cv::Mat makeMat3x3(const std::array<double, 9>& values)
    {
        cv::Mat mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                mat.at<double>(r, c) = values[static_cast<size_t>(r * 3 + c)];
            }
        }
        return mat;
    }

    static cv::Mat makeMat3x4(const std::array<double, 12>& values)
    {
        cv::Mat mat(3, 4, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                mat.at<double>(r, c) =
                    values[static_cast<size_t>(r * 4 + c)];
            }
        }
        return mat;
    }

    static cv::Mat makeDistortionRow(const std::vector<double>& values)
    {
        cv::Mat dist(1, static_cast<int>(values.size()), CV_64F);
        for (size_t i = 0; i < values.size(); ++i) {
            dist.at<double>(0, static_cast<int>(i)) = values[i];
        }
        return dist;
    }

    static bool isZeroDistortion(const std::vector<double>& values)
    {
        for (double value : values) {
            if (value != 0.0) {
                return false;
            }
        }
        return true;
    }

    static bool sameCalibration(
        const sensor_msgs::msg::CameraInfo& lhs,
        const sensor_msgs::msg::CameraInfo& rhs)
    {
        return lhs.width == rhs.width && lhs.height == rhs.height &&
               lhs.distortion_model == rhs.distortion_model &&
               lhs.d == rhs.d && lhs.k == rhs.k && lhs.r == rhs.r &&
               lhs.p == rhs.p && lhs.binning_x == rhs.binning_x &&
               lhs.binning_y == rhs.binning_y && lhs.roi.x_offset == rhs.roi.x_offset &&
               lhs.roi.y_offset == rhs.roi.y_offset &&
               lhs.roi.height == rhs.roi.height &&
               lhs.roi.width == rhs.roi.width &&
               lhs.roi.do_rectify == rhs.roi.do_rectify;
    }

    static bool getEncodingSpec(const std::string& encoding, int& cv_type)
    {
        if (encoding == sensor_msgs::image_encodings::RGB8 ||
            encoding == sensor_msgs::image_encodings::BGR8) {
            cv_type = CV_8UC3;
            return true;
        }
        if (encoding == sensor_msgs::image_encodings::MONO8) {
            cv_type = CV_8UC1;
            return true;
        }
        return false;
    }

    static size_t subscriberCount(
        const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& image_pub,
        const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr& info_pub)
    {
        size_t count = 0;
        if (image_pub) {
            count += image_pub->get_subscription_count();
            count += image_pub->get_intra_process_subscription_count();
        }
        if (info_pub) {
            count += info_pub->get_subscription_count();
            count += info_pub->get_intra_process_subscription_count();
        }
        return count;
    }

    void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
    {
        if (msg == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.valid && sameCalibration(cache_.source_info, *msg)) {
            return;
        }

        if (msg->width == 0 || msg->height == 0 || msg->k[0] == 0.0 ||
            msg->k[4] == 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "Ignore invalid camera info on %s",
                        input_camera_info_topic_.c_str());
            cache_.valid = false;
            return;
        }

        cv::Mat camera_matrix = makeMat3x3(msg->k);
        cv::Mat distortion = makeDistortionRow(msg->d);
        cv::Mat rectify_matrix = makeMat3x3(msg->r);
        cv::Mat projection = makeMat3x4(msg->p);

        bool projection_valid = projection.at<double>(0, 0) != 0.0 &&
                                projection.at<double>(1, 1) != 0.0;
        if (!projection_valid) {
            projection = cv::Mat::zeros(3, 4, CV_64F);
            camera_matrix.copyTo(projection(cv::Rect(0, 0, 3, 3)));
        }

        if (rectify_matrix.at<double>(0, 0) == 0.0 &&
            rectify_matrix.at<double>(1, 1) == 0.0 &&
            rectify_matrix.at<double>(2, 2) == 0.0) {
            rectify_matrix = cv::Mat::eye(3, 3, CV_64F);
        }

        cv::Mat new_camera_matrix = projection(cv::Rect(0, 0, 3, 3)).clone();

        cache_.source_info = *msg;
        cache_.rectified_info = *msg;
        cache_.rectified_info.d.assign(msg->d.size(), 0.0);
        cache_.rectified_info.r = {1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0};

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                cache_.rectified_info.k[static_cast<size_t>(r * 3 + c)] =
                    new_camera_matrix.at<double>(r, c);
            }
        }

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                cache_.rectified_info.p[static_cast<size_t>(r * 4 + c)] =
                    projection.at<double>(r, c);
            }
        }

        cache_.zero_distortion = isZeroDistortion(msg->d);
        cv::initUndistortRectifyMap(
            camera_matrix, distortion, rectify_matrix, new_camera_matrix,
            cv::Size(static_cast<int>(msg->width),
                     static_cast<int>(msg->height)),
            CV_16SC2, cache_.map1, cache_.map2);
        cache_.valid = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Rectification maps updated for %s (%ux%u, zero_distortion=%s)",
            input_image_topic_.c_str(), msg->width, msg->height,
            cache_.zero_distortion ? "true" : "false");
    }

    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        if (msg == nullptr) {
            return;
        }

        if (subscriberCount(image_pub_, camera_info_pub_) == 0) {
            return;
        }

        RectifyCache cache;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache = cache_;
        }

        if (!cache.valid) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Waiting for valid CameraInfo on %s before rectifying %s",
                input_camera_info_topic_.c_str(),
                input_image_topic_.c_str());
            return;
        }

        if (msg->width != cache.source_info.width ||
            msg->height != cache.source_info.height) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Image size %ux%u does not match CameraInfo size %ux%u on %s",
                msg->width, msg->height, cache.source_info.width,
                cache.source_info.height, input_image_topic_.c_str());
            return;
        }

        int cv_type = 0;
        if (!getEncodingSpec(msg->encoding, cv_type)) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Unsupported encoding '%s' on %s",
                msg->encoding.c_str(), input_image_topic_.c_str());
            return;
        }

        auto rectified_info =
            std::make_unique<sensor_msgs::msg::CameraInfo>(
                cache.rectified_info);
        rectified_info->header = msg->header;

        auto rectified_image = std::make_unique<sensor_msgs::msg::Image>();
        rectified_image->header = msg->header;
        rectified_image->height = msg->height;
        rectified_image->width = msg->width;
        rectified_image->encoding = msg->encoding;
        rectified_image->is_bigendian = msg->is_bigendian;
        rectified_image->step =
            static_cast<sensor_msgs::msg::Image::_step_type>(msg->width) *
            static_cast<sensor_msgs::msg::Image::_step_type>(CV_ELEM_SIZE(cv_type));
        rectified_image->data.resize(
            static_cast<size_t>(rectified_image->step) *
            static_cast<size_t>(rectified_image->height));

        const cv::Mat src(static_cast<int>(msg->height),
                          static_cast<int>(msg->width), cv_type,
                          const_cast<unsigned char*>(msg->data.data()),
                          static_cast<size_t>(msg->step));
        cv::Mat dst(static_cast<int>(rectified_image->height),
                    static_cast<int>(rectified_image->width), cv_type,
                    rectified_image->data.data(),
                    static_cast<size_t>(rectified_image->step));

        if (cache.zero_distortion) {
            src.copyTo(dst);
        } else {
            cv::remap(src, dst, cache.map1, cache.map2, interpolation_);
        }

        camera_info_pub_->publish(std::move(rectified_info));
        image_pub_->publish(std::move(rectified_image));
    }

private:
    std::string input_image_topic_;
    std::string input_camera_info_topic_;
    std::string output_image_topic_;
    std::string output_camera_info_topic_;
    int         interpolation_{cv::INTER_LINEAR};
    bool        use_sensor_data_qos_{true};

    std::mutex   cache_mutex_;
    RectifyCache cache_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_pub_;
};

}  // namespace hik_camera_ros2_driver

RCLCPP_COMPONENTS_REGISTER_NODE(
    hik_camera_ros2_driver::HikCameraRectifierNode)
