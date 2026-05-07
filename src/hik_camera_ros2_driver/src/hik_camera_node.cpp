#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "MvCameraControl.h"

#include "camera_info_manager/camera_info_manager.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/utilities.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#if __has_include(<opencv2/core.hpp>) && __has_include(<opencv2/cudaimgproc.hpp>) && __has_include(<opencv2/cudaarithm.hpp>)
  #include <opencv2/core.hpp>
  #include <opencv2/cudaarithm.hpp>
  #include <opencv2/cudaimgproc.hpp>
  #define HIK_HAS_OPENCV_CUDA 1
#else
  #define HIK_HAS_OPENCV_CUDA 0
#endif

namespace hik_camera_ros2_driver {

constexpr int kRingSize = 8;

[[maybe_unused]] static std::string cstr_to_string(const char* s)
{
    return (s == nullptr) ? std::string() : std::string(s);
}

struct FrameSlot {
    std::vector<uint8_t> data;
    uint32_t             width = 0;
    uint32_t             height = 0;
    rclcpp::Time         stamp{0, 0, RCL_ROS_TIME};
    bool                 valid = false;
};

struct CameraContext {
    int         index = 0;
    std::string prefix;

    std::string serial;
    bool        match_by_serial = false;
    std::string model_name;
    std::string device_type;  // USB / GigE / Unknown

    std::mutex                 sdk_mutex;
    void*                     handle = nullptr;
    MV_IMAGE_BASIC_INFO       img_info{};
    MV_CC_PIXEL_CONVERT_PARAM convert_param{};

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr  image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    std::unique_ptr<camera_info_manager::CameraInfoManager> info_mgr;
    sensor_msgs::msg::CameraInfo                            camera_info_msg;

    std::string camera_name;
    std::string frame_id;
    std::string topic;
    std::string camera_info_url;
    std::string configured_pixel_format;
    std::string output_encoding;
    MvGvspPixelType output_pixel_type = PixelType_Gvsp_RGB8_Packed;

    std::thread       capture_thread;
    std::thread       publish_thread;
    std::atomic<bool> running{false};
    bool              started = false;
    bool              connected = false;

    std::array<FrameSlot, kRingSize> ring;
    std::atomic<uint64_t> write_seq{0};
    std::atomic<uint64_t> read_seq{0};

    int fail_count = 0;

    std::chrono::steady_clock::time_point last_debug_log_time =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_restart_time =
        std::chrono::steady_clock::now();

    bool     output_direct_copy = false;
    bool     gamma_enabled = false;
    double   gamma = 0.0;
    std::array<uint8_t, 256> gamma_lut{};
    uint32_t payload_size = 0;

    // hardware timestamp mapping
    bool     has_hw_ts_freq = false;
    uint64_t hw_ts_freq_hz = 0;
    bool     hw_ts_base_initialized = false;
    uint64_t hw_ts_base_tick = 0;
    int64_t  hw_ts_base_ros_ns = 0;

#if HIK_HAS_OPENCV_CUDA
    cv::cuda::GpuMat gpu_bayer_;
    cv::cuda::GpuMat gpu_rgb_;
    cv::cuda::GpuMat gpu_gamma_;
    cv::Ptr<cv::cuda::LookUpTable> gpu_gamma_lut_;
#endif
};

class HikCameraRos2DriverNode : public rclcpp::Node {
public:
    explicit HikCameraRos2DriverNode(const rclcpp::NodeOptions& options)
        : Node("hik_camera_ros2_driver", options)
    {
        RCLCPP_INFO(this->get_logger(),
                    "Starting HikCameraRos2DriverNode (final industrial multi-cam)");

        declareGlobalParameters();

        initializeCameras();

        for (auto& cam_ptr : cameras_) {
            declareCameraParameters(*cam_ptr);
            startCamera(*cam_ptr);
        }

        params_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&HikCameraRos2DriverNode::dynamicParametersCallback,
                      this, std::placeholders::_1));

        for (auto& cam_ptr : cameras_) {
            auto& cam = *cam_ptr;
            cam.running = true;
            cam.capture_thread = std::thread(
                &HikCameraRos2DriverNode::captureLoop, this, &cam);
            cam.publish_thread = std::thread(
                &HikCameraRos2DriverNode::publishLoop, this, &cam);
        }

        size_t started_count = 0;
        for (const auto& cam_ptr : cameras_) {
            if (cam_ptr->started) {
                ++started_count;
            }
        }

        RCLCPP_INFO(this->get_logger(),
                    "HikCameraRos2DriverNode started with %zu/%zu camera(s).",
                    started_count, cameras_.size());
    }

    ~HikCameraRos2DriverNode() override
    {
        for (auto& cam_ptr : cameras_) {
            cam_ptr->running = false;
        }

        for (auto& cam_ptr : cameras_) {
            if (cam_ptr->capture_thread.joinable()) {
                cam_ptr->capture_thread.join();
            }
            if (cam_ptr->publish_thread.joinable()) {
                cam_ptr->publish_thread.join();
            }
        }

        for (auto& cam_ptr : cameras_) {
            auto& cam = *cam_ptr;
            disconnectCamera(cam, false);
        }

        RCLCPP_INFO(this->get_logger(), "HikCameraRos2DriverNode destroyed!");
    }

private:
    void declareGlobalParameters()
    {
        camera_count_ = this->declare_parameter<int>("camera_count", 1);
        if (camera_count_ < 1) camera_count_ = 1;
        if (camera_count_ > 8) camera_count_ = 8;

        use_sensor_data_qos_ =
            this->declare_parameter<bool>("use_sensor_data_qos", true);

        camera_serials_ = this->declare_parameter<std::vector<std::string>>(
            "camera_serials", std::vector<std::string>{});

        prefer_camera_rgb_output_ =
            this->declare_parameter<bool>("prefer_camera_rgb_output", true);

        enable_gpu_debayer_ =
            this->declare_parameter<bool>("enable_gpu_debayer", true);

        use_hardware_timestamp_ =
            this->declare_parameter<bool>("use_hardware_timestamp", true);

        frame_timeout_ms_ =
            this->declare_parameter<int>("frame_timeout_ms", 100);

        restart_fail_threshold_ =
            this->declare_parameter<int>("restart_fail_threshold", 20);

        restart_cooldown_ms_ =
            this->declare_parameter<int>("restart_cooldown_ms", 500);

        start_retry_count_ =
            this->declare_parameter<int>("start_retry_count", 5);

        start_retry_interval_ms_ =
            this->declare_parameter<int>("start_retry_interval_ms", 200);
    }

    static std::string getSerial(const MV_CC_DEVICE_INFO* dev_info)
    {
        if (!dev_info) return {};

        switch (dev_info->nTLayerType) {
        case MV_GIGE_DEVICE:
            return cstr_to_string(
                reinterpret_cast<const char*>(
                    dev_info->SpecialInfo.stGigEInfo.chSerialNumber));
        case MV_USB_DEVICE:
            return cstr_to_string(
                reinterpret_cast<const char*>(
                    dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber));
        default:
            return {};
        }
    }

    static std::string getModelName(const MV_CC_DEVICE_INFO* dev_info)
    {
        if (!dev_info) return {};

        switch (dev_info->nTLayerType) {
        case MV_GIGE_DEVICE:
            return cstr_to_string(
                reinterpret_cast<const char*>(
                    dev_info->SpecialInfo.stGigEInfo.chModelName));
        case MV_USB_DEVICE:
            return cstr_to_string(
                reinterpret_cast<const char*>(
                    dev_info->SpecialInfo.stUsb3VInfo.chModelName));
        default:
            return {};
        }
    }

    static std::string getDeviceTypeString(const MV_CC_DEVICE_INFO* dev_info)
    {
        if (!dev_info) return "Unknown";
        switch (dev_info->nTLayerType) {
        case MV_GIGE_DEVICE:
            return "GigE";
        case MV_USB_DEVICE:
            return "USB";
        default:
            return "Unknown";
        }
    }

    std::string normalizeOutputEncoding(
        const CameraContext& cam,
        const std::string& requested) const
    {
        std::string encoding = requested;
        std::transform(encoding.begin(), encoding.end(), encoding.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (encoding == sensor_msgs::image_encodings::RGB8 ||
            encoding == sensor_msgs::image_encodings::BGR8) {
            return encoding;
        }

        RCLCPP_WARN(this->get_logger(),
                    "[%s] Unsupported output_encoding=%s, fallback to rgb8.",
                    cam.prefix.c_str(), requested.c_str());
        return sensor_msgs::image_encodings::RGB8;
    }

    static MvGvspPixelType pixelTypeForEncoding(const std::string& encoding)
    {
        if (encoding == sensor_msgs::image_encodings::BGR8) {
            return PixelType_Gvsp_BGR8_Packed;
        }
        return PixelType_Gvsp_RGB8_Packed;
    }

    void resetSdkGamma(CameraContext& cam)
    {
        MV_CC_GAMMA_PARAM gamma_param{};
        gamma_param.enGammaType = MV_CC_GAMMA_TYPE_NONE;
        MV_CC_SetBayerGammaParam(cam.handle, &gamma_param);

        MV_CC_SetBoolValue(cam.handle, "GammaEnable", false);
        MV_CC_SetGammaSelector(cam.handle, MV_GAMMA_SELECTOR_USER);
        MV_CC_SetGamma(cam.handle, 1.0f);
    }

    int configureGamma(CameraContext& cam, const double gamma)
    {
        cam.gamma = gamma;
        cam.gamma_enabled = gamma > 0.0 && std::abs(gamma - 1.0) > 1e-6;

        for (size_t i = 0; i < cam.gamma_lut.size(); ++i) {
            int value = static_cast<int>(i);
            if (cam.gamma_enabled) {
                value = static_cast<int>(
                    std::lround(std::pow(static_cast<double>(i) / 255.0,
                                         gamma) * 255.0));
                value = std::clamp(value, 0, 255);
            }
            cam.gamma_lut[i] = static_cast<uint8_t>(value);
        }

#if HIK_HAS_OPENCV_CUDA
        if (cam.gamma_enabled) {
            cv::Mat lut(1, static_cast<int>(cam.gamma_lut.size()), CV_8U,
                        cam.gamma_lut.data());
            cam.gpu_gamma_lut_ = cv::cuda::createLookUpTable(lut);
        } else {
            cam.gpu_gamma_lut_.release();
        }
#endif

        return MV_OK;
    }

    bool initializeCameras()
    {
        cameras_.clear();

        if (!camera_serials_.empty()) {
            const size_t want_n =
                std::min<size_t>(static_cast<size_t>(camera_count_),
                                 camera_serials_.size());

            cameras_.reserve(want_n);

            for (size_t si = 0; si < want_n; ++si) {
                const std::string& wanted_serial = camera_serials_[si];

                auto cam = std::make_unique<CameraContext>();
                cam->index = static_cast<int>(si);
                cam->prefix = "cam" + std::to_string(cam->index);
                cam->serial = wanted_serial;
                cam->match_by_serial = true;

                cameras_.push_back(std::move(cam));
            }

            RCLCPP_INFO(this->get_logger(),
                        "Configured %zu camera slot(s) by serial. Missing devices will be retried.",
                        cameras_.size());
            return true;
        }

        const uint32_t use_n = static_cast<uint32_t>(camera_count_);

        cameras_.reserve(use_n);

        for (uint32_t i = 0; i < use_n; ++i) {
            auto cam = std::make_unique<CameraContext>();
            cam->index = static_cast<int>(i);
            cam->prefix = "cam" + std::to_string(cam->index);
            cam->match_by_serial = false;

            cameras_.push_back(std::move(cam));
        }

        RCLCPP_INFO(this->get_logger(),
                    "Configured %zu camera slot(s) by device index. Missing devices will be retried.",
                    cameras_.size());
        return true;
    }

    void declareCameraParameters(CameraContext& cam)
    {
        rcl_interfaces::msg::ParameterDescriptor param_desc;

        param_desc.description =
            "[" + cam.prefix + "] Acquisition frame rate in Hz";
        const double acquisition_frame_rate =
            this->declare_parameter<double>(
                cam.prefix + ".acquisition_frame_rate", 120.0, param_desc);

        param_desc.description =
            "[" + cam.prefix + "] Exposure time in microseconds";
        const int exposure_time =
            this->declare_parameter<int>(
                cam.prefix + ".exposure_time", 5000, param_desc);

        param_desc.description = "[" + cam.prefix + "] Gain";
        const double gain = this->declare_parameter<double>(
            cam.prefix + ".gain", 0.0, param_desc);

        param_desc.description =
            "[" + cam.prefix + "] Gamma value. Set <= 0 to skip setting";
        const double gamma = this->declare_parameter<double>(
            cam.prefix + ".gamma", 0.0, param_desc);
        configureGamma(cam, gamma);

        param_desc.description = "[" + cam.prefix + "] Pixel Format";
        const std::string default_pixel_format =
            prefer_camera_rgb_output_ ? "RGB8Packed" : "BayerRG8";
        const std::string pixel_format =
            this->declare_parameter<std::string>(
                cam.prefix + ".pixel_format", default_pixel_format, param_desc);

        cam.camera_name = this->declare_parameter<std::string>(
            cam.prefix + ".camera_name", cam.prefix);
        cam.frame_id = this->declare_parameter<std::string>(
            cam.prefix + ".frame_id", cam.camera_name + "_optical_frame");
        cam.topic = this->declare_parameter<std::string>(
            cam.prefix + ".camera_topic", cam.camera_name + "/image_raw");
        cam.output_encoding = normalizeOutputEncoding(
            cam,
            this->declare_parameter<std::string>(
                cam.prefix + ".output_encoding",
                sensor_msgs::image_encodings::RGB8));
        cam.camera_info_url = this->declare_parameter<std::string>(
            cam.prefix + ".camera_info_url",
            "package://hik_camera_ros2_driver/config/camera_info.yaml");

        (void)acquisition_frame_rate;
        (void)exposure_time;
        (void)gain;
        (void)gamma;
        (void)pixel_format;
    }

    bool startGrabbingWithRetry(CameraContext& cam)
    {
        int ret = MV_OK;
        for (int i = 0; i < start_retry_count_; ++i) {
            ret = MV_CC_StartGrabbing(cam.handle);
            if (ret == MV_OK) {
                return true;
            }
            RCLCPP_WARN(this->get_logger(),
                        "[%s] StartGrabbing retry %d/%d failed ret=%x",
                        cam.prefix.c_str(), i + 1, start_retry_count_, ret);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(start_retry_interval_ms_));
        }

        RCLCPP_ERROR(this->get_logger(),
                     "[%s] StartGrabbing failed ret=%x",
                     cam.prefix.c_str(), ret);
        return false;
    }

    void tryConfigureTimestampFrequency(CameraContext& cam)
    {
        cam.has_hw_ts_freq = false;
        cam.hw_ts_freq_hz = 0;

        MVCC_INTVALUE int_value{};

        const char* keys[] = {
            "GevTimestampTickFrequency",
            "DeviceTimestampFrequency",
            "TimestampTickFrequency"
        };

        for (const auto* key : keys) {
            if (MV_CC_GetIntValue(cam.handle, key, &int_value) == MV_OK &&
                int_value.nCurValue > 0) {
                cam.hw_ts_freq_hz = static_cast<uint64_t>(int_value.nCurValue);
                cam.has_hw_ts_freq = true;
                RCLCPP_INFO(this->get_logger(),
                            "[%s] Hardware timestamp frequency from %s: %llu Hz",
                            cam.prefix.c_str(), key,
                            static_cast<unsigned long long>(cam.hw_ts_freq_hz));
                return;
            }
        }

        RCLCPP_WARN(this->get_logger(),
                    "[%s] Hardware timestamp frequency unavailable, "
                    "fallback to host timestamp.",
                    cam.prefix.c_str());
    }

    static std::string deriveCameraInfoTopic(const std::string& image_topic)
    {
        const std::string suffix = "/image";
        const std::string suffix_raw = "/image_raw";
        if (image_topic.size() >= suffix_raw.size() &&
            image_topic.compare(image_topic.size() - suffix_raw.size(), suffix_raw.size(), suffix_raw) == 0) {
            return image_topic.substr(0, image_topic.size() - suffix_raw.size()) + "/camera_info";
        }
        if (image_topic.size() >= suffix.size() &&
            image_topic.compare(image_topic.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return image_topic.substr(0, image_topic.size() - suffix.size()) + "/camera_info";
        }
        return image_topic + "/camera_info";
    }

    void releaseCameraHandleLocked(CameraContext& cam)
    {
        if (cam.handle) {
            MV_CC_StopGrabbing(cam.handle);
            MV_CC_CloseDevice(cam.handle);
            MV_CC_DestroyHandle(&cam.handle);
            cam.handle = nullptr;
        }

        cam.connected = false;
        cam.has_hw_ts_freq = false;
        cam.hw_ts_freq_hz = 0;
        cam.hw_ts_base_initialized = false;
    }

    void disconnectCamera(CameraContext& cam, bool warn = true)
    {
        std::lock_guard<std::mutex> lock(cam.sdk_mutex);
        const bool had_handle = cam.handle != nullptr;
        releaseCameraHandleLocked(cam);
        cam.fail_count = 0;

        if (warn && had_handle) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Camera disconnected, entering reconnect loop.",
                        cam.prefix.c_str());
        }
    }

    const MV_CC_DEVICE_INFO* selectDeviceForCamera(
        const CameraContext& cam,
        const MV_CC_DEVICE_INFO_LIST& device_list,
        uint32_t& selected_idx) const
    {
        if (cam.match_by_serial && !cam.serial.empty()) {
            for (uint32_t i = 0; i < device_list.nDeviceNum; ++i) {
                const auto* dev = device_list.pDeviceInfo[i];
                const auto serial = getSerial(dev);
                if (!serial.empty() && serial == cam.serial) {
                    selected_idx = i;
                    return dev;
                }
            }
            return nullptr;
        }

        if (cam.index >= 0 &&
            static_cast<uint32_t>(cam.index) < device_list.nDeviceNum) {
            selected_idx = static_cast<uint32_t>(cam.index);
            return device_list.pDeviceInfo[selected_idx];
        }

        return nullptr;
    }

    bool connectCamera(CameraContext& cam, bool log_success = true)
    {
        {
            std::lock_guard<std::mutex> lock(cam.sdk_mutex);
            if (cam.handle) {
                return true;
            }
        }

        std::lock_guard<std::mutex> open_lock(camera_open_mutex_);
        std::lock_guard<std::mutex> lock(cam.sdk_mutex);

        if (cam.handle) {
            return true;
        }

        MV_CC_DEVICE_INFO_LIST device_list{};
        int ret = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE,
                                    &device_list);
        if (ret != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to enumerate cameras, ret=%x. Will retry.",
                        cam.prefix.c_str(), ret);
            return false;
        }

        if (device_list.nDeviceNum == 0) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] No Hik camera found. Will retry.",
                        cam.prefix.c_str());
            return false;
        }

        uint32_t selected_idx = 0;
        const MV_CC_DEVICE_INFO* dev =
            selectDeviceForCamera(cam, device_list, selected_idx);
        if (!dev) {
            if (cam.match_by_serial) {
                RCLCPP_WARN(this->get_logger(),
                            "[%s] Requested serial '%s' not found among %u device(s). Will retry.",
                            cam.prefix.c_str(), cam.serial.c_str(),
                            device_list.nDeviceNum);
            } else {
                RCLCPP_WARN(this->get_logger(),
                            "[%s] Device index %d not available among %u device(s). Will retry.",
                            cam.prefix.c_str(), cam.index,
                            device_list.nDeviceNum);
            }
            return false;
        }

        cam.serial = getSerial(dev);
        cam.model_name = getModelName(dev);
        cam.device_type = getDeviceTypeString(dev);

        ret = MV_CC_CreateHandle(&cam.handle,
                                 const_cast<MV_CC_DEVICE_INFO*>(dev));
        if (ret != MV_OK || cam.handle == nullptr) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to create camera handle, ret=%x. Will retry.",
                        cam.prefix.c_str(), ret);
            cam.handle = nullptr;
            return false;
        }

        auto cleanup_on_failure = [&]() {
            releaseCameraHandleLocked(cam);
            cam.fail_count = 0;
        };

        ret = MV_CC_OpenDevice(cam.handle);
        if (ret != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to open camera serial=%s, ret=%x. Will retry.",
                        cam.prefix.c_str(), cam.serial.c_str(), ret);
            cleanup_on_failure();
            return false;
        }

        ret = MV_CC_GetImageInfo(cam.handle, &cam.img_info);
        if (ret != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to get camera image info, ret=%x. Will retry.",
                        cam.prefix.c_str(), ret);
            cleanup_on_failure();
            return false;
        }

        MV_CC_SetEnumValueByString(cam.handle, "AcquisitionMode", "Continuous");
        MV_CC_SetEnumValueByString(cam.handle, "TriggerMode", "Off");
        MV_CC_SetEnumValueByString(cam.handle, "ExposureAuto", "Off");
        MV_CC_SetEnumValueByString(cam.handle, "GainAuto", "Off");
        resetSdkGamma(cam);

        if (cam.device_type == "GigE") {
            MV_CC_SetIntValue(cam.handle, "GevSCPSPacketSize", 1500);
        }

        const double acquisition_frame_rate =
            this->get_parameter(cam.prefix + ".acquisition_frame_rate").as_double();
        const int exposure_time =
            static_cast<int>(this->get_parameter(cam.prefix + ".exposure_time").as_int());
        const double gain =
            this->get_parameter(cam.prefix + ".gain").as_double();
        const double gamma =
            this->get_parameter(cam.prefix + ".gamma").as_double();
        configureGamma(cam, gamma);
        const std::string pixel_format =
            this->get_parameter(cam.prefix + ".pixel_format").as_string();
        cam.output_encoding = normalizeOutputEncoding(
            cam,
            this->get_parameter(cam.prefix + ".output_encoding").as_string());
        cam.output_pixel_type = pixelTypeForEncoding(cam.output_encoding);

        MV_CC_SetBoolValue(cam.handle, "AcquisitionFrameRateEnable", true);
        MV_CC_SetFloatValue(cam.handle, "AcquisitionFrameRate",
                            static_cast<float>(acquisition_frame_rate));
        MV_CC_SetFloatValue(cam.handle, "ExposureTime",
                            static_cast<float>(exposure_time));
        MV_CC_SetFloatValue(cam.handle, "Gain",
                            static_cast<float>(gain));

        const int status =
            MV_CC_SetEnumValueByString(cam.handle, "PixelFormat",
                                       pixel_format.c_str());
        if (status != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to set PixelFormat=%s, status=%d. Will retry.",
                        cam.prefix.c_str(), pixel_format.c_str(), status);
            cleanup_on_failure();
            return false;
        }

        cam.configured_pixel_format = pixel_format;
        cam.output_direct_copy =
            ((pixel_format == "RGB8Packed" &&
              cam.output_encoding == sensor_msgs::image_encodings::RGB8) ||
             (pixel_format == "BGR8Packed" &&
              cam.output_encoding == sensor_msgs::image_encodings::BGR8));

        MV_CC_SetImageNodeNum(cam.handle, 15);

        MVCC_INTVALUE payload{};
        if (MV_CC_GetIntValue(cam.handle, "PayloadSize", &payload) == MV_OK &&
            payload.nCurValue > 0) {
            cam.payload_size = static_cast<uint32_t>(payload.nCurValue);
        } else {
            cam.payload_size = cam.img_info.nWidthValue * cam.img_info.nHeightValue * 3U;
        }

        cam.convert_param.nWidth = cam.img_info.nWidthValue;
        cam.convert_param.nHeight = cam.img_info.nHeightValue;
        cam.convert_param.enDstPixelType = cam.output_pixel_type;

        for (auto& slot : cam.ring) {
            slot.data.resize(static_cast<size_t>(cam.img_info.nWidthValue) *
                             static_cast<size_t>(cam.img_info.nHeightValue) * 3U);
            slot.valid = false;
        }
        cam.read_seq.store(cam.write_seq.load(std::memory_order_relaxed),
                           std::memory_order_release);

        if (use_hardware_timestamp_) {
            tryConfigureTimestampFrequency(cam);
        }

        if (!startGrabbingWithRetry(cam)) {
            cleanup_on_failure();
            return false;
        }

        cam.connected = true;
        cam.fail_count = 0;
        cam.last_restart_time = std::chrono::steady_clock::now();

        if (log_success) {
            RCLCPP_INFO(this->get_logger(),
                        "[%s] Camera connected. idx=%u type=%s serial=%s model=%s pixel_format=%s output_encoding=%s gpu_debayer=%s",
                        cam.prefix.c_str(),
                        selected_idx,
                        cam.device_type.c_str(),
                        cam.serial.c_str(),
                        cam.model_name.c_str(),
                        cam.configured_pixel_format.c_str(),
                        cam.output_encoding.c_str(),
#if HIK_HAS_OPENCV_CUDA
                        (enable_gpu_debayer_ ? "enabled_if_raw" : "disabled"));
#else
                        "not_available");
#endif
        }

        return true;
    }

    bool ensureCameraConnected(CameraContext& cam)
    {
        {
            std::lock_guard<std::mutex> lock(cam.sdk_mutex);
            if (cam.handle) {
                return true;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - cam.last_restart_time).count();
        if (elapsed_ms < restart_cooldown_ms_) {
            return false;
        }

        cam.last_restart_time = now;
        return connectCamera(cam);
    }

    void startCamera(CameraContext& cam)
    {
        auto image_qos = use_sensor_data_qos_ ? rclcpp::SensorDataQoS()
                                             : rclcpp::QoS(rclcpp::KeepLast(10));

        cam.image_pub = this->create_publisher<sensor_msgs::msg::Image>(cam.topic, image_qos);
        cam.camera_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            deriveCameraInfoTopic(cam.topic), image_qos);

        cam.info_mgr =
            std::make_unique<camera_info_manager::CameraInfoManager>(
                this, cam.camera_name);
        if (cam.info_mgr->validateURL(cam.camera_info_url)) {
            cam.info_mgr->loadCameraInfo(cam.camera_info_url);
            cam.camera_info_msg = cam.info_mgr->getCameraInfo();
            RCLCPP_INFO(this->get_logger(),
                        "[%s] Loaded camera info from URL.",
                        cam.prefix.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Invalid camera info URL: %s",
                        cam.prefix.c_str(), cam.camera_info_url.c_str());
        }
        cam.camera_info_msg.header.frame_id = cam.frame_id;

        cam.started = true;
        if (!connectCamera(cam)) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Camera is not connected yet. Capture thread will keep retrying.",
                        cam.prefix.c_str());
        }
    }

    rclcpp::Time makeFrameStamp(CameraContext& cam,
                                const MV_FRAME_OUT& out_frame)
    {
        if (!use_hardware_timestamp_ || !cam.has_hw_ts_freq ||
            cam.hw_ts_freq_hz == 0) {
            return this->now();
        }

        const uint64_t dev_tick =
            (static_cast<uint64_t>(out_frame.stFrameInfo.nDevTimeStampHigh) << 32) |
             static_cast<uint64_t>(out_frame.stFrameInfo.nDevTimeStampLow);

        if (!cam.hw_ts_base_initialized) {
            cam.hw_ts_base_initialized = true;
            cam.hw_ts_base_tick = dev_tick;
            cam.hw_ts_base_ros_ns = this->now().nanoseconds();
            return rclcpp::Time(cam.hw_ts_base_ros_ns, RCL_ROS_TIME);
        }

        const uint64_t delta_tick = dev_tick - cam.hw_ts_base_tick;
        const int64_t delta_ns =
            static_cast<int64_t>((static_cast<long double>(delta_tick) * 1.0e9L) /
                                 static_cast<long double>(cam.hw_ts_freq_hz));

        return rclcpp::Time(cam.hw_ts_base_ros_ns + delta_ns, RCL_ROS_TIME);
    }

#if HIK_HAS_OPENCV_CUDA
    int gpuDemosaicCode(const CameraContext& cam) const
    {
        if (cam.output_encoding == sensor_msgs::image_encodings::BGR8) {
            return cv::cuda::COLOR_BayerRG2BGR_MHT;
        }
        return cv::cuda::COLOR_BayerRG2RGB_MHT;
    }

    bool gpuDebayer(CameraContext& cam,
                    const MV_FRAME_OUT& out_frame,
                    FrameSlot& slot)
    {
        const int width = static_cast<int>(out_frame.stFrameInfo.nWidth);
        const int height = static_cast<int>(out_frame.stFrameInfo.nHeight);

        cv::Mat cpu_bayer(height, width, CV_8UC1,
                          const_cast<unsigned char*>(out_frame.pBufAddr));
        cam.gpu_bayer_.upload(cpu_bayer);

        cv::cuda::demosaicing(cam.gpu_bayer_, cam.gpu_rgb_,
                              gpuDemosaicCode(cam), 3);

        cv::cuda::GpuMat* output = &cam.gpu_rgb_;
        if (cam.gamma_enabled && cam.gpu_gamma_lut_) {
            cam.gpu_gamma_lut_->transform(cam.gpu_rgb_, cam.gpu_gamma_);
            output = &cam.gpu_gamma_;
        }

        const size_t dst_size =
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;
        if (slot.data.size() < dst_size) {
            slot.data.resize(dst_size);
        }

        cv::Mat cpu_rgb(height, width, CV_8UC3, slot.data.data());
        output->download(cpu_rgb);
        return true;
    }
#endif

    void applyCpuGamma(CameraContext& cam, FrameSlot& slot) const
    {
        if (!cam.gamma_enabled) {
            return;
        }

        const size_t bytes =
            static_cast<size_t>(slot.width) * static_cast<size_t>(slot.height) * 3U;
        if (slot.data.size() < bytes) {
            return;
        }

        for (size_t i = 0; i < bytes; ++i) {
            slot.data[i] = cam.gamma_lut[slot.data[i]];
        }
    }

    bool cpuConvertToRgb(CameraContext& cam,
                         const MV_FRAME_OUT& out_frame,
                         FrameSlot& slot)
    {
        const uint32_t width = out_frame.stFrameInfo.nWidth;
        const uint32_t height = out_frame.stFrameInfo.nHeight;
        const size_t dst_size =
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;

        if (slot.data.size() < dst_size) {
            slot.data.resize(dst_size);
        }

        cam.convert_param.nWidth = width;
        cam.convert_param.nHeight = height;
        cam.convert_param.pDstBuffer = slot.data.data();
        cam.convert_param.nDstBufferSize =
            static_cast<unsigned int>(slot.data.size());
        cam.convert_param.pSrcData = out_frame.pBufAddr;
        cam.convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
        cam.convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;
        cam.convert_param.enDstPixelType = cam.output_pixel_type;

        const int cvt_ret = MV_CC_ConvertPixelType(cam.handle, &cam.convert_param);
        if (cvt_ret != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] ConvertPixelType failed: [%x]",
                        cam.prefix.c_str(), cvt_ret);
            return false;
        }
        applyCpuGamma(cam, slot);
        return true;
    }

    bool fillSlotFromFrame(CameraContext& cam,
                           const MV_FRAME_OUT& out_frame,
                           FrameSlot& slot)
    {
        slot.width = out_frame.stFrameInfo.nWidth;
        slot.height = out_frame.stFrameInfo.nHeight;
        slot.stamp = makeFrameStamp(cam, out_frame);

        if (cam.output_direct_copy &&
            out_frame.stFrameInfo.enPixelType == cam.output_pixel_type) {
            const size_t bytes = static_cast<size_t>(out_frame.stFrameInfo.nFrameLen);
            if (slot.data.size() < bytes) {
                slot.data.resize(bytes);
            }
            std::memcpy(slot.data.data(), out_frame.pBufAddr, bytes);
            applyCpuGamma(cam, slot);
            slot.valid = true;
            return true;
        }

#if HIK_HAS_OPENCV_CUDA
        if (enable_gpu_debayer_ && !cam.output_direct_copy) {
            if (gpuDebayer(cam, out_frame, slot)) {
                slot.valid = true;
                return true;
            }
        }
#endif

        if (cpuConvertToRgb(cam, out_frame, slot)) {
            slot.valid = true;
            return true;
        }

        slot.valid = false;
        return false;
    }

    void maybeRestartGrab(CameraContext& cam)
    {
        if (restart_fail_threshold_ <= 0 ||
            cam.fail_count < restart_fail_threshold_) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - cam.last_restart_time).count();
        if (elapsed_ms < restart_cooldown_ms_) {
            return;
        }

        RCLCPP_WARN(this->get_logger(),
                    "[%s] Camera has %d consecutive grab/convert failures, reconnecting.",
                    cam.prefix.c_str(), cam.fail_count);
        cam.last_restart_time = now;
        disconnectCamera(cam);
    }

    void captureLoop(CameraContext* cam)
    {
        if (cam == nullptr) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "[%s] Capture thread started.",
                    cam->prefix.c_str());

        while (rclcpp::ok() && cam->running) {
            if (!cam->started) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (!ensureCameraConnected(*cam)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            MV_FRAME_OUT out_frame{};
            int n_ret = MV_E_HANDLE;
            bool ok = false;

            {
                std::lock_guard<std::mutex> lock(cam->sdk_mutex);
                if (!cam->handle) {
                    continue;
                }

                n_ret = MV_CC_GetImageBuffer(cam->handle, &out_frame,
                                             frame_timeout_ms_);

                if (n_ret == MV_OK) {
                    const uint64_t seq =
                        cam->write_seq.load(std::memory_order_relaxed);
                    FrameSlot& slot = cam->ring[seq % kRingSize];

                    ok = fillSlotFromFrame(*cam, out_frame, slot);
                    MV_CC_FreeImageBuffer(cam->handle, &out_frame);

                    if (ok) {
                        cam->write_seq.store(seq + 1,
                                             std::memory_order_release);
                    }
                }
            }

            if (n_ret == MV_OK && ok) {
                cam->fail_count = 0;

                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        now - cam->last_debug_log_time)
                        .count() >= 3) {
                    MVCC_FLOATVALUE f_value{};
                    if (MV_CC_GetFloatValue(cam->handle, "ResultingFrameRate",
                                            &f_value) == MV_OK) {
                        RCLCPP_DEBUG(this->get_logger(),
                                     "[%s] ResultingFrameRate: %f Hz",
                                     cam->prefix.c_str(), f_value.fCurValue);
                    }
                    cam->last_debug_log_time = now;
                }

            } else {
                cam->fail_count++;
            }

            maybeRestartGrab(*cam);
        }
    }

    void publishLoop(CameraContext* cam)
    {
        if (cam == nullptr || !cam->started) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "[%s] Publish thread started.",
                    cam->prefix.c_str());

        while (rclcpp::ok() && cam->running) {
            const uint64_t read_seq =
                cam->read_seq.load(std::memory_order_relaxed);
            const uint64_t write_seq =
                cam->write_seq.load(std::memory_order_acquire);

            if (read_seq >= write_seq) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }

            FrameSlot& slot = cam->ring[read_seq % kRingSize];
            if (!slot.valid) {
                cam->read_seq.store(read_seq + 1, std::memory_order_release);
                continue;
            }

            auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
            image_msg->header.stamp = slot.stamp;
            image_msg->header.frame_id = cam->frame_id;
            image_msg->height = slot.height;
            image_msg->width = slot.width;
            image_msg->step = slot.width * 3U;
            image_msg->encoding = cam->output_encoding;

            image_msg->data.swap(slot.data);

            cam->camera_info_msg.header = image_msg->header;
            cam->camera_info_pub->publish(cam->camera_info_msg);
            cam->image_pub->publish(std::move(image_msg));

            cam->read_seq.store(read_seq + 1, std::memory_order_release);
        }
    }

    rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
        const std::vector<rclcpp::Parameter>& parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        auto find_cam_by_prefix =
            [&](const std::string& prefix) -> CameraContext* {
            for (auto& c : cameras_) {
                if (c->prefix == prefix) {
                    return c.get();
                }
            }
            return nullptr;
        };

        for (const auto& param : parameters) {
            const auto& name = param.get_name();
            const auto dot = name.find('.');
            if (dot == std::string::npos) {
                continue;
            }

            const std::string prefix = name.substr(0, dot);
            const std::string key = name.substr(dot + 1);

            CameraContext* cam = find_cam_by_prefix(prefix);
            if (!cam) {
                continue;
            }

            std::lock_guard<std::mutex> lock(cam->sdk_mutex);
            if (!cam->handle) {
                continue;
            }

            int status = MV_OK;

            if (key == "gain") {
                if (param.get_type() !=
                    rclcpp::ParameterType::PARAMETER_DOUBLE) {
                    result.successful = false;
                    result.reason = "Parameter type mismatch for " + name;
                    return result;
                }
                status = MV_CC_SetFloatValue(cam->handle, "Gain",
                                             static_cast<float>(param.as_double()));

            } else if (key == "gamma") {
                if (param.get_type() !=
                    rclcpp::ParameterType::PARAMETER_DOUBLE) {
                    result.successful = false;
                    result.reason = "Parameter type mismatch for " + name;
                    return result;
                }
                resetSdkGamma(*cam);
                status = configureGamma(*cam, param.as_double());

            } else if (key == "exposure_time") {
                if (param.get_type() !=
                    rclcpp::ParameterType::PARAMETER_INTEGER) {
                    result.successful = false;
                    result.reason = "Parameter type mismatch for " + name;
                    return result;
                }
                status = MV_CC_SetFloatValue(
                    cam->handle, "ExposureTime",
                    static_cast<float>(param.as_int()));

            } else if (key == "acquisition_frame_rate") {
                if (param.get_type() !=
                    rclcpp::ParameterType::PARAMETER_DOUBLE) {
                    result.successful = false;
                    result.reason = "Parameter type mismatch for " + name;
                    return result;
                }
                status = MV_CC_SetFloatValue(
                    cam->handle, "AcquisitionFrameRate",
                    static_cast<float>(param.as_double()));

            } else {
                continue;
            }

            if (status != MV_OK) {
                result.successful = false;
                result.reason = "Failed to set " + name +
                                ", status = " + std::to_string(status);
                return result;
            }
        }

        return result;
    }

private:
    std::vector<std::unique_ptr<CameraContext>> cameras_;

    int                      camera_count_ = 1;
    bool                     use_sensor_data_qos_ = true;
    std::vector<std::string> camera_serials_;

    bool prefer_camera_rgb_output_ = true;
    bool enable_gpu_debayer_ = true;
    bool use_hardware_timestamp_ = true;

    int frame_timeout_ms_ = 100;
    int restart_fail_threshold_ = 20;
    int restart_cooldown_ms_ = 500;
    int start_retry_count_ = 5;
    int start_retry_interval_ms_ = 200;

    std::mutex camera_open_mutex_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        params_callback_handle_;
};

}  // namespace hik_camera_ros2_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    hik_camera_ros2_driver::HikCameraRos2DriverNode)
