#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "MvCameraControl.h"

#include "camera_info_manager/camera_info_manager.hpp"
#include "image_transport/image_transport.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/utilities.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

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
    std::string model_name;
    std::string device_type;  // USB / GigE / Unknown

    void*                     handle = nullptr;
    MV_IMAGE_BASIC_INFO       img_info{};
    MV_CC_PIXEL_CONVERT_PARAM convert_param{};

    image_transport::CameraPublisher                        pub;
    std::unique_ptr<camera_info_manager::CameraInfoManager> info_mgr;
    sensor_msgs::msg::Image                                 publish_msg;
    sensor_msgs::msg::CameraInfo                            camera_info_msg;

    std::string camera_name;
    std::string frame_id;
    std::string topic;
    std::string camera_info_url;
    std::string configured_pixel_format;

    std::thread       capture_thread;
    std::thread       publish_thread;
    std::atomic<bool> running{false};
    bool              started = false;

    std::array<FrameSlot, kRingSize> ring;
    std::atomic<uint64_t> write_seq{0};
    std::atomic<uint64_t> read_seq{0};

    int fail_count = 0;

    std::chrono::steady_clock::time_point last_debug_log_time =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_restart_time =
        std::chrono::steady_clock::now();

    bool     output_direct_rgb = false;
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

        if (!initializeCameras()) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize cameras.");
            rclcpp::shutdown();
            return;
        }

        for (auto& cam_ptr : cameras_) {
            declareCameraParameters(*cam_ptr);
            startCamera(*cam_ptr);
        }

        params_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&HikCameraRos2DriverNode::dynamicParametersCallback,
                      this, std::placeholders::_1));

        for (auto& cam_ptr : cameras_) {
            auto& cam = *cam_ptr;
            if (!cam.started) {
                continue;
            }
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
            if (cam.handle) {
                MV_CC_StopGrabbing(cam.handle);
                MV_CC_CloseDevice(cam.handle);
                MV_CC_DestroyHandle(&cam.handle);
                cam.handle = nullptr;
            }
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

    bool initializeCameras()
    {
        MV_CC_DEVICE_INFO_LIST device_list{};
        int                    n_ret = MV_OK;

        while (rclcpp::ok()) {
            n_ret = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE,
                                      &device_list);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to enumerate devices, retrying...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            if (device_list.nDeviceNum == 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "No camera found, retrying...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            break;
        }

        RCLCPP_INFO(this->get_logger(), "Found camera count = %u",
                    device_list.nDeviceNum);

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

                const MV_CC_DEVICE_INFO* matched_dev = nullptr;
                uint32_t matched_dev_idx = 0;
                bool found = false;

                for (uint32_t di = 0; di < device_list.nDeviceNum; ++di) {
                    const auto* dev = device_list.pDeviceInfo[di];
                    const auto serial = getSerial(dev);
                    if (!serial.empty() && serial == wanted_serial) {
                        matched_dev = dev;
                        matched_dev_idx = di;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    RCLCPP_ERROR(this->get_logger(),
                                 "[%s] Requested serial '%s' not found. "
                                 "This slot will remain empty.",
                                 cam->prefix.c_str(),
                                 wanted_serial.c_str());
                    continue;
                }

                cam->model_name = getModelName(matched_dev);
                cam->device_type = getDeviceTypeString(matched_dev);

                RCLCPP_INFO(this->get_logger(),
                            "[%s] selected device idx=%u type=%s serial=%s model=%s",
                            cam->prefix.c_str(),
                            matched_dev_idx,
                            cam->device_type.c_str(),
                            cam->serial.c_str(),
                            cam->model_name.c_str());

                n_ret = MV_CC_CreateHandle(
                    &cam->handle,
                    const_cast<MV_CC_DEVICE_INFO*>(matched_dev));
                if (n_ret != MV_OK) {
                    RCLCPP_ERROR(this->get_logger(),
                                 "[%s] Failed to create camera handle! ret=%x",
                                 cam->prefix.c_str(), n_ret);
                    continue;
                }

                n_ret = MV_CC_OpenDevice(cam->handle);
                if (n_ret != MV_OK) {
                    RCLCPP_ERROR(this->get_logger(),
                                 "[%s] Failed to open camera device! ret=%x",
                                 cam->prefix.c_str(), n_ret);
                    MV_CC_DestroyHandle(&cam->handle);
                    cam->handle = nullptr;
                    continue;
                }

                n_ret = MV_CC_GetImageInfo(cam->handle, &cam->img_info);
                if (n_ret != MV_OK) {
                    RCLCPP_ERROR(this->get_logger(),
                                 "[%s] Failed to get camera image info! ret=%x",
                                 cam->prefix.c_str(), n_ret);
                    MV_CC_CloseDevice(cam->handle);
                    MV_CC_DestroyHandle(&cam->handle);
                    cam->handle = nullptr;
                    continue;
                }

                cam->convert_param.nWidth = cam->img_info.nWidthValue;
                cam->convert_param.nHeight = cam->img_info.nHeightValue;
                cam->convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

                cameras_.push_back(std::move(cam));
            }

            return !cameras_.empty();
        }

        const uint32_t available = device_list.nDeviceNum;
        const uint32_t use_n =
            std::min<uint32_t>(static_cast<uint32_t>(camera_count_), available);

        cameras_.reserve(use_n);

        for (uint32_t i = 0; i < use_n; ++i) {
            auto* dev_info = device_list.pDeviceInfo[i];

            auto cam = std::make_unique<CameraContext>();
            cam->index = static_cast<int>(i);
            cam->prefix = "cam" + std::to_string(cam->index);
            cam->serial = getSerial(dev_info);
            cam->model_name = getModelName(dev_info);
            cam->device_type = getDeviceTypeString(dev_info);

            RCLCPP_INFO(this->get_logger(),
                        "[%s] selected device idx=%u type=%s serial=%s model=%s",
                        cam->prefix.c_str(),
                        i,
                        cam->device_type.c_str(),
                        cam->serial.c_str(),
                        cam->model_name.c_str());

            n_ret = MV_CC_CreateHandle(&cam->handle, dev_info);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to create camera handle! ret=%x",
                             cam->prefix.c_str(), n_ret);
                continue;
            }

            n_ret = MV_CC_OpenDevice(cam->handle);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to open camera device! ret=%x",
                             cam->prefix.c_str(), n_ret);
                MV_CC_DestroyHandle(&cam->handle);
                cam->handle = nullptr;
                continue;
            }

            n_ret = MV_CC_GetImageInfo(cam->handle, &cam->img_info);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to get camera image info! ret=%x",
                             cam->prefix.c_str(), n_ret);
                MV_CC_CloseDevice(cam->handle);
                MV_CC_DestroyHandle(&cam->handle);
                cam->handle = nullptr;
                continue;
            }

            cam->convert_param.nWidth = cam->img_info.nWidthValue;
            cam->convert_param.nHeight = cam->img_info.nHeightValue;
            cam->convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

            cameras_.push_back(std::move(cam));
        }

        return !cameras_.empty();
    }

    void declareCameraParameters(CameraContext& cam)
    {
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        MVCC_FLOATVALUE                          f_value{};
        param_desc.integer_range.resize(1);
        param_desc.integer_range[0].step = 1;

        param_desc.description =
            "[" + cam.prefix + "] Acquisition frame rate in Hz";
        if (MV_CC_GetFloatValue(cam.handle, "AcquisitionFrameRate", &f_value) ==
            MV_OK) {
            param_desc.integer_range[0].from_value = f_value.fMin;
            param_desc.integer_range[0].to_value = f_value.fMax;
        }

        const double acquisition_frame_rate =
            this->declare_parameter<double>(
                cam.prefix + ".acquisition_frame_rate", 120.0, param_desc);

        param_desc.description =
            "[" + cam.prefix + "] Exposure time in microseconds";
        if (MV_CC_GetFloatValue(cam.handle, "ExposureTime", &f_value) == MV_OK) {
            param_desc.integer_range[0].from_value = f_value.fMin;
            param_desc.integer_range[0].to_value = f_value.fMax;
        }

        const int exposure_time =
            this->declare_parameter<int>(
                cam.prefix + ".exposure_time", 5000, param_desc);

        param_desc.description = "[" + cam.prefix + "] Gain";
        if (MV_CC_GetFloatValue(cam.handle, "Gain", &f_value) == MV_OK) {
            param_desc.integer_range[0].from_value = f_value.fMin;
            param_desc.integer_range[0].to_value = f_value.fMax;
        }

        const double gain = this->declare_parameter<double>(
            cam.prefix + ".gain",
            (MV_CC_GetFloatValue(cam.handle, "Gain", &f_value) == MV_OK)
                ? f_value.fCurValue
                : 0.0,
            param_desc);

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
        cam.camera_info_url = this->declare_parameter<std::string>(
            cam.prefix + ".camera_info_url",
            "package://hik_camera_ros2_driver/config/camera_info.yaml");

        (void)acquisition_frame_rate;
        (void)exposure_time;
        (void)gain;
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

    void startCamera(CameraContext& cam)
    {
        auto qos = use_sensor_data_qos_ ? rmw_qos_profile_sensor_data
                                        : rmw_qos_profile_default;

        cam.pub = image_transport::create_camera_publisher(this, cam.topic, qos);

        MV_CC_SetEnumValueByString(cam.handle, "AcquisitionMode", "Continuous");
        MV_CC_SetEnumValueByString(cam.handle, "TriggerMode", "Off");
        MV_CC_SetEnumValueByString(cam.handle, "ExposureAuto", "Off");
        MV_CC_SetEnumValueByString(cam.handle, "GainAuto", "Off");

        if (cam.device_type == "GigE") {
            MV_CC_SetIntValue(cam.handle, "GevSCPSPacketSize", 1500);
        }

        const double acquisition_frame_rate =
            this->get_parameter(cam.prefix + ".acquisition_frame_rate").as_double();
        const int exposure_time =
            static_cast<int>(this->get_parameter(cam.prefix + ".exposure_time").as_int());
        const double gain =
            this->get_parameter(cam.prefix + ".gain").as_double();
        const std::string pixel_format =
            this->get_parameter(cam.prefix + ".pixel_format").as_string();

        MV_CC_SetBoolValue(cam.handle, "AcquisitionFrameRateEnable", true);
        MV_CC_SetFloatValue(cam.handle, "AcquisitionFrameRate",
                            static_cast<float>(acquisition_frame_rate));
        MV_CC_SetFloatValue(cam.handle, "ExposureTime",
                            static_cast<float>(exposure_time));
        MV_CC_SetFloatValue(cam.handle, "Gain",
                            static_cast<float>(gain));

        int status = MV_CC_SetEnumValueByString(cam.handle, "PixelFormat",
                                                pixel_format.c_str());
        if (status != MV_OK) {
            RCLCPP_ERROR(this->get_logger(),
                         "[%s] Failed to set PixelFormat=%s, status=%d",
                         cam.prefix.c_str(), pixel_format.c_str(), status);
            cam.started = false;
            return;
        }
        cam.configured_pixel_format = pixel_format;
        cam.output_direct_rgb = (pixel_format == "RGB8Packed");

        MV_CC_SetImageNodeNum(cam.handle, 15);

        MVCC_INTVALUE payload{};
        if (MV_CC_GetIntValue(cam.handle, "PayloadSize", &payload) == MV_OK &&
            payload.nCurValue > 0) {
            cam.payload_size = static_cast<uint32_t>(payload.nCurValue);
        } else {
            cam.payload_size = cam.img_info.nWidthValue * cam.img_info.nHeightValue * 3U;
        }

        for (auto& slot : cam.ring) {
            slot.data.resize(static_cast<size_t>(cam.img_info.nWidthValue) *
                             static_cast<size_t>(cam.img_info.nHeightValue) * 3U);
        }

        cam.publish_msg.header.frame_id = cam.frame_id;
        cam.publish_msg.encoding = "rgb8";
        cam.publish_msg.data.resize(static_cast<size_t>(cam.img_info.nWidthValue) *
                                    static_cast<size_t>(cam.img_info.nHeightValue) * 3U);

        if (use_hardware_timestamp_) {
            tryConfigureTimestampFrequency(cam);
        }

        if (!startGrabbingWithRetry(cam)) {
            cam.started = false;
            return;
        }

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

        cam.started = true;

        RCLCPP_INFO(this->get_logger(),
                    "[%s] Camera ready. type=%s serial=%s pixel_format=%s gpu_debayer=%s",
                    cam.prefix.c_str(),
                    cam.device_type.c_str(),
                    cam.serial.c_str(),
                    cam.configured_pixel_format.c_str(),
#if HIK_HAS_OPENCV_CUDA
                    (enable_gpu_debayer_ ? "enabled_if_raw" : "disabled"));
#else
                    "not_available");
#endif
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
    bool gpuDebayer(CameraContext& cam,
                    const MV_FRAME_OUT& out_frame,
                    FrameSlot& slot)
    {
        const int width = static_cast<int>(out_frame.stFrameInfo.nWidth);
        const int height = static_cast<int>(out_frame.stFrameInfo.nHeight);

        cv::Mat cpu_bayer(height, width, CV_8UC1,
                          const_cast<unsigned char*>(out_frame.pBufAddr));
        cam.gpu_bayer_.upload(cpu_bayer);

        int code = -1;

        #ifdef COLOR_BayerRG2RGB_MHT
        code = cv::COLOR_BayerRG2RGB_MHT;
        #elif defined(COLOR_BayerRGGB2RGB_MHT)
        code = cv::COLOR_BayerRGGB2RGB_MHT;
        #elif defined(COLOR_BayerRG2RGB)
        code = cv::COLOR_BayerRG2RGB;
        #elif defined(COLOR_BayerRGGB2RGB)
        code = cv::COLOR_BayerRGGB2RGB;
        #else
        return false;
        #endif

        cv::cuda::demosaicing(cam.gpu_bayer_, cam.gpu_rgb_, code, 3);

        const size_t dst_size =
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;
        if (slot.data.size() < dst_size) {
            slot.data.resize(dst_size);
        }

        cv::Mat cpu_rgb(height, width, CV_8UC3, slot.data.data());
        cam.gpu_rgb_.download(cpu_rgb);
        return true;
    }
#endif

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
        cam.convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

        const int cvt_ret = MV_CC_ConvertPixelType(cam.handle, &cam.convert_param);
        if (cvt_ret != MV_OK) {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] ConvertPixelType failed: [%x]",
                        cam.prefix.c_str(), cvt_ret);
            return false;
        }
        return true;
    }

    bool fillSlotFromFrame(CameraContext& cam,
                           const MV_FRAME_OUT& out_frame,
                           FrameSlot& slot)
    {
        slot.width = out_frame.stFrameInfo.nWidth;
        slot.height = out_frame.stFrameInfo.nHeight;
        slot.stamp = makeFrameStamp(cam, out_frame);

        if (cam.output_direct_rgb &&
            out_frame.stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed) {
            const size_t bytes = static_cast<size_t>(out_frame.stFrameInfo.nFrameLen);
            if (slot.data.size() < bytes) {
                slot.data.resize(bytes);
            }
            std::memcpy(slot.data.data(), out_frame.pBufAddr, bytes);
            slot.valid = true;
            return true;
        }

#if HIK_HAS_OPENCV_CUDA
        if (enable_gpu_debayer_ && !cam.output_direct_rgb) {
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
        (void)cam;
        return;
    }

    void captureLoop(CameraContext* cam)
    {
        if (cam == nullptr || !cam->started) {
            return;
        }

        MV_FRAME_OUT out_frame{};
        RCLCPP_INFO(this->get_logger(), "[%s] Capture thread started.",
                    cam->prefix.c_str());

        while (rclcpp::ok() && cam->running) {
            const int n_ret =
                MV_CC_GetImageBuffer(cam->handle, &out_frame, frame_timeout_ms_);

            if (n_ret == MV_OK) {
                const uint64_t seq = cam->write_seq.load(std::memory_order_relaxed);
                FrameSlot& slot = cam->ring[seq % kRingSize];

                const bool ok = fillSlotFromFrame(*cam, out_frame, slot);
                MV_CC_FreeImageBuffer(cam->handle, &out_frame);

                if (ok) {
                    cam->write_seq.store(seq + 1, std::memory_order_release);
                    cam->fail_count = 0;
                } else {
                    cam->fail_count++;
                }

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

            cam->publish_msg.header.stamp = slot.stamp;
            cam->publish_msg.header.frame_id = cam->frame_id;
            cam->publish_msg.height = slot.height;
            cam->publish_msg.width = slot.width;
            cam->publish_msg.step = slot.width * 3U;
            cam->publish_msg.encoding = "rgb8";

            cam->publish_msg.data.swap(slot.data);

            cam->camera_info_msg.header = cam->publish_msg.header;
            cam->pub.publish(cam->publish_msg, cam->camera_info_msg);

            cam->publish_msg.data.swap(slot.data);

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
            if (!cam || !cam->handle) {
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

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        params_callback_handle_;
};

}  // namespace hik_camera_ros2_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    hik_camera_ros2_driver::HikCameraRos2DriverNode)