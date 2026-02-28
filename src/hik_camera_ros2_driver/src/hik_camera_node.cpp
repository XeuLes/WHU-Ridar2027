#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "MvCameraControl.h"

#include "camera_info_manager/camera_info_manager.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/utilities.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace hik_camera_ros2_driver {

// -----------------------------
// Helper: safe C-string to std::string
// -----------------------------
[[maybe_unused]] static std::string cstr_to_string(const char* s)
{
    return (s == nullptr) ? std::string() : std::string(s);
}

// -----------------------------
// Camera context for multi-cam
// -----------------------------
struct CameraContext {
    // index and naming
    int         index = 0;
    std::string prefix;  // "cam0" / "cam1"

    // MV handle & info
    void*                     handle = nullptr;
    MV_IMAGE_BASIC_INFO       img_info{};
    MV_CC_PIXEL_CONVERT_PARAM convert_param{};

    // ROS publishing
    image_transport::CameraPublisher                        pub;
    std::unique_ptr<camera_info_manager::CameraInfoManager> info_mgr;
    sensor_msgs::msg::Image                                 image_msg;
    sensor_msgs::msg::CameraInfo                            camera_info_msg;

    // parameters
    std::string camera_name;
    std::string frame_id;
    std::string topic;
    std::string camera_info_url;

    // capture thread
    std::thread th;
    int         fail_count = 0;

    // last log
    std::chrono::steady_clock::time_point last_log_time =
        std::chrono::steady_clock::now();
};

class HikCameraRos2DriverNode : public rclcpp::Node {
public:
    explicit HikCameraRos2DriverNode(const rclcpp::NodeOptions& options)
        : Node("hik_camera_ros2_driver", options)
    {
        RCLCPP_INFO(this->get_logger(),
                    "Starting HikCameraRos2DriverNode (multi-cam)!");

        declareGlobalParameters();

        // Enumerate and open requested cameras
        if (!initializeCameras()) {
            RCLCPP_FATAL(this->get_logger(),
                         "Failed to initialize cameras.");
            rclcpp::shutdown();
            return;
        }

        // Declare camera parameters and start grabbing/publishing for each
        // camera
        for (auto& cam : cameras_) {
            declareCameraParameters(cam);
            startCamera(cam);
        }

        params_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&HikCameraRos2DriverNode::dynamicParametersCallback,
                      this, std::placeholders::_1));

        // Start capture threads
        for (auto& cam : cameras_) {
            cam.th = std::thread(&HikCameraRos2DriverNode::captureLoop,
                                 this, &cam);
        }

        RCLCPP_INFO(this->get_logger(),
                    "HikCameraRos2DriverNode started with %zu camera(s).",
                    cameras_.size());
    }

    ~HikCameraRos2DriverNode() override
    {
        // Stop threads first
        for (auto& cam : cameras_) {
            if (cam.th.joinable()) {
                cam.th.join();
            }
        }

        // Close MV devices
        for (auto& cam : cameras_) {
            if (cam.handle) {
                MV_CC_StopGrabbing(cam.handle);
                MV_CC_CloseDevice(cam.handle);
                MV_CC_DestroyHandle(&cam.handle);
                cam.handle = nullptr;
            }
        }

        RCLCPP_INFO(this->get_logger(),
                    "HikCameraRos2DriverNode destroyed!");
    }

private:
    // -----------------------------
    // Global parameters
    // -----------------------------
    void declareGlobalParameters()
    {
        camera_count_ = this->declare_parameter<int>("camera_count", 1);
        if (camera_count_ < 1)
            camera_count_ = 1;
        if (camera_count_ > 2)
            camera_count_ = 2;

        use_sensor_data_qos_ =
            this->declare_parameter<bool>("use_sensor_data_qos", true);

        // Optional: specify serials to bind camera order deterministically.
        // Example:
        //   camera_serials: ["DA1234567", "DA7654321"]
        camera_serials_ = this->declare_parameter<std::vector<std::string>>(
            "camera_serials", std::vector<std::string>{});
    }

    // -----------------------------
    // MV device serial extraction
    // NOTE: This may need alignment with your SDK's exact device info
    // fields.
    // -----------------------------
    static std::string getSerial(const MV_CC_DEVICE_INFO* dev_info)
    {
        if (!dev_info)
            return {};

        // Many Hik SDK builds expose USB device info in
        // SpecialInfo.stUsb3VInfo / stUsb2VInfo. We try common candidates.
        // If your SDK differs, adjust here. The following is intentionally
        // defensive.
        try {
            // USB3/USB2: may have chSerialNumber or similar.
            // If these fields do not exist in your SDK headers, replace
            // with the correct ones. Common patterns:
            //   dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber
            //   dev_info->SpecialInfo.stUsb2VInfo.chSerialNumber
#ifdef MV_USB3_DEVICE
            // compile-time availability varies; keep minimal assumptions
#endif
            // Fallback: user-defined name / model, etc. if serial not
            // available. We cannot reliably guess without your exact header
            // structure, so return empty and rely on index order if serials
            // are not obtainable.
        }
        catch (...) {
            // ignore
        }
        return {};
    }

    // -----------------------------
    // Enumerate devices and open N cameras
    // -----------------------------
    bool initializeCameras()
    {
        MV_CC_DEVICE_INFO_LIST device_list{};
        int                    n_ret = MV_OK;

        // Enum until at least 1 device
        while (rclcpp::ok()) {
            n_ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
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
            RCLCPP_INFO(this->get_logger(), "Found camera count = %u",
                        device_list.nDeviceNum);
            break;
        }

        const uint32_t available = device_list.nDeviceNum;
        const uint32_t need = static_cast<uint32_t>(camera_count_);

        if (available < need) {
            RCLCPP_WARN(this->get_logger(),
                        "Requested %u camera(s) but only %u available. "
                        "Will use %u camera(s).",
                        need, available, available);
        }
        const uint32_t use_n = std::min<uint32_t>(need, available);

        // Decide which indices to use
        std::vector<uint32_t> selected_indices;

        if (!camera_serials_.empty()) {
            // Attempt to match by serials if obtainable; otherwise fall
            // back to first N. For many SDK variants, serial extraction
            // needs exact field names; if getSerial() returns empty,
            // matching will fail and we degrade gracefully.
            selected_indices.reserve(use_n);
            for (uint32_t si = 0; si < use_n && si < camera_serials_.size();
                 ++si) {
                const std::string& wanted = camera_serials_[si];
                bool               found = false;
                for (uint32_t di = 0; di < available; ++di) {
                    const auto*       dev = device_list.pDeviceInfo[di];
                    const std::string serial = getSerial(dev);
                    if (!serial.empty() && serial == wanted) {
                        selected_indices.push_back(di);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    RCLCPP_WARN(this->get_logger(),
                                "Could not match serial '%s'. Falling back "
                                "to index-based selection.",
                                wanted.c_str());
                    selected_indices.clear();
                    break;
                }
            }
        }

        if (selected_indices.empty()) {
            // Default: first N devices
            selected_indices.reserve(use_n);
            for (uint32_t i = 0; i < use_n; ++i) {
                selected_indices.push_back(i);
            }
        }

        // Create contexts and open cameras
        cameras_.clear();
        cameras_.reserve(selected_indices.size());

        for (size_t i = 0; i < selected_indices.size(); ++i) {
            const uint32_t dev_idx = selected_indices[i];
            auto*          dev_info = device_list.pDeviceInfo[dev_idx];

            CameraContext cam;
            cam.index = static_cast<int>(i);
            cam.prefix = "cam" + std::to_string(cam.index);

            // Create and open handle
            n_ret = MV_CC_CreateHandle(&cam.handle, dev_info);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to create camera handle!",
                             cam.prefix.c_str());
                return false;
            }

            n_ret = MV_CC_OpenDevice(cam.handle);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to open camera device!",
                             cam.prefix.c_str());
                return false;
            }

            // Get image info
            n_ret = MV_CC_GetImageInfo(cam.handle, &cam.img_info);
            if (n_ret != MV_OK) {
                RCLCPP_ERROR(this->get_logger(),
                             "[%s] Failed to get camera image info!",
                             cam.prefix.c_str());
                return false;
            }

            // Init convert param (width/height will be updated per frame if
            // needed)
            cam.convert_param.nWidth = cam.img_info.nWidthValue;
            cam.convert_param.nHeight = cam.img_info.nHeightValue;
            cam.convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;

            // Pre-allocate some space; actual resize will occur per frame
            cam.image_msg.data.reserve(cam.img_info.nHeightMax *
                                       cam.img_info.nWidthMax * 3);

            cameras_.push_back(std::move(cam));
        }

        return !cameras_.empty();
    }

    // -----------------------------
    // Per-camera parameter declaration & MV configuration
    // -----------------------------
    void declareCameraParameters(CameraContext& cam)
    {
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        MVCC_FLOATVALUE                          f_value{};
        param_desc.integer_range.resize(1);
        param_desc.integer_range[0].step = 1;

        // Acquisition frame rate
        param_desc.description =
            "[" + cam.prefix + "] Acquisition frame rate in Hz";
        MV_CC_GetFloatValue(cam.handle, "AcquisitionFrameRate", &f_value);
        param_desc.integer_range[0].from_value = f_value.fMin;
        param_desc.integer_range[0].to_value = f_value.fMax;

        const double acquisition_frame_rate =
            this->declare_parameter<double>(
                cam.prefix + ".acquisition_frame_rate", 165.0, param_desc);

        MV_CC_SetBoolValue(cam.handle, "AcquisitionFrameRateEnable", true);
        MV_CC_SetFloatValue(cam.handle, "AcquisitionFrameRate",
                            acquisition_frame_rate);
        RCLCPP_INFO(this->get_logger(), "[%s] Acquisition frame rate: %f",
                    cam.prefix.c_str(), acquisition_frame_rate);

        // Exposure time
        param_desc.description =
            "[" + cam.prefix + "] Exposure time in microseconds";
        MV_CC_GetFloatValue(cam.handle, "ExposureTime", &f_value);
        param_desc.integer_range[0].from_value = f_value.fMin;
        param_desc.integer_range[0].to_value = f_value.fMax;

        const int exposure_time_default = 5000;
        const int exposure_time =
            this->declare_parameter<int>(cam.prefix + ".exposure_time",
                                         exposure_time_default, param_desc);

        MV_CC_SetFloatValue(cam.handle, "ExposureTime",
                            static_cast<float>(exposure_time));
        RCLCPP_INFO(this->get_logger(), "[%s] Exposure time: %d",
                    cam.prefix.c_str(), exposure_time);

        // Gain
        param_desc.description = "[" + cam.prefix + "] Gain";
        MV_CC_GetFloatValue(cam.handle, "Gain", &f_value);
        param_desc.integer_range[0].from_value = f_value.fMin;
        param_desc.integer_range[0].to_value = f_value.fMax;

        const double gain = this->declare_parameter<double>(
            cam.prefix + ".gain", f_value.fCurValue, param_desc);

        MV_CC_SetFloatValue(cam.handle, "Gain", gain);
        RCLCPP_INFO(this->get_logger(), "[%s] Gain: %f", cam.prefix.c_str(),
                    gain);

        // ADC Bit Depth
        param_desc.description = "[" + cam.prefix + "] ADC Bit Depth";
        param_desc.additional_constraints =
            "Supported values: Bits_8, Bits_12";
        const std::string adc_bit_depth =
            this->declare_parameter<std::string>(
                cam.prefix + ".adc_bit_depth", "Bits_8", param_desc);

        int status = MV_CC_SetEnumValueByString(cam.handle, "ADCBitDepth",
                                                adc_bit_depth.c_str());
        if (status == MV_OK) {
            RCLCPP_INFO(this->get_logger(), "[%s] ADC Bit Depth set to %s",
                        cam.prefix.c_str(), adc_bit_depth.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(),
                        "[%s] Failed to set ADC Bit Depth to %s, status = "
                        "%d. Skip.",
                        cam.prefix.c_str(), adc_bit_depth.c_str(), status);
        }

        // Pixel format
        param_desc.description = "[" + cam.prefix + "] Pixel Format";
        const std::string pixel_format =
            this->declare_parameter<std::string>(
                cam.prefix + ".pixel_format", "RGB8Packed", param_desc);

        status = MV_CC_SetEnumValueByString(cam.handle, "PixelFormat",
                                            pixel_format.c_str());
        if (status == MV_OK) {
            RCLCPP_INFO(this->get_logger(), "[%s] Pixel Format set to %s",
                        cam.prefix.c_str(), pixel_format.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(),
                         "[%s] Failed to set Pixel Format, status = %d",
                         cam.prefix.c_str(), status);
        }

        // ROS naming/topic/frame parameters
        cam.camera_name = this->declare_parameter<std::string>(
            cam.prefix + ".camera_name", cam.prefix);
        cam.frame_id = this->declare_parameter<std::string>(
            cam.prefix + ".frame_id", cam.camera_name + "_optical_frame");
        cam.topic = this->declare_parameter<std::string>(
            cam.prefix + ".camera_topic", cam.camera_name + "/image");

        cam.camera_info_url = this->declare_parameter<std::string>(
            cam.prefix + ".camera_info_url",
            "package://hik_camera_ros2_driver/config/camera_info.yaml");
    }

    void startCamera(CameraContext& cam)
    {
        auto qos = use_sensor_data_qos_ ? rmw_qos_profile_sensor_data
                                        : rmw_qos_profile_default;

        cam.pub =
            image_transport::create_camera_publisher(this, cam.topic, qos);

        // Start grabbing
        MV_CC_StartGrabbing(cam.handle);

        // Load camera info
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
    }

    // -----------------------------
    // Capture loop per camera (thread)
    // -----------------------------
    void captureLoop(CameraContext* cam)
    {
        if (cam == nullptr)
            return;

        MV_FRAME_OUT out_frame{};
        RCLCPP_INFO(this->get_logger(), "[%s] Publishing image!",
                    cam->prefix.c_str());

        cam->image_msg.header.frame_id = cam->frame_id;
        cam->image_msg.encoding = "rgb8";

        while (rclcpp::ok()) {
            const int n_ret =
                MV_CC_GetImageBuffer(cam->handle, &out_frame, 1000);
            if (MV_OK == n_ret) {
                const uint32_t width = out_frame.stFrameInfo.nWidth;
                const uint32_t height = out_frame.stFrameInfo.nHeight;
                const size_t   dst_size = static_cast<size_t>(width) *
                                        static_cast<size_t>(height) * 3;

                // Ensure buffer is allocated BEFORE convert; pointer must
                // remain valid.
                cam->image_msg.data.resize(dst_size);

                cam->convert_param.nWidth = width;
                cam->convert_param.nHeight = height;
                cam->convert_param.pDstBuffer = cam->image_msg.data.data();
                cam->convert_param.nDstBufferSize =
                    static_cast<unsigned int>(cam->image_msg.data.size());
                cam->convert_param.pSrcData = out_frame.pBufAddr;
                cam->convert_param.nSrcDataLen =
                    out_frame.stFrameInfo.nFrameLen;
                cam->convert_param.enSrcPixelType =
                    out_frame.stFrameInfo.enPixelType;
                cam->convert_param.enDstPixelType =
                    PixelType_Gvsp_RGB8_Packed;

                const int cvt_ret = MV_CC_ConvertPixelType(
                    cam->handle, &cam->convert_param);
                if (cvt_ret != MV_OK) {
                    RCLCPP_WARN(this->get_logger(),
                                "[%s] ConvertPixelType failed: [%x]",
                                cam->prefix.c_str(), cvt_ret);
                    MV_CC_FreeImageBuffer(cam->handle, &out_frame);
                    cam->fail_count++;
                } else {
                    cam->image_msg.header.stamp = this->now();
                    cam->image_msg.height = height;
                    cam->image_msg.width = width;
                    cam->image_msg.step = width * 3;

                    cam->camera_info_msg.header = cam->image_msg.header;
                    cam->pub.publish(cam->image_msg, cam->camera_info_msg);

                    MV_CC_FreeImageBuffer(cam->handle, &out_frame);
                    cam->fail_count = 0;
                }

                // periodic debug
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        now - cam->last_log_time)
                        .count() >= 3) {
                    MVCC_FLOATVALUE f_value{};
                    MV_CC_GetFloatValue(cam->handle, "ResultingFrameRate",
                                        &f_value);
                    RCLCPP_DEBUG(this->get_logger(),
                                 "[%s] ResultingFrameRate: %f Hz",
                                 cam->prefix.c_str(), f_value.fCurValue);
                    cam->last_log_time = now;
                }

            } else {
                RCLCPP_WARN(this->get_logger(),
                            "[%s] Get buffer failed! nRet: [%x]",
                            cam->prefix.c_str(), n_ret);
                MV_CC_StopGrabbing(cam->handle);
                MV_CC_StartGrabbing(cam->handle);
                cam->fail_count++;
            }

            if (cam->fail_count > 5) {
                RCLCPP_FATAL(this->get_logger(), "[%s] Camera failed!",
                             cam->prefix.c_str());
                rclcpp::shutdown();
                return;
            }
        }
    }

    // -----------------------------
    // Dynamic parameter callback (multi-cam)
    // -----------------------------
    rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
        const std::vector<rclcpp::Parameter>& parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        auto find_cam_by_prefix =
            [&](const std::string& prefix) -> CameraContext* {
            for (auto& c : cameras_) {
                if (c.prefix == prefix)
                    return &c;
            }
            return nullptr;
        };

        for (const auto& param : parameters) {
            const auto& name = param.get_name();

            // expecting name like "cam0.gain" / "cam1.exposure_time"
            const auto dot = name.find('.');
            if (dot == std::string::npos) {
                // Not ours
                continue;
            }
            const std::string prefix = name.substr(0, dot);
            const std::string key = name.substr(dot + 1);

            CameraContext* cam = find_cam_by_prefix(prefix);
            if (!cam) {
                // not ours
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
                                             param.as_double());

            } else if (key == "exposure_time") {
                // declared as int
                if (param.get_type() !=
                    rclcpp::ParameterType::PARAMETER_INTEGER) {
                    result.successful = false;
                    result.reason = "Parameter type mismatch for " + name;
                    return result;
                }
                status =
                    MV_CC_SetFloatValue(cam->handle, "ExposureTime",
                                        static_cast<float>(param.as_int()));

            } else {
                // ignore unknown keys to avoid breaking other node params
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
    // cameras
    std::vector<CameraContext> cameras_;

    // global params
    int                      camera_count_ = 1;
    bool                     use_sensor_data_qos_ = true;
    std::vector<std::string> camera_serials_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        params_callback_handle_;
};

}  // namespace hik_camera_ros2_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    hik_camera_ros2_driver::HikCameraRos2DriverNode)
