import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hik_share = get_package_share_directory("hik_camera_ros2_driver")
    aim_share = get_package_share_directory("auto_aim_fast")

    hik_params_file = LaunchConfiguration("hik_params_file")
    log_level = LaunchConfiguration("log_level")
    tracker_file = LaunchConfiguration("tracker_params")
    camera_file = LaunchConfiguration("camera_params")
    detector_file = LaunchConfiguration("detector_params")
    qos_file = LaunchConfiguration("qos_params")
    info_file = LaunchConfiguration("camera_info_params")

    detector_mode = LaunchConfiguration("detector_mode")
    yolo_engine_path = LaunchConfiguration("yolo_engine_path")
    yolo_input_w = LaunchConfiguration("yolo_input_w")
    yolo_input_h = LaunchConfiguration("yolo_input_h")
    yolo_conf_thresh = LaunchConfiguration("yolo_conf_thresh")
    yolo_kpt_conf_thresh = LaunchConfiguration("yolo_kpt_conf_thresh")
    yolo_max_det = LaunchConfiguration("yolo_max_det")
    yolo_use_fp16 = LaunchConfiguration("yolo_use_fp16")
    yolo_class_id = LaunchConfiguration("yolo_class_id")
    yolo_swap_rb_input = LaunchConfiguration("yolo_swap_rb_input")

    stdout_linebuf_envvar = SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    declare_hik_params = DeclareLaunchArgument(
        "hik_params_file",
        default_value=os.path.join(hik_share, "config", "camera_params.yaml"),
        description="Hik camera parameter file",
    )
    declare_log_level = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="log level",
    )
    declare_tracker = DeclareLaunchArgument(
        "tracker_params",
        default_value=os.path.join(aim_share, "config", "tracker.yaml"),
        description="auto_aim tracker params",
    )
    declare_camera = DeclareLaunchArgument(
        "camera_params",
        default_value=os.path.join(aim_share, "config", "camera.yaml"),
        description="auto_aim camera intrinsics fallback",
    )
    declare_detector = DeclareLaunchArgument(
        "detector_params",
        default_value=os.path.join(aim_share, "config", "detector.yaml"),
        description="auto_aim detector params",
    )
    declare_qos = DeclareLaunchArgument(
        "qos_params",
        default_value=os.path.join(aim_share, "config", "qos.yaml"),
        description="auto_aim qos/publish/debug params",
    )
    declare_info = DeclareLaunchArgument(
        "camera_info_params",
        default_value=os.path.join(aim_share, "config", "camera_info.yaml"),
        description="auto_aim camera_info usage params",
    )
    declare_detector_mode = DeclareLaunchArgument(
        "detector_mode",
        default_value="yolo",
        description="detector mode: legacy or yolo",
    )
    declare_yolo_engine_path = DeclareLaunchArgument(
        "yolo_engine_path",
        default_value=os.path.join(aim_share, "model", "best-fp16.engine"),
        description="TensorRT engine path for YOLO pose detector",
    )
    declare_yolo_input_w = DeclareLaunchArgument(
        "yolo_input_w",
        default_value="640",
        description="YOLO TensorRT input width",
    )
    declare_yolo_input_h = DeclareLaunchArgument(
        "yolo_input_h",
        default_value="640",
        description="YOLO TensorRT input height",
    )
    declare_yolo_conf_thresh = DeclareLaunchArgument(
        "yolo_conf_thresh",
        default_value="0.35",
        description="YOLO object confidence threshold",
    )
    declare_yolo_kpt_conf_thresh = DeclareLaunchArgument(
        "yolo_kpt_conf_thresh",
        default_value="0.35",
        description="YOLO keypoint confidence threshold",
    )
    declare_yolo_max_det = DeclareLaunchArgument(
        "yolo_max_det",
        default_value="32",
        description="YOLO max detections per frame",
    )
    declare_yolo_use_fp16 = DeclareLaunchArgument(
        "yolo_use_fp16",
        default_value="true",
        description="Use FP16 TensorRT inference",
    )
    declare_yolo_class_id = DeclareLaunchArgument(
        "yolo_class_id",
        default_value="0",
        description="Target class id for YOLO detector",
    )
    declare_yolo_swap_rb = DeclareLaunchArgument(
        "yolo_swap_rb_input",
        default_value="false",
        description="Swap R/B channels before YOLO inference",
    )
    hik_camera_node = Node(
        package="hik_camera_ros2_driver",
        executable="hik_camera_ros2_driver_node",
        name="hik_camera_ros2_driver",
        output="screen",
        parameters=[hik_params_file],
        arguments=["--ros-args", "--log-level", log_level],
    )

    auto_aim_node = Node(
        package="auto_aim_fast",
        executable="auto_aim_node",
        name="auto_aim_node",
        output="screen",
        parameters=[
            tracker_file,
            camera_file,
            detector_file,
            qos_file,
            info_file,
            {
                "detector.mode": detector_mode,
                "detector.yolo.engine_path": yolo_engine_path,
                "detector.yolo.input_w": yolo_input_w,
                "detector.yolo.input_h": yolo_input_h,
                "detector.yolo.conf_thresh": yolo_conf_thresh,
                "detector.yolo.kpt_conf_thresh": yolo_kpt_conf_thresh,
                "detector.yolo.max_det": yolo_max_det,
                "detector.yolo.use_fp16_input": yolo_use_fp16,
                "detector.yolo.target_class_id": yolo_class_id,
                "detector.yolo.swap_rb_input": yolo_swap_rb_input,
            },
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)
    ld.add_action(declare_hik_params)
    ld.add_action(declare_log_level)
    ld.add_action(declare_tracker)
    ld.add_action(declare_camera)
    ld.add_action(declare_detector)
    ld.add_action(declare_qos)
    ld.add_action(declare_info)
    ld.add_action(declare_detector_mode)
    ld.add_action(declare_yolo_engine_path)
    ld.add_action(declare_yolo_input_w)
    ld.add_action(declare_yolo_input_h)
    ld.add_action(declare_yolo_conf_thresh)
    ld.add_action(declare_yolo_kpt_conf_thresh)
    ld.add_action(declare_yolo_max_det)
    ld.add_action(declare_yolo_use_fp16)
    ld.add_action(declare_yolo_class_id)
    ld.add_action(declare_yolo_swap_rb)
    ld.add_action(hik_camera_node)
    ld.add_action(auto_aim_node)
    return ld
