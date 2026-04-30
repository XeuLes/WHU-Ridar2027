# ros2 launch camera_field_calib_tool pnp_calib_new.launch.py
#   image_topic:=/camera0/image
#   out_matrix_yaml:=/home/user/code/T-DT_Radar-main/src/camera_field_calib_tool/config/out_matrix_new.yaml
#   reference_points_yaml:=/home/user/code/T-DT_Radar-main/src/camera_field_calib_tool/config/reference_points.yaml

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -----------------------------
    # Hik camera driver defaults
    # -----------------------------
    hik_pkg_share = get_package_share_directory("hik_camera_ros2_driver")
    hik_default_params = os.path.join(hik_pkg_share, "config", "camera_params.yaml")

    params_file = LaunchConfiguration("params_file")
    log_level = LaunchConfiguration("log_level")
    start_hik_camera = LaunchConfiguration("start_hik_camera")
    start_static_image_publisher = LaunchConfiguration("start_static_image_publisher")

    # -----------------------------
    # Calib tool defaults
    # -----------------------------
    tool_pkg_share = FindPackageShare("camera_field_calib_tool")
    default_intrinsics = PathJoinSubstitution([tool_pkg_share, "config", "camera_intrinsics.yaml"])
    default_out_matrix = PathJoinSubstitution([tool_pkg_share, "config", "out_matrix_new.yaml"])
    default_reference_points = PathJoinSubstitution([tool_pkg_share, "config", "reference_points.yaml"])

    # -----------------------------
    # Environment variables
    # -----------------------------
    stdout_linebuf_envvar = SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    # -----------------------------
    # Launch arguments (Hik driver)
    # -----------------------------
    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=hik_default_params,
        description="Hik camera driver parameter file path",
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level for hik_camera_ros2_driver (debug/info/warn/error/fatal)",
    )

    declare_start_hik_camera_cmd = DeclareLaunchArgument(
        "start_hik_camera",
        default_value="true",
        description="Whether to start hik_camera_ros2_driver for live calibration input.",
    )

    declare_start_static_image_publisher_cmd = DeclareLaunchArgument(
        "start_static_image_publisher",
        default_value="false",
        description="Whether to publish static_image_file to image_topic for calibration.",
    )

    declare_static_image_file_cmd = DeclareLaunchArgument(
        "static_image_file",
        default_value="",
        description="Image file to publish when start_static_image_publisher is true.",
    )

    declare_static_image_rate_cmd = DeclareLaunchArgument(
        "static_image_rate",
        default_value="5.0",
        description="Publish rate for the static calibration image.",
    )

    declare_static_image_frame_cmd = DeclareLaunchArgument(
        "static_image_frame",
        default_value="calib_camera",
        description="Frame id used by image_publisher for the static calibration image.",
    )

    # -----------------------------
    # Launch arguments (PnP tool new)
    # -----------------------------
    declare_image_topic_cmd = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera0/image",
        description="Input image topic for pnp_calib_node_new (sensor_msgs/Image). "
                    "Set this to hik driver output if needed.",
    )

    declare_intrinsics_yaml_cmd = DeclareLaunchArgument(
        "intrinsics_yaml",
        default_value=default_intrinsics,
        description="Path to camera intrinsics YAML.",
    )

    declare_out_matrix_yaml_cmd = DeclareLaunchArgument(
        "out_matrix_yaml",
        default_value=default_out_matrix,
        description="Output path for extrinsics YAML. "
                    "Recommend setting to a writable workspace path.",
    )

    declare_reference_points_yaml_cmd = DeclareLaunchArgument(
        "reference_points_yaml",
        default_value=default_reference_points,
        description="Path to reference world points YAML.",
    )

    declare_use_refine_cmd = DeclareLaunchArgument(
        "use_refine",
        default_value="true",
        description="If true, run solvePnPRefineLM after solvePnP.",
    )

    declare_pnp_method_cmd = DeclareLaunchArgument(
        "pnp_method",
        default_value="0",
        description="OpenCV solvePnP method flag. 0 means default (EPnP).",
    )

    declare_min_points_cmd = DeclareLaunchArgument(
        "min_points",
        default_value="6",
        description="Minimum required point pairs before solving.",
    )

    declare_freeze_first_frame_cmd = DeclareLaunchArgument(
        "freeze_first_frame",
        default_value="false",
        description="If true, pnp_calib_node_new keeps the first received image as a static frame.",
    )

    declare_shutdown_on_save_cmd = DeclareLaunchArgument(
        "shutdown_on_save",
        default_value="false",
        description="If true, pnp_calib_node_new exits after a successful save.",
    )

    # -----------------------------
    # Nodes
    # -----------------------------
    start_hik_camera_cmd = Node(
        name="hik_camera_ros2_driver",
        package="hik_camera_ros2_driver",
        executable="hik_camera_ros2_driver_node",
        parameters=[params_file],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
        condition=IfCondition(start_hik_camera),
    )

    start_static_image_publisher_cmd = Node(
        name="calib_static_image_publisher",
        package="image_publisher",
        executable="image_publisher_node",
        output="screen",
        arguments=[LaunchConfiguration("static_image_file")],
        parameters=[{
            "publish_rate": LaunchConfiguration("static_image_rate"),
            "frame_id": LaunchConfiguration("static_image_frame"),
        }],
        remappings=[
            ("image_raw", LaunchConfiguration("image_topic")),
            ("camera_info", "/calib_camera_info"),
        ],
        condition=IfCondition(start_static_image_publisher),
    )

    start_pnp_calib_new_cmd = Node(
        name="pnp_calib_node_new",
        package="camera_field_calib_tool",
        executable="pnp_calib_node_new",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "intrinsics_yaml": LaunchConfiguration("intrinsics_yaml"),
            "out_matrix_yaml": LaunchConfiguration("out_matrix_yaml"),
            "reference_points_yaml": LaunchConfiguration("reference_points_yaml"),
            "use_refine": LaunchConfiguration("use_refine"),
            "pnp_method": LaunchConfiguration("pnp_method"),
            "min_points": LaunchConfiguration("min_points"),
            "freeze_first_frame": LaunchConfiguration("freeze_first_frame"),
            "shutdown_on_save": LaunchConfiguration("shutdown_on_save"),
        }],
    )

    # -----------------------------
    # LaunchDescription
    # -----------------------------
    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)

    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_start_hik_camera_cmd)
    ld.add_action(declare_start_static_image_publisher_cmd)
    ld.add_action(declare_static_image_file_cmd)
    ld.add_action(declare_static_image_rate_cmd)
    ld.add_action(declare_static_image_frame_cmd)

    ld.add_action(declare_image_topic_cmd)
    ld.add_action(declare_intrinsics_yaml_cmd)
    ld.add_action(declare_out_matrix_yaml_cmd)
    ld.add_action(declare_reference_points_yaml_cmd)
    ld.add_action(declare_use_refine_cmd)
    ld.add_action(declare_pnp_method_cmd)
    ld.add_action(declare_min_points_cmd)
    ld.add_action(declare_freeze_first_frame_cmd)
    ld.add_action(declare_shutdown_on_save_cmd)

    ld.add_action(start_hik_camera_cmd)
    ld.add_action(start_static_image_publisher_cmd)
    ld.add_action(start_pnp_calib_new_cmd)

    return ld
