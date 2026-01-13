# ros2 launch camera_field_calib_tool pnp_calib.launch.py   image_topic:=/camera/image   out_matrix_yaml:=/home/user/code/T-DT_Radar-main/src/camera_field_calib_tool/config/out_matrix.yaml


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
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

    # -----------------------------
    # Calib tool defaults
    # -----------------------------
    tool_pkg_share = FindPackageShare("camera_field_calib_tool")
    default_intrinsics = PathJoinSubstitution([tool_pkg_share, "config", "camera_intrinsics.yaml"])
    default_out_matrix = PathJoinSubstitution([tool_pkg_share, "config", "out_matrix.yaml"])

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

    # -----------------------------
    # Launch arguments (PnP tool)
    # -----------------------------
    declare_image_topic_cmd = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/image",
        description="Input image topic for pnp_calib_node (sensor_msgs/Image). "
                    "Set this to hik driver output, e.g. /hik_camera/image_raw if needed.",
    )

    declare_intrinsics_yaml_cmd = DeclareLaunchArgument(
        "intrinsics_yaml",
        default_value=default_intrinsics,
        description="Path to camera intrinsics YAML.",
    )

    declare_out_matrix_yaml_cmd = DeclareLaunchArgument(
        "out_matrix_yaml",
        default_value=default_out_matrix,
        description="Output path for extrinsics (out_matrix.yaml). "
                    "Recommend setting to a writable workspace path.",
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
    )

    start_pnp_calib_cmd = Node(
        name="pnp_calib_node",
        package="camera_field_calib_tool",
        executable="pnp_calib_node",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "intrinsics_yaml": LaunchConfiguration("intrinsics_yaml"),
            "out_matrix_yaml": LaunchConfiguration("out_matrix_yaml"),
            "use_refine": LaunchConfiguration("use_refine"),
            "pnp_method": LaunchConfiguration("pnp_method"),
            "min_points": LaunchConfiguration("min_points"),
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

    ld.add_action(declare_image_topic_cmd)
    ld.add_action(declare_intrinsics_yaml_cmd)
    ld.add_action(declare_out_matrix_yaml_cmd)
    ld.add_action(declare_use_refine_cmd)
    ld.add_action(declare_pnp_method_cmd)
    ld.add_action(declare_min_points_cmd)

    ld.add_action(start_hik_camera_cmd)
    ld.add_action(start_pnp_calib_cmd)

    return ld
