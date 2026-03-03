import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Directories
    hik_share = get_package_share_directory("hik_camera_ros2_driver")
    aim_share = get_package_share_directory("auto_aim_fast")

    # Launch args
    hik_params_file = LaunchConfiguration("hik_params_file")
    log_level = LaunchConfiguration("log_level")

    # auto_aim params
    tracker_file = LaunchConfiguration("tracker_params")
    camera_file = LaunchConfiguration("camera_params")
    detector_file = LaunchConfiguration("detector_params")
    qos_file = LaunchConfiguration("qos_params")
    info_file = LaunchConfiguration("camera_info_params")

    # Environment
    stdout_linebuf_envvar = SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    # Declare args
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

    # Directly launch Hik camera driver node
    hik_camera_node = Node(
        package="hik_camera_ros2_driver",
        executable="hik_camera_ros2_driver_node",
        name="hik_camera_ros2_driver",
        output="screen",
        parameters=[hik_params_file],
        arguments=["--ros-args", "--log-level", log_level],
    )

    # auto_aim node
    auto_aim_node = Node(
        package="auto_aim_fast",
        executable="auto_aim_node",
        name="auto_aim_node",
        output="screen",
        parameters=[tracker_file, camera_file, detector_file, qos_file, info_file],
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

    ld.add_action(hik_camera_node)
    ld.add_action(auto_aim_node)

    return ld