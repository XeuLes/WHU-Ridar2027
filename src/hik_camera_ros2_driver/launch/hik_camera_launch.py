import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    bringup_dir = get_package_share_directory("hik_camera_ros2_driver")
    params_file = LaunchConfiguration("params_file")
    log_level = LaunchConfiguration("log_level")
    intra = [{"use_intra_process_comms": True}]

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(bringup_dir, "config", "camera_params.yaml"),
        description="Hik camera params yaml path",
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level", default_value="info", description="log level"
    )

    def rectifier_component(camera_name):
        return ComposableNode(
            package="hik_camera_ros2_driver",
            plugin="hik_camera_ros2_driver::HikCameraRectifierNode",
            name=f"{camera_name}_rectifier",
            parameters=[
                {
                    "input_image_topic": f"/{camera_name}/image",
                    "input_camera_info_topic": f"/{camera_name}/camera_info",
                    "output_image_topic": f"/{camera_name}_rect/image",
                    "output_camera_info_topic": f"/{camera_name}_rect/camera_info",
                    "use_sensor_data_qos": True,
                }
            ],
            extra_arguments=intra,
        )

    start_hik_camera_cmd = ComposableNodeContainer(
        name="hik_camera_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="hik_camera_ros2_driver",
                plugin="hik_camera_ros2_driver::HikCameraRos2DriverNode",
                name="hik_camera_ros2_driver",
                parameters=[
                    params_file,
                    {"camera_count": 3},
                ],
                extra_arguments=intra,
            ),
            rectifier_component("camera0"),
            rectifier_component("camera1"),
            rectifier_component("camera2"),
        ],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(start_hik_camera_cmd)
    return ld
