import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_solver_params(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("gimbal_aim_solver_node", {}).get("ros__parameters", {})


def _build_node(context):
    params_file = LaunchConfiguration("params_file").perform(context)
    gimbal_serial_device = LaunchConfiguration("gimbal_serial_device").perform(context)
    solver_params = _load_solver_params(params_file)
    return [
        Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("start_bridge")),
            parameters=[
                LaunchConfiguration("bridge_params_file"),
                {
                    "serial_ports": [gimbal_serial_device],
                    "serial_port": gimbal_serial_device,
                },
            ],
        ),
        Node(
            package="gimbal_aim_solver",
            executable="gimbal_camera_translation_debug_node",
            name="gimbal_camera_translation_debug_node",
            output="screen",
            parameters=[
                solver_params,
                {
                    "cam0_image_topic": LaunchConfiguration("cam0_image_topic"),
                    "cam2_image_topic": LaunchConfiguration("cam2_image_topic"),
                    "cam0_info_topic": LaunchConfiguration("cam0_info_topic"),
                    "cmd_topic": LaunchConfiguration("cmd_topic"),
                    "enable_topic": LaunchConfiguration("enable_topic"),
                    "feedback_topic": LaunchConfiguration("feedback_topic"),
                    "default_depth_m": ParameterValue(
                        LaunchConfiguration("default_depth_m"), value_type=float),
                    "manual_step_deg": ParameterValue(
                        LaunchConfiguration("manual_step_deg"), value_type=float),
                    "max_display_width": ParameterValue(
                        LaunchConfiguration("max_display_width"), value_type=int),
                    "max_display_height": ParameterValue(
                        LaunchConfiguration("max_display_height"), value_type=int),
                },
            ],
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("gimbal_aim_solver")
    bridge_share = get_package_share_directory("gimbal_serial_bridge")
    default_params = os.path.join(pkg_share, "config", "gimbal_aim_solver.yaml")
    default_bridge_params = os.path.join(bridge_share, "config", "gimbal_serial_bridge.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("bridge_params_file", default_value=default_bridge_params),
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("gimbal_serial_device", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("cam0_image_topic", default_value="/camera0/image"),
        DeclareLaunchArgument("cam2_image_topic", default_value="/camera2/image"),
        DeclareLaunchArgument("cam0_info_topic", default_value="/camera0/camera_info"),
        DeclareLaunchArgument("cmd_topic", default_value="/gimbal/command_angles"),
        DeclareLaunchArgument("enable_topic", default_value="/gimbal/enable"),
        DeclareLaunchArgument("feedback_topic", default_value="/gimbal/angles_true"),
        DeclareLaunchArgument("default_depth_m", default_value="20.0"),
        DeclareLaunchArgument("manual_step_deg", default_value="0.05"),
        DeclareLaunchArgument("max_display_width", default_value="1800"),
        DeclareLaunchArgument("max_display_height", default_value="950"),
        OpaqueFunction(function=_build_node),
    ])
