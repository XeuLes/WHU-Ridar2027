from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("gimbal_serial_bridge")
    default_bridge_params = os.path.join(pkg_share, "config", "gimbal_serial_bridge.yaml")

    start_bridge = LaunchConfiguration("start_bridge")
    bridge_params_file = LaunchConfiguration("bridge_params_file")
    cmd_topic = LaunchConfiguration("cmd_topic")
    enable_topic = LaunchConfiguration("enable_topic")
    yaw_deg = LaunchConfiguration("yaw_deg")
    pitch_deg = LaunchConfiguration("pitch_deg")
    step_deg = LaunchConfiguration("step_deg")
    publish_hz = LaunchConfiguration("publish_hz")
    publish_enable = LaunchConfiguration("publish_enable")
    enable_value = LaunchConfiguration("enable_value")
    yaw_index = LaunchConfiguration("yaw_index")
    pitch_index = LaunchConfiguration("pitch_index")
    publish_dim = LaunchConfiguration("publish_dim")

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_bridge",
            default_value="true",
            description="Whether to start the gimbal_serial_bridge_node together with manual control",
        ),
        DeclareLaunchArgument(
            "bridge_params_file",
            default_value=default_bridge_params,
            description="Parameter file for gimbal_serial_bridge_node",
        ),
        DeclareLaunchArgument(
            "cmd_topic",
            default_value="/gimbal/command_angles",
            description="Command topic for accumulated absolute yaw/pitch angles",
        ),
        DeclareLaunchArgument(
            "enable_topic",
            default_value="/gimbal/enable",
            description="Enable topic for gimbal output",
        ),
        DeclareLaunchArgument(
            "yaw_deg",
            default_value="0.0",
            description="Initial absolute yaw command in degrees",
        ),
        DeclareLaunchArgument(
            "pitch_deg",
            default_value="0.0",
            description="Initial absolute pitch command in degrees",
        ),
        DeclareLaunchArgument(
            "step_deg",
            default_value="0.5",
            description="Keyboard increment step in degrees",
        ),
        DeclareLaunchArgument(
            "publish_hz",
            default_value="20.0",
            description="Periodic publish rate",
        ),
        DeclareLaunchArgument(
            "publish_enable",
            default_value="true",
            description="Whether to publish /gimbal/enable together with angle commands",
        ),
        DeclareLaunchArgument(
            "enable_value",
            default_value="true",
            description="Value published to the enable topic",
        ),
        DeclareLaunchArgument(
            "yaw_index",
            default_value="0",
            description="Yaw index in the Float64MultiArray command",
        ),
        DeclareLaunchArgument(
            "pitch_index",
            default_value="1",
            description="Pitch index in the Float64MultiArray command",
        ),
        DeclareLaunchArgument(
            "publish_dim",
            default_value="2",
            description="Float64MultiArray size",
        ),
        LogInfo(msg="Manual gimbal control: W/S pitch, A/D yaw, Q quit keyboard thread"),
        Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            condition=IfCondition(start_bridge),
            parameters=[bridge_params_file],
        ),
        Node(
            package="gimbal_serial_bridge",
            executable="manual_angle_cmd_node",
            name="manual_angle_cmd_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "cmd_topic": cmd_topic,
                "enable_topic": enable_topic,
                "yaw_deg": ParameterValue(yaw_deg, value_type=float),
                "pitch_deg": ParameterValue(pitch_deg, value_type=float),
                "step_deg": ParameterValue(step_deg, value_type=float),
                "publish_hz": ParameterValue(publish_hz, value_type=float),
                "publish_enable": ParameterValue(publish_enable, value_type=bool),
                "enable_value": ParameterValue(enable_value, value_type=bool),
                "yaw_index": ParameterValue(yaw_index, value_type=int),
                "pitch_index": ParameterValue(pitch_index, value_type=int),
                "publish_dim": ParameterValue(publish_dim, value_type=int),
            }],
        ),
    ])
