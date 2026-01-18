from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory("gimbal_serial_bridge")
    params = os.path.join(pkg, "config", "gimbal_serial_bridge.yaml")

    return LaunchDescription([
        Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            parameters=[params],
        )
    ])
