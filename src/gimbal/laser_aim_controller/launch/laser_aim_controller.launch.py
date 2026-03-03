import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("laser_aim_controller")
    config_file = os.path.join(pkg_share, "config", "laser_aim_controller.yaml")

    return LaunchDescription([
        Node(
            package="laser_aim_controller",
            executable="laser_aim_controller_node",
            name="laser_aim_controller_node",
            output="screen",
            parameters=[config_file]
        )
    ])