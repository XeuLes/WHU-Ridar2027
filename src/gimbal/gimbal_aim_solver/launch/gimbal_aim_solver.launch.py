from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory("gimbal_aim_solver")
    params = os.path.join(pkg, "config", "gimbal_aim_solver.yaml")

    return LaunchDescription([
        Node(
            package="gimbal_aim_solver",
            executable="gimbal_aim_solver_node",
            name="gimbal_aim_solver",
            output="screen",
            parameters=[params],
        )
    ])
