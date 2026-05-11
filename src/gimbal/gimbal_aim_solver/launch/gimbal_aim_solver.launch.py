from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('gimbal_aim_solver')
    localization_share = get_package_share_directory('localization')
    config_file = os.path.join(pkg_share, 'config', 'gimbal_aim_solver.yaml')
    camera_to_lidar_yaml = os.path.join(
        localization_share, 'config', 'camera_to_lidar.yaml')

    return LaunchDescription([
        Node(
            package='gimbal_aim_solver',
            executable='gimbal_aim_solver_node',
            name='gimbal_aim_solver_node',
            output='screen',
            parameters=[
                config_file,
                {'camera_to_lidar_yaml': camera_to_lidar_yaml},
            ]
        )
    ])
