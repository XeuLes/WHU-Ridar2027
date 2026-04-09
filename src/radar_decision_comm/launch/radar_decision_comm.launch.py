from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='radar_pdu_bridge',
            executable='shm_pdu_broadcast_node',
            name='shm_pdu_broadcast_node',
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('radar_decision_comm'),
                    'launch',
                    'radar_decision_comm_container.launch.py'
                ])
            )
        )
    ])
