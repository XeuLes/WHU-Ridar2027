from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("gimbal_tf_publisher")
    params = os.path.join(pkg_share, "config", "gimbal_tf.yaml")

    return LaunchDescription([
        Node(
            package="gimbal_tf_publisher",
            executable="gimbal_tf_publisher_node",
            name="gimbal_tf_publisher",
            output="screen",
            parameters=[params],
        )
    ])
