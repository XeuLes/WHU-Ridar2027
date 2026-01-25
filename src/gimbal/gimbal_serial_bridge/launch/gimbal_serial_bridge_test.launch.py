from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory("gimbal_serial_bridge")
    params = os.path.join(pkg, "config", "gimbal_serial_bridge.yaml")

    bridge = Node(
        package="gimbal_serial_bridge",
        executable="gimbal_serial_bridge_node",
        name="gimbal_serial_bridge",
        output="screen",
        parameters=[
            params,
            # 如果你的 yaml 里不是 /dev/ttyACM0，这里直接覆盖（更推荐）
            {"serial_port": "/tmp/ttyACM0"},

        ],
    )

    tester = Node(
        package="gimbal_serial_bridge",
        executable="xxx_test_node",
        name="xxx_test_node",
        output="screen",
        parameters=[{
            "angles_topic": "/gimbal/angles",
            "cmd_topic": "/gimbal/command_angles",
            "enable_topic": "/gimbal/enable",
            "debug_topic": "/gimbal/test_rx_angles",

            "cmd_unit": "deg",      # 若下位机实际用 rad，就改为 "rad"
            "yaw_min_deg": -90.0,
            "yaw_max_deg": 90.0,
            "yaw_speed_dps": 5.0,
            "pitch_deg": 0.0,

            "publish_hz": 50.0,
            "enable_hz": 5.0,
        }],
    )

    return LaunchDescription([bridge, tester])
