import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def common_args():
    return [
        DeclareLaunchArgument("device", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baudrate", default_value="115200"),
        DeclareLaunchArgument("my_color", default_value="Blue"),
        DeclareLaunchArgument("tx_mode", default_value="serial"),
        DeclareLaunchArgument("rx_mode", default_value="serial"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("times", default_value="1"),
        DeclareLaunchArgument("raw_tx_topic", default_value="/radar/debug/referee_tx_raw_bytes"),
        DeclareLaunchArgument("raw_rx_topic", default_value="/radar/debug/referee_raw_bytes"),
    ]


def tx_container():
    return ComposableNodeContainer(
        name="referee_debug_tx_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::RefereeSerialTxComponent",
                name="referee_debug_tx",
                parameters=[
                    {
                        "enabled": True,
                        "tx_mode": LaunchConfiguration("tx_mode"),
                        "device": LaunchConfiguration("device"),
                        "baudrate": LaunchConfiguration("baudrate"),
                        "my_color": LaunchConfiguration("my_color"),
                        "use_referee_identity_color": False,
                        "map_0305_layout": "rm2026_48",
                        "publish_debug_frames": True,
                        "debug_raw_frame_topic": LaunchConfiguration("raw_tx_topic"),
                        "input_map_request_topic": "/radar/debug/tx/map_robot_data",
                        "input_robot_interaction_topic": "/radar/debug/tx/robot_interaction",
                        "input_radar_command_topic": "/radar/debug/tx/radar_cmd",
                    }
                ],
            )
        ],
    )


def rx_bridge_container(mock=False):
    rx_mode = "mock" if mock else LaunchConfiguration("rx_mode")
    return ComposableNodeContainer(
        name="referee_debug_rx_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::RefereeSerialBridgeComponent",
                name="referee_debug_rx_bridge",
                parameters=[
                    {
                        "enabled": True,
                        "rx_mode": rx_mode,
                        "tx_mode": "serial",
                        "device": LaunchConfiguration("device"),
                        "baudrate": LaunchConfiguration("baudrate"),
                        "my_color": LaunchConfiguration("my_color"),
                        "publish_raw_debug": True,
                        "publish_debug_frames": True,
                        "debug_mock_publish": True,
                        "debug_mock_rate_hz": LaunchConfiguration("rate"),
                        "legacy_mark_progress": False,
                        "legacy_dart_bits": False,
                        "map_0305_layout": "rm2026_48",
                        "use_referee_identity_color": False,
                        "debug_raw_bytes_topic": LaunchConfiguration("raw_rx_topic"),
                        "debug_raw_frame_topic": LaunchConfiguration("raw_tx_topic"),
                    }
                ],
            )
        ],
    )


def topic_pub(topic, msg_type, yaml_text):
    return ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "--rate",
            LaunchConfiguration("rate"),
            "--times",
            LaunchConfiguration("times"),
            topic,
            msg_type,
            yaml_text,
        ],
        output="screen",
    )


def topic_echo(topic, msg_type=None):
    cmd = ["ros2", "topic", "echo", topic]
    if msg_type:
        cmd.append(msg_type)
    return ExecuteProcess(cmd=cmd, output="screen")


def launch_tx(pub_action):
    return LaunchDescription([*common_args(), tx_container(), pub_action])


def launch_rx(echo_actions, mock=False):
    return LaunchDescription([*common_args(), rx_bridge_container(mock=mock), *echo_actions])


def append_float_le(out, value):
    import struct

    out.extend(struct.pack("<f", float(value)))


def u16_bytes(value):
    value = int(value) & 0xFFFF
    return [value & 0xFF, (value >> 8) & 0xFF]


def i16_bytes(value):
    value = int(value)
    if value < 0:
        value = (1 << 16) + value
    return u16_bytes(value)


def sentry_alert_payload():
    out = []
    out.extend(u16_bytes(1))
    append_float_le(out, 12.4)
    out.extend(u16_bytes(2))
    return list(out)


def sentry_field_payload():
    slots = [
        (1, 24.5, 3.3),
        (1, 21.8, 7.1),
        (1, 18.4, 5.2),
        (1, 15.4, 9.5),
        (1, 12.6, 6.9),
        (1, 26.5, 11.7),
    ]
    out = []
    for valid, x, y in slots:
        out.append(valid)
        append_float_le(out, x)
        append_float_le(out, y)
    return list(out)


def hero_lob_payload():
    out = [1, 3]
    out.extend(i16_bytes(1250))
    out.extend(i16_bytes(3200))
    out.extend(u16_bytes(120))
    out.extend(u16_bytes(35))
    out.extend(u16_bytes(80))
    out.extend([6, 1])
    return out


MAP_REQUEST_YAML = """{
  force: true,
  robots: [
    {robot_id: 101, track_id: 101, is_valid: true, confidence: 0.95, field_position: {x: 24.5, y: 3.3, z: 0.0}},
    {robot_id: 102, track_id: 102, is_valid: true, confidence: 0.91, field_position: {x: 21.8, y: 7.1, z: 0.0}},
    {robot_id: 103, track_id: 103, is_valid: true, confidence: 0.88, field_position: {x: 18.4, y: 5.2, z: 0.0}},
    {robot_id: 104, track_id: 104, is_valid: true, confidence: 0.86, field_position: {x: 15.4, y: 9.5, z: 0.0}},
    {robot_id: 106, track_id: 106, is_valid: true, confidence: 0.82, field_position: {x: 12.6, y: 6.9, z: 0.0}},
    {robot_id: 107, track_id: 107, is_valid: true, confidence: 0.90, field_position: {x: 26.5, y: 11.7, z: 0.0}},
    {robot_id: 1, track_id: 1, is_valid: true, confidence: 0.95, field_position: {x: 2.5, y: 3.3, z: 0.0}},
    {robot_id: 2, track_id: 2, is_valid: true, confidence: 0.91, field_position: {x: 4.8, y: 7.1, z: 0.0}},
    {robot_id: 3, track_id: 3, is_valid: true, confidence: 0.88, field_position: {x: 6.4, y: 5.2, z: 0.0}},
    {robot_id: 4, track_id: 4, is_valid: true, confidence: 0.86, field_position: {x: 8.4, y: 9.5, z: 0.0}},
    {robot_id: 6, track_id: 6, is_valid: true, confidence: 0.82, field_position: {x: 10.6, y: 6.9, z: 0.0}},
    {robot_id: 7, track_id: 7, is_valid: true, confidence: 0.90, field_position: {x: 12.5, y: 11.7, z: 0.0}}
  ]
}"""


def interaction_yaml(data_cmd_id, receiver_id, user_data, sender_id=0):
    return (
        "{"
        f"data_cmd_id: {int(data_cmd_id)}, "
        f"sender_id: {int(sender_id)}, "
        f"receiver_id: {int(receiver_id)}, "
        f"user_data: {list(user_data)}"
        "}"
    )


def main_guard():
    if __name__ == "__main__":
        print("This file is a launch helper and is not meant to be executed directly.", file=sys.stderr)
        return 1
    return 0
