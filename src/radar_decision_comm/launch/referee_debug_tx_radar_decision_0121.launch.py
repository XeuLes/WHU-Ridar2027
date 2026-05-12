import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from _referee_debug_launch_common import tx_container


def _args():
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


def _referee_status_window():
    return Node(
        package="debug_map",
        executable="referee_status_window_node",
        name="referee_status_window_node",
        output="screen",
        parameters=[
            {
                "game_status_topic": "/radar/referee/game_status",
                "identity_topic": "/radar/referee/robot_identity",
                "enemy_hp_topic": "/radar/referee/enemy_hp",
                "dart_status_topic": "/radar/referee/dart_status",
                "mark_progress_topic": "/radar/referee/mark_progress",
                "radar_sync_topic": "/radar/referee/radar_decision_sync",
                "raw_rx_topic": LaunchConfiguration("raw_rx_topic"),
                "raw_tx_topic": LaunchConfiguration("raw_tx_topic"),
                "pdu_all_topic": "/radar/all_pdu",
                "pdu_info_topic": "/radar/info_pdu",
                "pdu_jam_topic": "/radar/jam_pdu",
                "wireless_hp_topic": "/radar/wireless/enemy_hp",
                "wireless_ammo_topic": "/radar/wireless/enemy_ammo",
                "wireless_macro_topic": "/radar/wireless/enemy_macro_state",
                "wireless_buffs_topic": "/radar/wireless/enemy_buffs",
                "wireless_key_topic": "/radar/wireless/interference_key",
                "tx_map_topic": "/radar/debug/tx/map_robot_data",
                "tx_interaction_topic": "/radar/debug/tx/robot_interaction",
                "tx_radar_cmd_topic": "/radar/debug/tx/radar_cmd",
            }
        ],
    )


def _radar_0121_panel():
    return Node(
        package="radar_decision_comm",
        executable="radar_0121_debug_panel.py",
        name="radar_0121_debug_panel",
        output="screen",
        parameters=[
            {
                "output_topic": "/radar/debug/tx/radar_cmd",
                "default_radar_cmd": LaunchConfiguration("radar_cmd"),
                "default_password_cmd": LaunchConfiguration("password_cmd"),
                "default_verify_key": LaunchConfiguration("verify_key"),
            }
        ],
    )


def generate_launch_description():
    return LaunchDescription(
        [
            *_args(),
            DeclareLaunchArgument("radar_cmd", default_value="1"),
            DeclareLaunchArgument("password_cmd", default_value="2"),
            DeclareLaunchArgument("verify_key", default_value="RM2026"),
            tx_container(),
            _referee_status_window(),
            _radar_0121_panel(),
        ]
    )
