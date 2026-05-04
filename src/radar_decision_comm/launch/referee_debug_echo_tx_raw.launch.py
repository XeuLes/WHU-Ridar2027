import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch import LaunchDescription

from _referee_debug_launch_common import common_args, topic_echo


def generate_launch_description():
    return LaunchDescription(
        [
            *common_args(),
            topic_echo("/radar/debug/referee_tx_raw_bytes", "std_msgs/msg/UInt8MultiArray"),
        ]
    )
