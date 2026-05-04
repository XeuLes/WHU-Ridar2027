import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _referee_debug_launch_common import MAP_REQUEST_YAML, launch_tx, topic_pub


def generate_launch_description():
    return launch_tx(
        topic_pub(
            "/radar/debug/tx/map_robot_data",
            "radar_decision_comm/msg/MapPublishRequest",
            MAP_REQUEST_YAML,
        )
    )
