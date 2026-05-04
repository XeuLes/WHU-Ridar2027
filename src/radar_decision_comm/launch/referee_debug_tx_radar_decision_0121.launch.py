import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _referee_debug_launch_common import launch_tx, topic_pub


def generate_launch_description():
    return launch_tx(
        topic_pub(
            "/radar/debug/tx/radar_cmd",
            "radar_decision_comm/msg/RadarDecisionCommand",
            "{command_type: 1, command_value: 1, note: debug_double_effect_once}",
        )
    )
