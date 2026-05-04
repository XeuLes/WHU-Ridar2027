import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _referee_debug_launch_common import launch_rx, topic_echo


def generate_launch_description():
    return launch_rx(
        [topic_echo("/radar/referee/radar_decision_sync", "radar_decision_comm/msg/RadarDecisionSync")],
        mock=False,
    )
