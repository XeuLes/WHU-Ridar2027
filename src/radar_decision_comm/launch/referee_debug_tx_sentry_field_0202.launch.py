import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _referee_debug_launch_common import interaction_yaml, launch_tx, sentry_field_payload, topic_pub


def generate_launch_description():
    return launch_tx(
        topic_pub(
            "/radar/debug/tx/robot_interaction",
            "radar_decision_comm/msg/RobotInteractionRequest",
            interaction_yaml(0x0202, 107, sentry_field_payload()),
        )
    )
