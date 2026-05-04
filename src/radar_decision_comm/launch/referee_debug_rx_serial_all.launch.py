import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _referee_debug_launch_common import launch_rx, topic_echo


def generate_launch_description():
    return launch_rx(
        [
            topic_echo("/radar/referee/game_status", "radar_decision_comm/msg/RefereeGameState"),
            topic_echo("/radar/referee/enemy_hp", "radar_decision_comm/msg/RefereeEnemyHp"),
            topic_echo("/radar/referee/dart_status", "radar_decision_comm/msg/DartStatus"),
            topic_echo("/radar/referee/mark_progress", "radar_decision_comm/msg/RadarMarkProgress"),
            topic_echo("/radar/referee/radar_decision_sync", "radar_decision_comm/msg/RadarDecisionSync"),
            topic_echo("/radar/referee/robot_identity", "radar_decision_comm/msg/RefereeRobotIdentity"),
            topic_echo("/radar/debug/referee_raw_bytes", "std_msgs/msg/UInt8MultiArray"),
        ],
        mock=False,
    )
