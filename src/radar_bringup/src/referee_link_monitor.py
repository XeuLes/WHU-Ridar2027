#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from radar_decision_comm.msg import RefereeGameState


RED_BOLD = "\033[1;31m"
RESET = "\033[0m"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class RefereeLinkMonitor(Node):
    def __init__(self):
        super().__init__("referee_link_monitor")

        self.enabled = _as_bool(self.declare_parameter("enabled", True).value)
        self.game_status_topic = str(self.declare_parameter(
            "game_status_topic", "/radar/referee/game_status"
        ).value)
        self.warn_after_sec = float(
            self.declare_parameter("warn_after_sec", 3.0).value
        )

        self._received_game_status = False
        self._warned = False

        self._game_sub = self.create_subscription(
            RefereeGameState,
            self.game_status_topic,
            self._on_game_status,
            10,
        )

        if not self.enabled:
            self.get_logger().info("referee link monitor disabled")
            return

        self.get_logger().info(
            "referee link monitor armed: topic=%s warn_after=%.1fs"
            % (self.game_status_topic, self.warn_after_sec)
        )
        self._timer = None
        if self.warn_after_sec > 0.0:
            self._timer = self.create_timer(self.warn_after_sec, self._on_timeout)

    def _on_game_status(self, msg):
        if not self.enabled:
            return

        first_message = not self._received_game_status
        self._received_game_status = True

        if first_message:
            self.get_logger().info(
                "referee game_status received: game_type=%d game_progress=%d "
                "stage_remain_time=%d"
                % (msg.game_type, msg.game_progress, msg.stage_remain_time)
            )
            if self._warned:
                self.get_logger().info(
                    "referee communication recovered: game_status is now received"
                )

        if hasattr(self, "_timer") and self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None

    def _on_timeout(self):
        if not self.enabled or self._received_game_status or self._warned:
            return

        self._warned = True
        self.get_logger().error(
            "%sREFEREE COMMUNICATION WARNING: no game_status message received "
            "from %s within %.1f seconds. Check referee serial link, rx_mode, "
            "device path, baudrate, and referee system power.%s"
            % (RED_BOLD, self.game_status_topic, self.warn_after_sec, RESET)
        )

        if hasattr(self, "_timer") and self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None


def main(args=None):
    rclpy.init(args=args)
    node = RefereeLinkMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
