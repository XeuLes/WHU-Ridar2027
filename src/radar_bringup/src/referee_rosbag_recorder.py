#!/usr/bin/env python3

import os
import shutil
import signal
import subprocess
import time
from datetime import datetime

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


class RefereeRosbagRecorder(Node):
    def __init__(self):
        super().__init__("referee_rosbag_recorder")

        self.enabled = _as_bool(self.declare_parameter("enabled", True).value)
        self.output_dir = str(self.declare_parameter(
            "output_dir", "/home/user/DATA/RADAR_ROSBAG"
        ).value)
        self.record_duration_sec = float(
            self.declare_parameter("record_duration_sec", 420.0).value
        )
        self.game_status_topic = str(self.declare_parameter(
            "game_status_topic", "/radar/referee/game_status"
        ).value)
        self.start_game_progress = int(
            self.declare_parameter("start_game_progress", 4).value
        )
        self.auto_start_after_sec = float(
            self.declare_parameter("auto_start_after_sec", 60.0).value
        )
        self.record_topics = [
            str(topic)
            for topic in self.declare_parameter(
                "record_topics",
                [
                    "/camera0/image",
                    "/camera0/camera_info",
                    "/livox/lidar",
                    "/livox/imu",
                ],
            ).value
        ]

        self._record_process = None
        self._record_started = False
        self._stop_time = None
        self._armed_time = time.monotonic()

        self._game_sub = self.create_subscription(
            RefereeGameState,
            self.game_status_topic,
            self._on_game_status,
            10,
        )
        self._timer = self.create_timer(1.0, self._on_timer)

        if not self.enabled:
            self.get_logger().info("referee rosbag recorder disabled")
            return

        self.get_logger().info(
            "referee rosbag recorder armed: topic=%s start_game_progress=%d "
            "auto_start_after=%.1fs duration=%.1fs output_dir=%s record_topics=%s"
            % (
                self.game_status_topic,
                self.start_game_progress,
                self.auto_start_after_sec,
                self.record_duration_sec,
                self.output_dir,
                ", ".join(self.record_topics),
            )
        )

    def _on_game_status(self, msg):
        if not self.enabled or self._record_started:
            return

        if msg.game_progress == self.start_game_progress:
            self._start_recording(
                "referee game_progress reached %d" % self.start_game_progress
            )

    def _start_recording(self, reason):
        ros2_executable = shutil.which("ros2")
        if ros2_executable is None:
            self.get_logger().error(
                "%scannot start rosbag recording: ros2 executable not found%s"
                % (RED_BOLD, RESET)
            )
            self._record_started = True
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as exc:
            self.get_logger().error(
                "%scannot create rosbag output dir %s: %s%s"
                % (RED_BOLD, self.output_dir, exc, RESET)
            )
            self._record_started = True
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = os.path.join(self.output_dir, "radar_match_%s" % timestamp)
        if not self.record_topics:
            self.get_logger().error(
                "%scannot start rosbag recording: record_topics is empty%s"
                % (RED_BOLD, RESET)
            )
            self._record_started = True
            return

        command = [ros2_executable, "bag", "record", "-o", bag_path, *self.record_topics]

        try:
            self._record_process = subprocess.Popen(
                command,
                preexec_fn=os.setsid,
            )
        except OSError as exc:
            self.get_logger().error(
                "%sfailed to start rosbag recording: %s%s"
                % (RED_BOLD, exc, RESET)
            )
            self._record_started = True
            return

        self._record_started = True
        self._stop_time = time.monotonic() + self.record_duration_sec
        self.get_logger().info(
            "rosbag recording started by %s: %s, will stop after %.1f seconds"
            % (reason, bag_path, self.record_duration_sec)
        )

    def _on_timer(self):
        if (
            self.enabled
            and not self._record_started
            and self.auto_start_after_sec > 0.0
            and time.monotonic() - self._armed_time >= self.auto_start_after_sec
        ):
            self._start_recording(
                "auto-start timeout %.1fs without referee trigger"
                % self.auto_start_after_sec
            )
            return

        if self._record_process is None:
            return

        return_code = self._record_process.poll()
        if return_code is not None:
            self.get_logger().info("rosbag recording process exited with code %d" % return_code)
            self._record_process = None
            self._stop_time = None
            return

        if self._stop_time is not None and time.monotonic() >= self._stop_time:
            self._stop_recording("configured duration reached")

    def _stop_recording(self, reason):
        if self._record_process is None:
            return

        self.get_logger().info("stopping rosbag recording: %s" % reason)
        process = self._record_process
        self._record_process = None
        self._stop_time = None

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except ProcessLookupError:
            return
        except OSError as exc:
            self.get_logger().warning(
                "%sfailed to send SIGINT to rosbag recorder: %s%s"
                % (RED_BOLD, exc, RESET)
            )
            return

        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warning("rosbag recorder did not stop after SIGINT; terminating")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except OSError as exc:
                self.get_logger().warning(
                    "%sfailed to terminate rosbag recorder: %s%s"
                    % (RED_BOLD, exc, RESET)
                )

    def destroy_node(self):
        self._stop_recording("node shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RefereeRosbagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
