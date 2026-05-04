import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events.process import ShutdownProcess
from launch.substitutions import LaunchConfiguration


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _build_launch(context):
    output_dir = os.path.expandvars(
        os.path.expanduser(LaunchConfiguration("output_dir").perform(context))
    )
    record_duration_sec = float(
        LaunchConfiguration("record_duration_sec").perform(context)
    )
    bag_prefix = LaunchConfiguration("bag_prefix").perform(context).strip()
    storage = LaunchConfiguration("storage").perform(context).strip()
    max_cache_size = LaunchConfiguration("max_cache_size").perform(context).strip()
    compression_mode = LaunchConfiguration("compression_mode").perform(context).strip()
    compression_format = LaunchConfiguration("compression_format").perform(context).strip()
    include_hidden_topics = _as_bool(
        LaunchConfiguration("include_hidden_topics").perform(context)
    )
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))

    if not output_dir:
        raise RuntimeError("output_dir must not be empty")
    if not bag_prefix:
        raise RuntimeError("bag_prefix must not be empty")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join(output_dir, f"{bag_prefix}_{timestamp}")

    command = [
        "ros2",
        "bag",
        "record",
        "-a",
        "-o",
        bag_path,
        "--storage",
        storage,
        "--max-cache-size",
        max_cache_size,
    ]

    if include_hidden_topics:
        command.append("--include-hidden-topics")
    if use_sim_time:
        command.append("--use-sim-time")
    if compression_mode != "none":
        command.extend([
            "--compression-mode",
            compression_mode,
            "--compression-format",
            compression_format,
        ])

    recorder_name = "radar_rosbag_recorder"
    recorder_process = ExecuteProcess(
        cmd=command,
        name=recorder_name,
        output="screen",
        emulate_tty=True,
        sigterm_timeout="15",
        sigkill_timeout="5",
    )

    def matches_recorder(action):
        return action is recorder_process

    actions = [
        LogInfo(msg=f"recording all ROS topics to: {bag_path}"),
        LogInfo(msg=f"record duration: {record_duration_sec:.1f}s"),
        recorder_process,
        RegisterEventHandler(
            OnProcessExit(
                target_action=recorder_process,
                on_exit=[LogInfo(msg="rosbag recorder exited")],
            )
        ),
    ]

    if record_duration_sec > 0.0:
        actions.append(
            TimerAction(
                period=record_duration_sec,
                actions=[
                    LogInfo(msg="record duration reached; stopping rosbag recorder"),
                    EmitEvent(
                        event=ShutdownProcess(process_matcher=matches_recorder)
                    ),
                ],
            )
        )
    else:
        actions.append(
            LogInfo(msg="record_duration_sec <= 0, recording until launch shutdown")
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_dir",
                default_value="/home/user/DATA/RADAR_ROSBAG",
                description="base directory for rosbag recordings",
            ),
            DeclareLaunchArgument(
                "record_duration_sec",
                default_value="480.0",
                description="recording duration in seconds; <=0 means until Ctrl-C",
            ),
            DeclareLaunchArgument(
                "bag_prefix",
                default_value="radar_match",
                description="prefix for timestamped bag directory names",
            ),
            DeclareLaunchArgument(
                "storage",
                default_value="sqlite3",
                description="rosbag2 storage plugin",
            ),
            DeclareLaunchArgument(
                "max_cache_size",
                default_value="104857600",
                description="rosbag2 max cache size in bytes",
            ),
            DeclareLaunchArgument(
                "compression_mode",
                default_value="none",
                description="rosbag2 compression mode: none/file/message",
            ),
            DeclareLaunchArgument(
                "compression_format",
                default_value="zstd",
                description="rosbag2 compression format used when compression is enabled",
            ),
            DeclareLaunchArgument(
                "include_hidden_topics",
                default_value="false",
                description="whether to include hidden ROS topics",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="record with ROS simulation time",
            ),
            OpaqueFunction(function=_build_launch),
        ]
    )
