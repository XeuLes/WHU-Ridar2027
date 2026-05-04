from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _build_launch(context):
    arguments = [
        "--input-bag-dir",
        LaunchConfiguration("input_bag_dir").perform(context),
        "--camera-topic",
        LaunchConfiguration("camera_topic").perform(context),
        "--lidar-topic",
        LaunchConfiguration("lidar_topic").perform(context),
        "--input-storage-id",
        LaunchConfiguration("input_storage_id").perform(context),
        "--output-storage-id",
        LaunchConfiguration("output_storage_id").perform(context),
        "--start-sec",
        LaunchConfiguration("start_sec").perform(context),
        "--end-sec",
        LaunchConfiguration("end_sec").perform(context),
    ]

    output_bag_dir = LaunchConfiguration("output_bag_dir").perform(context).strip()
    if output_bag_dir:
        arguments.extend(["--output-bag-dir", output_bag_dir])

    if _as_bool(LaunchConfiguration("overwrite").perform(context)):
        arguments.append("--overwrite")
    if _as_bool(LaunchConfiguration("allow_missing").perform(context)):
        arguments.append("--allow-missing")

    return [
        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "radar_bringup",
                "camera_lidar_bag_extract.py",
                *arguments,
            ],
            name="camera_lidar_bag_extract",
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_bag_dir",
                description="input rosbag2 directory, or a .db3 file inside that directory",
            ),
            DeclareLaunchArgument(
                "output_bag_dir",
                default_value="",
                description="output rosbag2 directory; empty means <input>_camera_lidar",
            ),
            DeclareLaunchArgument(
                "camera_topic",
                default_value="/camera0/image",
                description="camera topic to extract",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/livox/lidar_integrated",
                description="lidar topic to extract",
            ),
            DeclareLaunchArgument(
                "input_storage_id",
                default_value="sqlite3",
                description="input rosbag2 storage id",
            ),
            DeclareLaunchArgument(
                "output_storage_id",
                default_value="sqlite3",
                description="output rosbag2 storage id",
            ),
            DeclareLaunchArgument(
                "overwrite",
                default_value="false",
                description="replace output bag directory if it already exists",
            ),
            DeclareLaunchArgument(
                "allow_missing",
                default_value="false",
                description="write available requested topics instead of failing on missing topics",
            ),
            DeclareLaunchArgument(
                "start_sec",
                default_value="0.0",
                description="relative start time in seconds from bag beginning",
            ),
            DeclareLaunchArgument(
                "end_sec",
                default_value="-1.0",
                description="relative end time in seconds from bag beginning; <=0 means bag end",
            ),
            OpaqueFunction(function=_build_launch),
        ]
    )
