from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    topic = LaunchConfiguration("topic")
    output_file = LaunchConfiguration("output_file")
    fourcc = LaunchConfiguration("fourcc")
    fps = LaunchConfiguration("fps")
    save_every_n = LaunchConfiguration("save_every_n")
    max_frames = LaunchConfiguration("max_frames")
    use_timestamp_name = LaunchConfiguration("use_timestamp_name")
    resize_to_first_frame = LaunchConfiguration("resize_to_first_frame")
    log_level = LaunchConfiguration("log_level")

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    video_saver_node = Node(
        name="video_saver_node",
        package="hik_camera_ros2_driver",
        executable="video_saver_node",
        parameters=[
            {
                "topic": topic,
                "output_file": output_file,
                "fourcc": fourcc,
                "fps": ParameterValue(fps, value_type=float),
                "save_every_n": ParameterValue(save_every_n, value_type=int),
                "max_frames": ParameterValue(max_frames, value_type=int),
                "use_timestamp_name": ParameterValue(
                    use_timestamp_name, value_type=bool
                ),
                "resize_to_first_frame": ParameterValue(
                    resize_to_first_frame, value_type=bool
                ),
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
    )

    return LaunchDescription(
        [
            stdout_linebuf_envvar,
            colorized_output_envvar,
            DeclareLaunchArgument(
                "topic",
                default_value="/camera2/image",
                description="Image topic to save as video",
            ),
            DeclareLaunchArgument(
                "output_file",
                default_value="/tmp/camera2_video.avi",
                description="Output video file path",
            ),
            DeclareLaunchArgument(
                "fourcc",
                default_value="MJPG",
                description="OpenCV VideoWriter fourcc code, for example MJPG or mp4v",
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="60.0",
                description="Output video fps",
            ),
            DeclareLaunchArgument(
                "save_every_n",
                default_value="1",
                description="Save one frame every N received frames",
            ),
            DeclareLaunchArgument(
                "max_frames",
                default_value="0",
                description="Stop writing after this many frames, 0 means unlimited",
            ),
            DeclareLaunchArgument(
                "use_timestamp_name",
                default_value="false",
                description="Append timestamp before output file extension",
            ),
            DeclareLaunchArgument(
                "resize_to_first_frame",
                default_value="true",
                description="Resize changed-size frames to the first frame size",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="log level",
            ),
            video_saver_node,
        ]
    )
