from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    topic = LaunchConfiguration("topic")
    save_dir = LaunchConfiguration("save_dir")
    image_format = LaunchConfiguration("image_format")
    save_every_n = LaunchConfiguration("save_every_n")
    keyboard_save_enabled = LaunchConfiguration("keyboard_save_enabled")
    jpeg_quality = LaunchConfiguration("jpeg_quality")
    png_compression = LaunchConfiguration("png_compression")
    use_timestamp_name = LaunchConfiguration("use_timestamp_name")
    log_level = LaunchConfiguration("log_level")

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    image_saver_node = Node(
        name="image_saver_node",
        package="hik_camera_ros2_driver",
        executable="image_saver_node",
        parameters=[
            {
                "topic": topic,
                "save_dir": save_dir,
                "image_format": image_format,
                "save_every_n": ParameterValue(save_every_n, value_type=int),
                "keyboard_save_enabled": ParameterValue(
                    keyboard_save_enabled, value_type=bool
                ),
                "jpeg_quality": ParameterValue(jpeg_quality, value_type=int),
                "png_compression": ParameterValue(png_compression, value_type=int),
                "use_timestamp_name": ParameterValue(
                    use_timestamp_name, value_type=bool
                ),
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            stdout_linebuf_envvar,
            colorized_output_envvar,
            DeclareLaunchArgument(
                "topic",
                default_value="/camera2/image",
                description="Image topic to save",
            ),
            DeclareLaunchArgument(
                "save_dir",
                default_value="/tmp/camera2_frames",
                description="Directory for saved images",
            ),
            DeclareLaunchArgument(
                "image_format",
                default_value="jpg",
                description="Image format: jpg/jpeg/png",
            ),
            DeclareLaunchArgument(
                "save_every_n",
                default_value="0",
                description="Save one frame every N received frames, 0 disables periodic save",
            ),
            DeclareLaunchArgument(
                "keyboard_save_enabled",
                default_value="true",
                description="Press S in this terminal to save one next image frame",
            ),
            DeclareLaunchArgument(
                "jpeg_quality",
                default_value="95",
                description="JPEG quality",
            ),
            DeclareLaunchArgument(
                "png_compression",
                default_value="3",
                description="PNG compression level 0-9",
            ),
            DeclareLaunchArgument(
                "use_timestamp_name",
                default_value="false",
                description="Use ROS image timestamp in output filename",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="log level",
            ),
            image_saver_node,
        ]
    )
