from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- Launch Args ----------
    serial_port = LaunchConfiguration("serial_port")
    baud = LaunchConfiguration("baud")

    angles_topic = LaunchConfiguration("angles_topic")
    mode_up_topic = LaunchConfiguration("mode_up_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    enable_topic = LaunchConfiguration("enable_topic")

    cmd_unit = LaunchConfiguration("cmd_unit")

    publish_hz = LaunchConfiguration("publish_hz")
    enable_hz = LaunchConfiguration("enable_hz")

    yaw_min_deg = LaunchConfiguration("yaw_min_deg")
    yaw_max_deg = LaunchConfiguration("yaw_max_deg")
    yaw_speed_dps = LaunchConfiguration("yaw_speed_dps")
    pitch_deg = LaunchConfiguration("pitch_deg")

    debug_topic = LaunchConfiguration("debug_topic")

    # 你需要替换成你工程实际的包名/可执行名
    bridge_package = LaunchConfiguration("bridge_package")
    bridge_exe = LaunchConfiguration("bridge_exe")

    test_package = LaunchConfiguration("test_package")
    test_exe = LaunchConfiguration("test_exe")

    # ---------- Declare ----------
    declare_args = [
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("baud", default_value="115200"),

        DeclareLaunchArgument("angles_topic", default_value="/gimbal/angles"),
        DeclareLaunchArgument("mode_up_topic", default_value="/gimbal/mode_up"),
        DeclareLaunchArgument("cmd_topic", default_value="/gimbal/command_angles"),
        DeclareLaunchArgument("enable_topic", default_value="/gimbal/enable"),

        DeclareLaunchArgument("cmd_unit", default_value="deg"),  # 若下位机为 rad，这里改成 rad

        DeclareLaunchArgument("publish_hz", default_value="100.0"),
        DeclareLaunchArgument("enable_hz", default_value="10.0"),

        DeclareLaunchArgument("yaw_min_deg", default_value="-90.0"),
        DeclareLaunchArgument("yaw_max_deg", default_value="90.0"),
        DeclareLaunchArgument("yaw_speed_dps", default_value="10.0"),
        DeclareLaunchArgument("pitch_deg", default_value="0.0"),

        DeclareLaunchArgument("debug_topic", default_value="/gimbal/test_rx_angles"),

        # ⚠️ 按你工程实际修改
        DeclareLaunchArgument("bridge_package", default_value="gimbal_serial_bridge"),
        DeclareLaunchArgument("bridge_exe", default_value="gimbal_serial_bridge_node"),

        DeclareLaunchArgument("test_package", default_value="gimbal_serial_bridge"),
        DeclareLaunchArgument("test_exe", default_value="xxx_test_node"),
    ]

    # ---------- Nodes ----------
    bridge_node = Node(
        package=bridge_package,
        executable=bridge_exe,
        name="gimbal_serial_bridge",
        output="screen",
        parameters=[{
            "serial_port": serial_port,
            "baud": baud,

            "angles_topic": angles_topic,
            "mode_up_topic": mode_up_topic,
            "cmd_topic": cmd_topic,
            "enable_topic": enable_topic,
        }]
    )

    test_node = Node(
        package=test_package,
        executable=test_exe,
        name="xxx_test_node",
        output="screen",
        parameters=[{
            "angles_topic": angles_topic,
            "cmd_topic": cmd_topic,
            "enable_topic": enable_topic,

            "debug_topic": debug_topic,

            "cmd_unit": cmd_unit,
            "yaw_min_deg": yaw_min_deg,
            "yaw_max_deg": yaw_max_deg,
            "yaw_speed_dps": yaw_speed_dps,
            "pitch_deg": pitch_deg,

            "publish_hz": publish_hz,
            "enable_hz": enable_hz,
        }]
    )

    return LaunchDescription(declare_args + [bridge_node, test_node])
