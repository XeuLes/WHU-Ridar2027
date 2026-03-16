from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    serial_port = LaunchConfiguration("serial_port")
    baud = LaunchConfiguration("baud")

    angles_topic = LaunchConfiguration("angles_topic")
    mode_up_topic = LaunchConfiguration("mode_up_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    enable_topic = LaunchConfiguration("enable_topic")

    default_enable = LaunchConfiguration("default_enable")
    tx_period_ms = LaunchConfiguration("tx_period_ms")

    # yaw sweep
    yaw_sweep_enable = LaunchConfiguration("yaw_sweep_enable")
    yaw_sweep_add_to_cmd = LaunchConfiguration("yaw_sweep_add_to_cmd")
    yaw_sweep_center = LaunchConfiguration("yaw_sweep_center")
    yaw_sweep_amplitude = LaunchConfiguration("yaw_sweep_amplitude")
    yaw_sweep_period = LaunchConfiguration("yaw_sweep_period")
    yaw_sweep_phase01 = LaunchConfiguration("yaw_sweep_phase01")
    yaw_sweep_wave = LaunchConfiguration("yaw_sweep_wave")
    yaw_step_dt = LaunchConfiguration("yaw_step_dt")
    yaw_step_levels = LaunchConfiguration("yaw_step_levels")

    # pitch sweep
    pitch_sweep_enable = LaunchConfiguration("pitch_sweep_enable")
    pitch_sweep_add_to_cmd = LaunchConfiguration("pitch_sweep_add_to_cmd")
    pitch_sweep_center = LaunchConfiguration("pitch_sweep_center")
    pitch_sweep_amplitude = LaunchConfiguration("pitch_sweep_amplitude")
    pitch_sweep_period = LaunchConfiguration("pitch_sweep_period")
    pitch_sweep_phase01 = LaunchConfiguration("pitch_sweep_phase01")
    pitch_sweep_wave = LaunchConfiguration("pitch_sweep_wave")
    pitch_step_dt = LaunchConfiguration("pitch_step_dt")
    pitch_step_levels = LaunchConfiguration("pitch_step_levels")

    # lissajous image
    lissajous_enable = LaunchConfiguration("lissajous_enable")
    lissajous_topic = LaunchConfiguration("lissajous_topic")
    lissajous_width = LaunchConfiguration("lissajous_width")
    lissajous_height = LaunchConfiguration("lissajous_height")
    lissajous_history = LaunchConfiguration("lissajous_history")
    lissajous_rate_hz = LaunchConfiguration("lissajous_rate_hz")
    lissajous_yaw_range = LaunchConfiguration("lissajous_yaw_range")
    lissajous_pitch_range = LaunchConfiguration("lissajous_pitch_range")
    lissajous_draw_axes = LaunchConfiguration("lissajous_draw_axes")

    view_lissajous = LaunchConfiguration("view_lissajous")

    bridge_package = LaunchConfiguration("bridge_package")
    bridge_exe = LaunchConfiguration("bridge_exe")

    declare_args = [
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("baud", default_value="115200"),

        DeclareLaunchArgument("angles_topic", default_value="/gimbal/angles"),
        DeclareLaunchArgument("mode_up_topic", default_value="/gimbal/mode_up"),
        DeclareLaunchArgument("cmd_topic", default_value="/gimbal/command_angles"),
        DeclareLaunchArgument("enable_topic", default_value="/gimbal/enable"),

        DeclareLaunchArgument("default_enable", default_value="true"),
        DeclareLaunchArgument("tx_period_ms", default_value="5.0"),

        # ========= yaw =========
        DeclareLaunchArgument("yaw_sweep_enable", default_value="true"),
        DeclareLaunchArgument("yaw_sweep_add_to_cmd", default_value="false"),
        DeclareLaunchArgument("yaw_sweep_center", default_value="0.0"),
        DeclareLaunchArgument("yaw_sweep_amplitude", default_value="30"),
        DeclareLaunchArgument("yaw_sweep_period", default_value="5.0"),
        DeclareLaunchArgument("yaw_sweep_phase01", default_value="0.0"),
        # wave: sine / triangle / step
        DeclareLaunchArgument("yaw_sweep_wave", default_value="sine"),
        # step only
        DeclareLaunchArgument("yaw_step_dt", default_value="1.0"),
        DeclareLaunchArgument("yaw_step_levels", default_value="9"),

        # ========= pitch =========
        DeclareLaunchArgument("pitch_sweep_enable", default_value="true"),
        DeclareLaunchArgument("pitch_sweep_add_to_cmd", default_value="false"),
        DeclareLaunchArgument("pitch_sweep_center", default_value="0.0"),
        DeclareLaunchArgument("pitch_sweep_amplitude", default_value="30"),
        DeclareLaunchArgument("pitch_sweep_period", default_value="5.0"),
        DeclareLaunchArgument("pitch_sweep_phase01", default_value="0.25"),
        DeclareLaunchArgument("pitch_sweep_wave", default_value="sine"),
        DeclareLaunchArgument("pitch_step_dt", default_value="1.0"),
        DeclareLaunchArgument("pitch_step_levels", default_value="9"),

        # ========= Lissajous display =========
        DeclareLaunchArgument("lissajous_enable", default_value="true"),
        DeclareLaunchArgument("lissajous_topic", default_value="/gimbal/lissajous_image"),
        DeclareLaunchArgument("lissajous_width", default_value="640"),
        DeclareLaunchArgument("lissajous_height", default_value="640"),
        DeclareLaunchArgument("lissajous_history", default_value="2000"),
        DeclareLaunchArgument("lissajous_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("lissajous_yaw_range", default_value="0.0"),
        DeclareLaunchArgument("lissajous_pitch_range", default_value="0.0"),
        DeclareLaunchArgument("lissajous_draw_axes", default_value="true"),

        DeclareLaunchArgument("view_lissajous", default_value="true"),

        # ⚠️ 按你工程实际修改
        DeclareLaunchArgument("bridge_package", default_value="gimbal_serial_bridge"),
        DeclareLaunchArgument("bridge_exe", default_value="xxx_test_node"),
    ]

    bridge_node = Node(
        package=bridge_package,
        executable=bridge_exe,
        name="gimbal_serial_bridge",
        output="screen",
        parameters=[{
            "serial_port": serial_port,
            "baud": ParameterValue(baud, value_type=int),

            "angles_topic": angles_topic,
            "mode_up_topic": mode_up_topic,
            "cmd_topic": cmd_topic,
            "enable_topic": enable_topic,

            "default_enable": ParameterValue(default_enable, value_type=bool),
            "tx_period_ms": ParameterValue(tx_period_ms, value_type=float),

            "yaw_sweep_enable": ParameterValue(yaw_sweep_enable, value_type=bool),
            "yaw_sweep_add_to_cmd": ParameterValue(yaw_sweep_add_to_cmd, value_type=bool),
            "yaw_sweep_center": ParameterValue(yaw_sweep_center, value_type=float),
            "yaw_sweep_amplitude": ParameterValue(yaw_sweep_amplitude, value_type=float),
            "yaw_sweep_period": ParameterValue(yaw_sweep_period, value_type=float),
            "yaw_sweep_phase01": ParameterValue(yaw_sweep_phase01, value_type=float),
            "yaw_sweep_wave": yaw_sweep_wave,
            "yaw_step_dt": ParameterValue(yaw_step_dt, value_type=float),
            "yaw_step_levels": ParameterValue(yaw_step_levels, value_type=int),

            "pitch_sweep_enable": ParameterValue(pitch_sweep_enable, value_type=bool),
            "pitch_sweep_add_to_cmd": ParameterValue(pitch_sweep_add_to_cmd, value_type=bool),
            "pitch_sweep_center": ParameterValue(pitch_sweep_center, value_type=float),
            "pitch_sweep_amplitude": ParameterValue(pitch_sweep_amplitude, value_type=float),
            "pitch_sweep_period": ParameterValue(pitch_sweep_period, value_type=float),
            "pitch_sweep_phase01": ParameterValue(pitch_sweep_phase01, value_type=float),
            "pitch_sweep_wave": pitch_sweep_wave,
            "pitch_step_dt": ParameterValue(pitch_step_dt, value_type=float),
            "pitch_step_levels": ParameterValue(pitch_step_levels, value_type=int),

            "lissajous_enable": ParameterValue(lissajous_enable, value_type=bool),
            "lissajous_topic": lissajous_topic,
            "lissajous_width": ParameterValue(lissajous_width, value_type=int),
            "lissajous_height": ParameterValue(lissajous_height, value_type=int),
            "lissajous_history": ParameterValue(lissajous_history, value_type=int),
            "lissajous_rate_hz": ParameterValue(lissajous_rate_hz, value_type=float),
            "lissajous_yaw_range": ParameterValue(lissajous_yaw_range, value_type=float),
            "lissajous_pitch_range": ParameterValue(lissajous_pitch_range, value_type=float),
            "lissajous_draw_axes": ParameterValue(lissajous_draw_axes, value_type=bool),
        }]
    )

    image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="lissajous_view",
        output="screen",
        arguments=[lissajous_topic],
        condition=IfCondition(view_lissajous),
    )

    return LaunchDescription(declare_args + [bridge_node, image_view])
