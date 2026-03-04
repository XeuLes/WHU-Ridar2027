import os

from launch import LaunchDescription
from launch.actions import Shutdown, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ============================================================
    # 获取包路径
    # ============================================================
    tdt_vision_share_dir = get_package_share_directory("tdt_vision")
    hik_camera_share_dir = get_package_share_directory("hik_camera_ros2_driver")

    # ============================================================
    # 环境变量
    # ============================================================
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable(
        "RCUTILS_COLORIZED_OUTPUT", "1"
    )

    # ============================================================
    # 启动参数
    # ============================================================
    declare_hik_params_cmd = DeclareLaunchArgument(
        "hik_params_file",
        default_value=os.path.join(hik_camera_share_dir, "config", "camera_params.yaml"),
        description="Hik camera configuration file path"
    )

    declare_hik_log_level_cmd = DeclareLaunchArgument(
        "hik_log_level",
        default_value="info",
        description="Hik camera log level"
    )

    declare_image_topic_cmd = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera0/image",
        description="Image topic for detect node"
    )

    declare_rviz_config_cmd = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(tdt_vision_share_dir, "config", "default.rviz"),
        description="RViz config file path"
    )

    # ============================================================
    # 海康相机节点
    # ============================================================
    hik_camera_node = Node(
        name="hik_camera_ros2_driver",
        package="hik_camera_ros2_driver",
        executable="hik_camera_ros2_driver_node",
        parameters=[LaunchConfiguration("hik_params_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("hik_log_level")],
        output="screen",
    )

    # ============================================================
    # Detect 组件节点
    # ============================================================
    detect_node = ComposableNode(
        package="tdt_vision",
        plugin="tdt_radar::Detect",
        name="radar_detect_node",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
        }],
        extra_arguments=[{"use_intra_process_comms": True}]
    )

    # ============================================================
    # 组件容器
    # ============================================================
    detect_container = ComposableNodeContainer(
        name="detect_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            detect_node
        ],
        output="both",
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # ============================================================
    # RViz
    # ============================================================
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen"
    )

    # ============================================================
    # LaunchDescription
    # ============================================================
    return LaunchDescription([
        stdout_linebuf_envvar,
        colorized_output_envvar,

        declare_hik_params_cmd,
        declare_hik_log_level_cmd,
        declare_image_topic_cmd,
        declare_rviz_config_cmd,

        hik_camera_node,
        detect_container,
        rviz_node,
    ])