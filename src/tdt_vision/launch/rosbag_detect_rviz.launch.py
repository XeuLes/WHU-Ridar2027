import os
from launch import LaunchDescription
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch.actions import Shutdown, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包共享目录
    tdt_vision_share = get_package_share_directory("tdt_vision")
    hik_camera_share_dir = get_package_share_directory('hik_camera_ros2_driver')
    
    # ============================================================
    # 环境变量设置（用于日志输出优化）
    # ============================================================
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")
    
    # ============================================================
    # 启动参数声明
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
    
    # 定义rosbag播放器节点
    def get_rosbag_player_node():
        return ComposableNode(
            package="rosbag_player",
            plugin="RosbagPlayer",
            name="rosbag_player_node",
            parameters=[{
                "rosbag_file": "/home/user/DATA/rosbag.db3"
            }],
            extra_arguments=[{"use_intra_process_comms": True}]
        )
    
    # 定义检测节点
    def get_detect_node():
        return ComposableNode(
            package="tdt_vision",
            plugin="tdt_radar::Detect",
            name="radar_detect_node",
            extra_arguments=[{"use_intra_process_comms": True}]
        )
    
    # 定义组件容器
    def get_component_container():
        return ComposableNodeContainer(
            name="detect_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                # get_rosbag_player_node(),  # 已注释 - 使用真实相机
                get_detect_node()
            ],
            output="both",
            emulate_tty=True,
            on_exit=Shutdown()
        )
    
    # 定义海康相机驱动节点
    def get_hik_camera_node():
        return Node(
            name="hik_camera_ros2_driver",
            package="hik_camera_ros2_driver",
            executable="hik_camera_ros2_driver_node",
            parameters=[LaunchConfiguration("hik_params_file")],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("hik_log_level")],
            output="screen",
        )
    
    # 定义RViz节点
    def get_rviz_node():
        rviz_config_path = os.path.join(
            tdt_vision_share,
            "config",  # 假设配置文件放在tdt_vision包的config目录下
            "default.rviz"  # 替换为实际的RViz配置文件名
        )
        return Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_path],  # 指定RViz配置文件
            output="screen"
        )
    
    # 组装启动描述
    return LaunchDescription([
        # 环境变量
        stdout_linebuf_envvar,
        colorized_output_envvar,
        # 参数声明
        declare_hik_params_cmd,
        declare_hik_log_level_cmd,
        # 核心节点
        get_hik_camera_node(),  # 海康相机节点
        get_component_container(),
        get_rviz_node()
    ])
