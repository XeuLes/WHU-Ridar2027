import os
from launch import LaunchDescription
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch.actions import Shutdown
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包共享目录
    tdt_vision_share = get_package_share_directory("tdt_vision")
    
    # 定义rosbag播放器节点
    def get_rosbag_player_node():
        return ComposableNode(
            package="rosbag_player",
            plugin="RosbagPlayer",
            name="rosbag_player_node",
            parameters=[{
                "rosbag_file": "/home/z/code/DataForTest/rosbag.db3"  # 保持与现有路径一致
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
    
    # 定义组件容器（包含rosbag和detect节点）
    def get_component_container():
        return ComposableNodeContainer(
            name="detect_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                get_rosbag_player_node(),
                get_detect_node()
            ],
            output="both",
            emulate_tty=True,
            on_exit=Shutdown()
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
        get_component_container(),
        get_rviz_node()
    ])
