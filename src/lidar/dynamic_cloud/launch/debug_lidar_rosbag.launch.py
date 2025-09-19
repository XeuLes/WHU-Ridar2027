import os
import yaml
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch import LaunchDescription
from launch.actions import Shutdown, ExecuteProcess, LogInfo
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 定义节点（保持不变）
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_cluster_node():
        return ComposableNode(
            package='cluster',
            plugin='tdt_radar::Cluster',
            name='cluster_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_rosbag_player_node():
        return ComposableNode(
            package='rosbag_player',
            plugin='RosbagPlayer',
            name='rosbag_player_node',
            parameters=[{
                'rosbag_file': "/home/z/code/DataForTest/rosbag.db3"
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # 创建节点容器
    container = ComposableNodeContainer(
        name='debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node(),
            get_dynamic_cloud_node(),
            get_cluster_node(),
            get_rosbag_player_node()
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # 关键修改：根据实际路径配置RViz2
    # 1. 确定配置文件所在的功能包（dynamic_cloud）
    # 2. 构建完整路径（适配你的目录结构：lidar/dynamic_cloud/config/）
    rviz_config_package = 'dynamic_cloud'
    # 获取dynamic_cloud功能包的共享目录（编译后会对应到install下的路径）
    rviz_package_share_dir = get_package_share_directory(rviz_config_package)
    # 拼接配置文件路径（因为你的config在lidar/dynamic_cloud下，编译后会映射到对应共享目录）
    rviz_config_path = os.path.join(rviz_package_share_dir, 'config', 'config.rviz')

    # 打印路径日志（用于调试）
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")

    # 启动RViz2并加载配置
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )

    return LaunchDescription([
        log_rviz_path,
        container,
        rviz_node
    ])
    