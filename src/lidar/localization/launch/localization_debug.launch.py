import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess  # 添加了ExecuteProcess的导入
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取功能包路径
    localization_package = 'localization'
    localization_share_dir = get_package_share_directory(localization_package)
    
    # 配置RViz2路径（假设配置文件放在localization/config目录下）
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'localization_rviz_config.rviz'
    )
    
    # 创建Localization节点
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                # 可在这里覆盖PCD地图路径参数
                # "target_pcd_file": os.path.join(localization_share_dir, "config", "RM2025.pcd")
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    # 创建Rosbag播放节点
    def get_rosbag_player_node():
        return ComposableNode(
            package='rosbag_player',
            plugin='RosbagPlayer',
            name='rosbag_player_node',
            parameters=[{
                'rosbag_file': "/home/z/code/DataForTest/rosbag.db3"  # 替换为实际rosbag路径
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    # 创建节点容器
    container = ComposableNodeContainer(
        name='localization_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node(),
            get_rosbag_player_node()
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )
    
    # 打印RViz配置路径日志
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    
    # 启动RViz2
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
