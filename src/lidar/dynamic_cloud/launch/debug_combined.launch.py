import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取localization包的路径（参考localization的launch文件风格）
    localization_share_dir = get_package_share_directory('localization')
    # 地图点云文件路径（通常放在localization包的data目录下）
    map_pcd_path = os.path.join(localization_share_dir, 'config', 'RM2025.pcd')

    # 定义各个节点
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                # 可添加localization节点所需参数
                'use_map': True,
                'map_frame': 'map'
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
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

    # 地图点云发布节点（参考localization包的pcd发布方式）
    def get_map_pcd_publisher():
        return ComposableNode(
            package='pcl_ros',
            plugin='pcl_ros::PCDToPointCloud',
            name='map_pcd_publisher',  # 符合localization的命名风格
            parameters=[{
                'pcd_file': map_pcd_path,  # 使用包内路径
                'frame_id': 'map',         # 与定位模块坐标系保持一致
                'publish_rate': 0.1,       # 地图点云低频率发布
                'repeat_delay': -1         # 只发布一次（参考地图发布习惯）
            }],
            remappings=[
                ('cloud', '/map_point_cloud')  # 明确的地图点云话题
            ],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # 创建节点容器
    container = ComposableNodeContainer(
        name='combined_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node(),
            get_dynamic_cloud_node(),
            get_rosbag_player_node(),
            get_map_pcd_publisher()  # 添加地图点云发布节点
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # 配置RViz2（使用localization或dynamic_cloud的配置）
    rviz_config_package = 'localization'  # 可改为使用localization的rviz配置
    rviz_package_share_dir = get_package_share_directory(rviz_config_package)
    rviz_config_path = os.path.join(rviz_package_share_dir, 'rviz', 'map_visualization.rviz')

    # 打印配置路径日志
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")

    # 启动RViz2
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )

    return LaunchDescription([
        log_rviz_path,
        log_map_path,
        container,
        rviz_node
    ])
