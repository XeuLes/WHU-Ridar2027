import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import LogInfo, Shutdown

def generate_launch_description():
    # 获取功能包路径
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros_driver2')
    radar_bringup_dir = get_package_share_directory('radar_bringup')
    
    # 地图点云文件路径
    map_pcd_path = os.path.join(localization_share_dir, 'config', 'gongchuang.pcd')
    
    # Livox驱动配置
    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'MID360_config.json')

    # RViz配置文件路径
    rviz_config_path = os.path.join(radar_bringup_dir, 'config', 'radar_view.rviz')
    
    # 确保RViz配置文件目录存在
    if not os.path.exists(rviz_config_path):
        rviz_config_path = ''  # 如果配置文件不存在，使用默认配置

    # Livox驱动节点
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            "xfer_format": 0,
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "frame_id": 'livox_frame',
            "user_config_path": user_config_path,
            "cmdline_input_bd_code": 'livox0000000001'
        }]
    )

    # 定位节点
    localization_node = Node(
        package='localization',
        executable='localization_node',  # 确保存在对应的可执行文件
        name='localization_node',
        output='screen',
        parameters=[{
            'use_map': True,
            'map_frame': 'map'
        }]
    )
    
    # 动态点云处理节点
    dynamic_cloud_node = Node(
        package='dynamic_cloud',
        executable='dynamic_cloud_node',  # 确保存在对应的可执行文件
        name='dynamic_cloud_node',
        output='screen'
    )

    # 聚类节点
    cluster_node = Node(
        package='cluster',
        executable='cluster_node',  # 确保存在对应的可执行文件
        name='cluster_node',
        output='screen'
    )

    # 地图点云发布节点
    map_pcd_publisher = Node(
        package='pcl_ros',
        executable='pcd_to_pointcloud',  # 使用pcl_ros的独立可执行文件
        name='map_pcd_publisher',
        parameters=[{
            'pcd_file': map_pcd_path,
            'frame_id': 'map',
            'publish_rate': 0.1
        }],
        remappings=[('cloud', '/map_point_cloud')]
    )

    # RViz2节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path] if rviz_config_path else []
    )

    # 日志信息
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")
    log_rviz_config = LogInfo(msg=f"RViz config file path: {rviz_config_path if rviz_config_path else 'Using default config'}")

    return LaunchDescription([
        log_map_path,
        log_rviz_config,
        livox_driver,
        localization_node,
        dynamic_cloud_node,
        cluster_node,
        map_pcd_publisher,
        rviz_node
    ])
    