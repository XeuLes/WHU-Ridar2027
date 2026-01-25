import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
# 导入QoS配置类（仅用于节点内部参考，不直接作为参数传递）
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

def generate_launch_description():
    # 定义统一的QoS配置（供节点代码内部使用的参考，不在launch参数中传递）
    unified_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    # 获取各个功能包的路径
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros2_driver')
    dynamic_cloud_share_dir = get_package_share_directory('dynamic_cloud')
    tdt_vision_share_dir = get_package_share_directory('tdt_vision')
    debug_map_share_dir = get_package_share_directory('debug_map')
    kalman_filter_share_dir = get_package_share_directory('kalman_filter')

    
    # 地图点云文件路径
    map_pcd_path = os.path.join(localization_share_dir, 'config', 'RM2025.pcd')

    # Livox驱动配置
    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'MID360_config.json')


    
    # RViz配置文件路径
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'radar_debug.rviz'
    )
    
    # 定义各个节点（移除参数中的qos_profile，QoS在节点代码内配置）
    
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                'use_map': True,
                'map_frame': 'rm_frame',
                'target_pcd_path': map_pcd_path
            }],
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',

            parameters=[{
                'map_pcd_path': map_pcd_path
            }],

            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_cluster_node():
        return ComposableNode(
            package='cluster',
            plugin='tdt_radar::Cluster',
            name='cluster_node',
            # 移除qos_profile参数
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_detect_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Detect',
            name='radar_detect_node',
            # 移除qos_profile参数
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_resolve_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Resolve',
            name='radar_resolve_node',
            # 移除qos_profile参数
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_kalman_filter_node():
        return ComposableNode(
            package='kalman_filter',
            plugin='tdt_radar::KalmanFilter',
            name='kalman_filter_node',
            # 移除qos_profile参数
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    def get_debug_map_node():
        return ComposableNode(
            package='debug_map',
            plugin='debug_map::DebugMap',
            name='debug_map_node',
            parameters=[{
                'map_path': os.path.join(debug_map_share_dir, 'config', 'RM2025.png')
                
            }],
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
    
    def get_rosbag_player_node():
        return ComposableNode(
            package='rosbag_player',
            plugin='RosbagPlayer',
            name='rosbag_player_node',
            parameters=[{
                'rosbag_file': "/home/user/DATA/rosbag.db3"
                # 移除qos_profile参数
            }],
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )
        
    # 地图点云发布节点
    def get_map_pcd_publisher(): 
        return Node(  # ✅ 改为独立节点
            package='pcl_ros',
            executable='pcd_to_pointcloud',  # 独立节点需要executable参数
            name='map_pcd_publisher',
            parameters=[{
                'file_name': map_pcd_path,  
                'frame_id': 'map',
                'publish_rate': 0.1
            }],
            remappings=[('cloud', '/map_point_cloud')],
            output='screen'  # 可选：输出日志到终端
        )

    

    # 创建节点容器
    container = ComposableNodeContainer(
        name='radar_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_rosbag_player_node(),
            get_localization_node(),
            get_dynamic_cloud_node(),
            get_debug_map_node(),
            get_cluster_node(),
            get_detect_node(),
            get_resolve_node(),
            get_kalman_filter_node(),
            # get_radar_calib_node('tdt_vision', 'tdt_radar::Calibrate')
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # 打印路径日志
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")
    log_debug_map_path = LogInfo(msg=f"Debug map path: {os.path.join(debug_map_share_dir, 'config', 'RM2025.png')}")

    # 启动RViz2
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )
    
    mappcd_node = get_map_pcd_publisher()


    return LaunchDescription([
        log_rviz_path,
        log_map_path,
        log_debug_map_path,

        container,
        mappcd_node,

        rviz_node
    ])