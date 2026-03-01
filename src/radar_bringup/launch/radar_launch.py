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
    hik_camera_share_dir = get_package_share_directory('hik_camera_ros2_driver')
    gim_rough_share = get_package_share_directory('gimbal_aim_solver')
    gim_bridge_share = get_package_share_directory("gimbal_serial_bridge")
    
    




    # 海康相机参数文件路径
    hik_params_file = os.path.join(hik_camera_share_dir, 'config', 'camera_params.yaml')
    # 海康相机节点
    def get_hik_camera_node():
        return Node(
            name="hik_camera_ros2_driver",
            package="hik_camera_ros2_driver",
            executable="hik_camera_ros2_driver_node",
            parameters=[hik_params_file],
            arguments=["--ros-args", "--log-level", "info"],
            output="screen",
        )

    
    # 地图点云文件路径
    map_pcd_path = os.path.join(localization_share_dir, 'config', '基础制造.pcd')

    # Livox驱动配置
    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'MID360_config.json')


    # 云台配置
    gim_rough_cfg = os.path.join(gim_rough_share, 'config', 'gimbal_aim_solver.yaml')
    gim_bridge_params = os.path.join(gim_bridge_share, "config", "gimbal_serial_bridge.yaml")


    
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
            parameters=[{
                "image_topic": "/camera0/image",          # 你要订阅的真实话题
            }],
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
            parameters=[{
            "out_matrix_path": f"/home/user/code/T-DT_Radar-main/config/out_matrix_gc.yaml",
            "camera_params_path": f"/home/user/code/T-DT_Radar-main/config/camera_params.yaml",
            "region_points_path": f"/home/user/code/T-DT_Radar-main/config/GCZX_Points.yaml",
            "minimap_path": f"/home/user/code/T-DT_Radar-main/config/jichuzhizao.png",
            "is_test":True,
            }],
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
                'map_path': os.path.join(debug_map_share_dir, 'config', 'jichuzhizao.png')
                
            }],
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

def get_new_map_node():
    return Node(
        package='debug_map',
        executable='debug_map_homo_node',
        name='debug_map_homo_node',
        output='screen',
        parameters=[{
            'map_path': os.path.join(debug_map_share_dir, 'config', 'jichuzhizao.png'),
            'homography_path': os.path.join(debug_map_share_dir, 'config', 'debug_map_homography.yaml'),
            'auto_load_homography': True,
            'auto_save_homography': True,

            'target_w': 700,
            'target_h': 375,
            'kalman_topic': '/kalman_detect',
            'resolve_topic': '/resolve_result',
            'uav_topic': '/uav_positions',
            'match_info_topic': '/match_info',
            'debug_image_topic': '/debug_map_homo',
            'draw_timeout_ms': 500,
            'show_window': True,
            'prefer_kalman': True,
        }]
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
    
    def get_radar_calib_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='radar_calib_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    radar_calib_node = get_radar_calib_node('tdt_vision', 'tdt_radar::Calibrate')
    
    def get_gimbal_rough_node():
        return  Node(
            package='gimbal_aim_solver',
            executable='gimbal_aim_solver_node',
            name='gimbal_aim_solver_node',
            output='screen',
            parameters=[gim_rough_cfg]
        )

    def get_gimbal_bridge_node():
        return Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            parameters=[gim_bridge_params],
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
        # get_rosbag_player_node(),
        # get_radar_calib_node('tdt_vision', 'tdt_radar::Calibrate'),
        get_localization_node(),
        get_dynamic_cloud_node(),
        get_cluster_node(),
        get_detect_node(),
        get_resolve_node(),
        get_kalman_filter_node(),
        
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

    get_hik_camera_node(),     # Node ✅
    get_gimbal_rough_node(),   # Node ✅（移到外面）
    get_gimbal_bridge_node(),  # Node ✅（移到外面）
    get_new_map_node(),
    container,                 # Container ✅（只装 composable）
    mappcd_node,               # Node ✅
    rviz_node
])