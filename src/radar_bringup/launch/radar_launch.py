import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch import LaunchDescription
from launch.actions import (Shutdown, LogInfo, ExecuteProcess, 
                           DeclareLaunchArgument, SetEnvironmentVariable)
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取各个功能包的路径（移除world_to_camera_calib相关路径）
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros_driver2')
    dynamic_cloud_share_dir = get_package_share_directory('dynamic_cloud')
    tdt_vision_share_dir = get_package_share_directory('tdt_vision')
    debug_map_share_dir = get_package_share_directory('debug_map')
    kalman_filter_share_dir = get_package_share_directory('kalman_filter')
    hik_camera_share_dir = get_package_share_directory('hik_camera_ros2_driver')
    
    # 地图点云文件路径
    map_pcd_path = os.path.join(localization_share_dir, 'config', 'RM2025.pcd')

    # Livox驱动配置（使用功能包绝对路径）
    livox_config_path = os.path.join(livox_share_dir, 'config', 'MID360_config.json')

    # RViz配置文件路径
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'radar_debug.rviz'
    )

    # ======================== 声明启动参数 ========================
    # Hik相机参数（保留原有配置）
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

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

    # Livox雷达参数（保留原有配置）
    declare_livox_log_level_cmd = DeclareLaunchArgument(
        "livox_log_level", 
        default_value="info", 
        description="Livox lidar log level"
    )

    # ======================== 定义功能节点 ========================
    # 定位节点
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                'use_map': True,
                'map_frame': 'map'
            }],
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 动态点云处理节点
    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 聚类节点
    def get_cluster_node():
        return ComposableNode(
            package='cluster',
            plugin='tdt_radar::Cluster',
            name='cluster_node',
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 检测节点
    def get_detect_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Detect',
            name='radar_detect_node',
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 解析节点
    def get_resolve_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Resolve',
            name='radar_resolve_node',
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 卡尔曼滤波节点
    def get_kalman_filter_node():
        return ComposableNode(
            package='kalman_filter',
            plugin='tdt_radar::KalmanFilter',
            name='kalman_filter_node',
            extra_arguments=[
                {'use_intra_process_comms': True},
                {'qos_overrides./parameter_events.publisher.reliability': 'best_effort'}
            ]
        )

    # 调试地图节点
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

    # 地图点云发布节点
    def get_map_pcd_publisher():
        return Node(
            package='pcl_ros',
            executable='pcd_to_pointcloud',
            name='map_pcd_publisher',
            parameters=[{
                'file_name': map_pcd_path,
                'frame_id': 'map',
                'publish_rate': 0.1
            }],
            remappings=[('cloud', '/map_point_cloud')],
            output='screen'
        )

    # Hik相机驱动节点（保留原有配置）
    def get_hik_camera_node():
        return Node(
            name="hik_camera_ros2_driver",
            package="hik_camera_ros2_driver",
            executable="hik_camera_ros2_driver_node",
            parameters=[LaunchConfiguration("hik_params_file")],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("hik_log_level")],
            output="screen",
        )

    # Livox雷达驱动节点（保留原有配置）
    def get_livox_driver_node():
        livox_params = [
            {"xfer_format": 0},                # Pointcloud2(PointXYZRTL)
            {"multi_topic": 0},                # 所有雷达共享话题
            {"data_src": 0},                   # 数据源为雷达
            {"publish_freq": 10.0},            # 发布频率
            {"output_data_type": 0},
            {"frame_id": 'livox_frame'},       # 坐标系ID
            {"lvx_file_path": ''},             # 不使用LVX文件
            {"user_config_path": livox_config_path},  # 配置文件路径
            {"cmdline_input_bd_code": 'livox0000000001'}
        ]
        return Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_lidar_publisher',
            parameters=livox_params,
            arguments=["--ros-args", "--log-level", LaunchConfiguration("livox_log_level")],
            output='screen',
        )

    # ======================== 节点容器 ========================
    container = ComposableNodeContainer(
        name='radar_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node(),
            get_dynamic_cloud_node(),
            get_debug_map_node(),
            get_cluster_node(),
            get_detect_node(),
            get_resolve_node(),
            get_kalman_filter_node(),
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # ======================== 日志与RViz ========================
    # 路径日志（移除world_to_camera相关日志）
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")
    log_debug_map_path = LogInfo(msg=f"Debug map path: {os.path.join(debug_map_share_dir, 'config', 'RM2025.png')}")
    log_livox_config = LogInfo(msg=f"Livox config path: {livox_config_path}")
    log_hik_config = LogInfo(msg=f"Hik camera config path: {os.path.join(hik_camera_share_dir, 'config', 'camera_params.yaml')}")

    # 启动RViz2（保留原有配置）
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )

    # ======================== 组装启动描述 ========================
    return LaunchDescription([
        # 环境变量与参数声明
        stdout_linebuf_envvar,
        colorized_output_envvar,
        declare_hik_params_cmd,
        declare_hik_log_level_cmd,
        declare_livox_log_level_cmd,

        # 日志输出（移除world_to_camera相关日志）
        log_rviz_path,
        log_map_path,
        log_debug_map_path,
        log_livox_config,
        log_hik_config,

        # 核心节点（移除world_to_camera_calib_node）
        container,
        get_map_pcd_publisher(),
        get_hik_camera_node(),
        get_livox_driver_node(),
        rviz_node
    ])
