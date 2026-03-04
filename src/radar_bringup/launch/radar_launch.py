import os

from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

# 仅供参考，不直接用于 launch 参数
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def generate_launch_description():
    # 定义统一的QoS配置（仅供节点内部参考，不在launch参数中传递）
    unified_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    # 获取各个功能包路径
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros2_driver')
    dynamic_cloud_share_dir = get_package_share_directory('dynamic_cloud')
    tdt_vision_share_dir = get_package_share_directory('tdt_vision')
    debug_map_share_dir = get_package_share_directory('debug_map')
    kalman_filter_share_dir = get_package_share_directory('kalman_filter')
    hik_camera_share_dir = get_package_share_directory('hik_camera_ros2_driver')
    gim_rough_share = get_package_share_directory('gimbal_aim_solver')
    gim_bridge_share = get_package_share_directory("gimbal_serial_bridge")
    auto_aim_share = get_package_share_directory("auto_aim_fast")
    laser_aim_share = get_package_share_directory("laser_aim_controller")

    # -----------------------------
    # 文件路径
    # -----------------------------
    # 海康相机参数
    hik_params_file = os.path.join(hik_camera_share_dir, 'config', 'camera_params.yaml')

    # 地图点云
    map_pcd_path = os.path.join(localization_share_dir, 'config', '基础制造_rm_frame.pcd')

    # localization 在线初值所需配置
    camera_extrinsic_yaml = "/home/user/code/T-DT_Radar-main/config/out_matrix_gc.yaml"
    camera_to_lidar_yaml = os.path.join(localization_share_dir, 'config', 'camera_to_lidar.yaml')

    # Livox驱动配置
    cur_config_path = os.path.join(livox_share_dir, 'config')
    livox_rviz_config_path = os.path.join(cur_config_path, 'livox_lidar.rviz')
    user_config_path = os.path.join(cur_config_path, 'livox_lidar_config.json')

    xfer_format = 0
    multi_topic = 0
    data_src = 0
    publish_freq = 10.0
    output_type = 0
    livox_frame_id = 'livox_frame'
    lvx_file_path = '/home/livox/livox_test.lvx'
    cmdline_bd_code = 'livox0000000001'

    livox_ros2_params = [
        {"xfer_format": xfer_format},
        {"multi_topic": multi_topic},
        {"data_src": data_src},
        {"publish_freq": publish_freq},
        {"output_data_type": output_type},
        {"frame_id": livox_frame_id},
        {"lvx_file_path": lvx_file_path},
        {"user_config_path": user_config_path},
        {"cmdline_input_bd_code": cmdline_bd_code}
    ]

    # 云台配置
    gim_rough_cfg = os.path.join(gim_rough_share, 'config', 'gimbal_aim_solver.yaml')
    gim_bridge_params = os.path.join(gim_bridge_share, "config", "gimbal_serial_bridge.yaml")
    laser_aim_cfg = os.path.join(laser_aim_share, "config", "laser_aim_controller.yaml")

    # auto_aim_fast 配置
    auto_aim_tracker_cfg = os.path.join(auto_aim_share, "config", "tracker.yaml")
    auto_aim_camera_cfg = os.path.join(auto_aim_share, "config", "camera.yaml")
    auto_aim_detector_cfg = os.path.join(auto_aim_share, "config", "detector.yaml")
    auto_aim_qos_cfg = os.path.join(auto_aim_share, "config", "qos.yaml")
    auto_aim_info_cfg = os.path.join(auto_aim_share, "config", "camera_info.yaml")

    # RViz 配置
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'radar_debug.rviz'
    )

    # -----------------------------
    # 普通 Node
    # -----------------------------
    def get_hik_camera_node():
        return Node(
            name="hik_camera_ros2_driver",
            package="hik_camera_ros2_driver",
            executable="hik_camera_ros2_driver_node",
            parameters=[hik_params_file],
            arguments=["--ros-args", "--log-level", "info"],
            output="screen",
        )

    def get_auto_aim_node():
        return Node(
            package="auto_aim_fast",
            executable="auto_aim_node",
            name="auto_aim_node",
            output="screen",
            parameters=[
                auto_aim_tracker_cfg,
                auto_aim_camera_cfg,
                auto_aim_detector_cfg,
                auto_aim_qos_cfg,
                auto_aim_info_cfg,
            ],
            arguments=["--ros-args", "--log-level", "info"],
        )

    def get_gimbal_rough_node():
        return Node(
            package='gimbal_aim_solver',
            executable='gimbal_aim_solver_node',
            name='gimbal_aim_solver_node',
            output='screen',
            parameters=[gim_rough_cfg]
        )

    def get_laser_aim_node():
        return Node(
            package="laser_aim_controller",
            executable="laser_aim_controller_node",
            name="laser_aim_controller_node",
            output="screen",
            parameters=[laser_aim_cfg]
        )

    def get_gimbal_bridge_node():
        return Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            parameters=[gim_bridge_params],
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

    # -----------------------------
    # ComposableNode
    # -----------------------------
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                # 地图
                'target_pcd_path': map_pcd_path,

                # 初值模式: offline / online / auto
                'initial_pose_mode': 'online',

                # online 失败是否自动退回 offline
                'fallback_to_offline_if_online_failed': True,

                # offline 模式备用参数
                'initial_x': 0.0,
                'initial_y': 0.0,
                'initial_z': 0.0,
                'initial_roll': 0.0,
                'initial_pitch': 0.0,
                'initial_yaw': 0.0,

                # online 模式读取两个 yaml
                'camera_extrinsic_yaml': camera_extrinsic_yaml,
                'camera_to_lidar_yaml': camera_to_lidar_yaml,

                # GICP 总时间超时
                'enable_gicp_total_timeout': True,
                'gicp_total_timeout_sec': 20.0,
                'fallback_to_initial_after_timeout': True,

                # GICP 参数
                'gicp_max_iterations': 20,
                'gicp_transformation_epsilon': 1e-8,
                'gicp_euclidean_fitness_epsilon': 1e-6,
                'gicp_success_fitness_threshold': 0.15,

                # 点云累积与滤波
                'accumulate_time': 40,
                'grid_size_degrees': 0.05,
                'downsample_leaf_size': 0.2,

                # ROI
                'roi_min_x': 0.0,
                'roi_max_x': 12.0,
                'roi_min_y': -10.0,
                'roi_max_y': 10.0,
                'roi_max_z': 5.0,

                # 异步TF/GICP频率
                'tf_publish_rate_hz': 20.0,
                'gicp_worker_rate_hz': 0.1,
            }],
            extra_arguments=[
                {'use_intra_process_comms': False}
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
                {'use_intra_process_comms': True}
            ]
        )

    def get_cluster_node():
        return ComposableNode(
            package='cluster',
            plugin='tdt_radar::Cluster',
            name='cluster_node',
            extra_arguments=[
                {'use_intra_process_comms': True}
            ]
        )

    def get_detect_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Detect',
            name='radar_detect_node',
            parameters=[{
                "image_topic": "/camera0/image",
            }],
            extra_arguments=[
                {'use_intra_process_comms': True}
            ]
        )

    def get_resolve_node():
        return ComposableNode(
            package='tdt_vision',
            plugin='tdt_radar::Resolve',
            name='radar_resolve_node',
            parameters=[{
                "out_matrix_path": "/home/user/code/T-DT_Radar-main/config/out_matrix_gc.yaml",
                "camera_params_path": "/home/user/code/T-DT_Radar-main/config/camera_params.yaml",
                "region_points_path": "/home/user/code/T-DT_Radar-main/config/GCZX_Points.yaml",
                "minimap_path": "/home/user/code/T-DT_Radar-main/config/jichuzhizao.png",
                "is_test": True,
            }],
            extra_arguments=[
                {'use_intra_process_comms': True}
            ]
        )

    def get_kalman_filter_node():
        return ComposableNode(
            package='kalman_filter',
            plugin='tdt_radar::KalmanFilter',
            name='kalman_filter_node',
            extra_arguments=[
                {'use_intra_process_comms': True}
            ]
        )

    def get_livox_driver_node():
        return Node(
            package='livox_ros2_driver',
            executable='livox_ros2_driver_node',
            name='livox_lidar_publisher',
            output='screen',
            parameters=livox_ros2_params
        )

    # -----------------------------
    # 容器
    # -----------------------------
    container = ComposableNodeContainer(
        name='radar_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
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

    # -----------------------------
    # 日志
    # -----------------------------
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")
    log_debug_map_path = LogInfo(
        msg=f"Debug map path: {os.path.join(debug_map_share_dir, 'config', 'RM2025.png')}"
    )
    log_camera_extrinsic = LogInfo(msg=f"camera_extrinsic_yaml: {camera_extrinsic_yaml}")
    log_camera_to_lidar = LogInfo(msg=f"camera_to_lidar_yaml: {camera_to_lidar_yaml}")

    # -----------------------------
    # RViz2
    # -----------------------------
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )

    mappcd_node = get_map_pcd_publisher()

    integrator = Node(
        package='livox_cloud_integrator',
        executable='cloud_integrator_node',
        name='cloud_integrator',
        output='screen',
        parameters=[
            {"input_topic": '/livox/lidar'},
            {"output_topic": '/livox/lidar_integrated'},
            {"window_sec": 0.5},
            {"use_header_stamp": True},
            {"max_points": int(2000000)},
        ],
    )

    return LaunchDescription([
        log_rviz_path,
        log_map_path,
        log_debug_map_path,
        log_camera_extrinsic,
        log_camera_to_lidar,

        # 普通节点
        get_hik_camera_node(),
        # get_auto_aim_node(),
        get_gimbal_rough_node(),
        get_laser_aim_node(),
        get_gimbal_bridge_node(),
        get_new_map_node(),
        get_livox_driver_node(),
        integrator,

        # 组件容器
        container,

        # 辅助
        mappcd_node,
        rviz_node
    ])