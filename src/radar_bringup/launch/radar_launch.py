import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def _team_config(our_color: str):
    color = our_color.strip().lower()
    if color == "red":
        return {
            "decision_my_color": "Red",
            "match_self_color": 2,
            "gimbal_target_side": "blue",
            "aim_target_color": 1,
            "aim_swap_rb_input": False,
            "my_hero_id": 1,
            "my_sentinel_id": 7,
            "infantry_3_id": 3,
            "infantry_4_id": 4,
            "enemy_hero_id": 101,
        }
    if color == "blue":
        return {
            "decision_my_color": "Blue",
            "match_self_color": 0,
            "gimbal_target_side": "red",
            "aim_target_color": 0,
            "aim_swap_rb_input": True,
            "my_hero_id": 101,
            "my_sentinel_id": 107,
            "infantry_3_id": 103,
            "infantry_4_id": 104,
            "enemy_hero_id": 1,
        }
    raise RuntimeError(f"our_color must be 'red' or 'blue', got: {our_color}")


def _build_launch(context):
    team = _team_config(LaunchConfiguration("our_color").perform(context))
    referee_rx_mode = LaunchConfiguration("referee_rx_mode").perform(context)
    referee_tx_mode = LaunchConfiguration("referee_tx_mode").perform(context)
    referee_serial_device = LaunchConfiguration("referee_serial_device").perform(context)
    referee_serial_baudrate = int(LaunchConfiguration("referee_serial_baudrate").perform(context))
    decision_params_file = LaunchConfiguration("decision_params_file").perform(context)
    pdu_poll_period_ms = int(LaunchConfiguration("pdu_poll_period_ms").perform(context))

    bringup_share = get_package_share_directory("radar_bringup")
    localization_share = get_package_share_directory("localization")
    livox_share = get_package_share_directory("livox_ros2_driver")
    debug_map_share = get_package_share_directory("debug_map")
    hik_share = get_package_share_directory("hik_camera_ros2_driver")
    gimbal_solver_share = get_package_share_directory("gimbal_aim_solver")
    gimbal_bridge_share = get_package_share_directory("gimbal_serial_bridge")
    auto_aim_share = get_package_share_directory("auto_aim_fast")
    laser_aim_share = get_package_share_directory("laser_aim_controller")

    bringup_config = os.path.join(bringup_share, "config")
    hik_params_file = os.path.join(hik_share, "config", "camera_params.yaml")
    map_pcd_path = os.path.join(localization_share, "config", "基础制造_rm_frame.pcd")
    camera_extrinsic_yaml = os.path.join(bringup_config, "out_matrix_gc.yaml")
    resolve_camera_params_yaml = os.path.join(bringup_config, "camera_params_rect.yaml")
    resolve_region_points_yaml = os.path.join(bringup_config, "GCZX_Points.yaml")
    resolve_minimap_png = os.path.join(bringup_config, "jichuzhizao.png")
    camera_to_lidar_yaml = os.path.join(localization_share, "config", "camera_to_lidar.yaml")
    manual_yaml = os.path.join(localization_share, "config", "livox_rm_manual.yaml")
    rviz_config_path = os.path.join(localization_share, "config", "localization_rviz_config.rviz")

    livox_config_dir = os.path.join(livox_share, "config")
    user_config_path = os.path.join(livox_config_dir, "livox_lidar_config.json")

    gimbal_solver_cfg = os.path.join(gimbal_solver_share, "config", "gimbal_aim_solver.yaml")
    gimbal_bridge_cfg = os.path.join(gimbal_bridge_share, "config", "gimbal_serial_bridge.yaml")
    laser_aim_cfg = os.path.join(laser_aim_share, "config", "laser_aim_controller.yaml")
    auto_aim_tracker_cfg = os.path.join(auto_aim_share, "config", "tracker.yaml")
    auto_aim_camera_cfg = os.path.join(auto_aim_share, "config", "camera.yaml")
    auto_aim_detector_cfg = os.path.join(auto_aim_share, "config", "detector.yaml")
    auto_aim_qos_cfg = os.path.join(auto_aim_share, "config", "qos.yaml")
    auto_aim_info_cfg = os.path.join(auto_aim_share, "config", "camera_info.yaml")

    intra = [{"use_intra_process_comms": True}]

    def make_rectifier_component(camera_name: str):
        return ComposableNode(
            package="hik_camera_ros2_driver",
            plugin="hik_camera_ros2_driver::HikCameraRectifierNode",
            name=f"{camera_name}_rectifier",
            parameters=[
                {
                    "input_image_topic": f"/{camera_name}/image",
                    "input_camera_info_topic": f"/{camera_name}/camera_info",
                    "output_image_topic": f"/{camera_name}_rect/image",
                    "output_camera_info_topic": f"/{camera_name}_rect/camera_info",
                    "use_sensor_data_qos": True,
                }
            ],
            extra_arguments=intra,
        )

    hik_camera_component = ComposableNode(
        package="hik_camera_ros2_driver",
        plugin="hik_camera_ros2_driver::HikCameraRos2DriverNode",
        name="hik_camera_ros2_driver",
        parameters=[hik_params_file],
        extra_arguments=intra,
    )

    detect_component = ComposableNode(
        package="tdt_vision",
        plugin="tdt_radar::Detect",
        name="radar_detect_node",
        parameters=[{"image_topic": "/camera0_rect/image"}],
        extra_arguments=intra,
    )

    camera0_rectifier_component = make_rectifier_component("camera0")
    camera1_rectifier_component = make_rectifier_component("camera1")
    camera2_rectifier_component = make_rectifier_component("camera2")

    localization_component = ComposableNode(
        package="localization",
        plugin="tdt_radar::Localization",
        name="localization_node",
        parameters=[
            manual_yaml,
            {
                "target_pcd_path": map_pcd_path,
                "initial_pose_mode": "online",
                "fallback_to_offline_if_online_failed": True,
                "initial_x": 0.0,
                "initial_y": 0.0,
                "initial_z": 0.0,
                "initial_roll": 0.0,
                "initial_pitch": 0.0,
                "initial_yaw": 0.0,
                "camera_extrinsic_yaml": camera_extrinsic_yaml,
                "camera_to_lidar_yaml": camera_to_lidar_yaml,
                "enable_gicp_total_timeout": True,
                "gicp_total_timeout_sec": 5.0,
                "fallback_to_initial_after_timeout": True,
                "gicp_max_iterations": 100,
                "gicp_transformation_epsilon": 1e-8,
                "gicp_euclidean_fitness_epsilon": 1e-6,
                "gicp_success_fitness_threshold": 0.3,
                "accumulate_time": 30,
                "grid_size_degrees": 0.05,
                "downsample_leaf_size": 0.2,
                "roi_min_x": 0.0,
                "roi_max_x": 28.0,
                "roi_min_y": -15.0,
                "roi_max_y": 15.0,
                "roi_max_z": 1.0,
                "tf_publish_rate_hz": 20.0,
                "gicp_worker_rate_hz": 20.0,
            },
        ],
        extra_arguments=[{"use_intra_process_comms": False}],
    )

    dynamic_cloud_component = ComposableNode(
        package="dynamic_cloud",
        plugin="tdt_radar::DynamicCloud",
        name="dynamic_cloud_node",
        parameters=[{"map_pcd_path": map_pcd_path}],
        extra_arguments=intra,
    )

    cluster_component = ComposableNode(
        package="cluster",
        plugin="tdt_radar::Cluster",
        name="cluster_node",
        extra_arguments=intra,
    )

    resolve_component = ComposableNode(
        package="tdt_vision",
        plugin="tdt_radar::Resolve",
        name="radar_resolve_node",
        parameters=[
            {
                "out_matrix_path": camera_extrinsic_yaml,
                "camera_params_path": resolve_camera_params_yaml,
                "region_points_path": resolve_region_points_yaml,
                "minimap_path": resolve_minimap_png,
                "is_test": True,
                "uav_publish_default_output": False,
                "uav_default_out_blue_x": 0.0,
                "uav_default_out_blue_y": 0.0,
                "uav_default_out_blue_z": 0.0,
                "uav_default_out_red_x": 0.0,
                "uav_default_out_red_y": 0.0,
                "uav_default_out_red_z": 0.0,
            }
        ],
        extra_arguments=intra,
    )

    kalman_filter_component = ComposableNode(
        package="kalman_filter",
        plugin="tdt_radar::KalmanFilter",
        name="kalman_filter_node",
        parameters=[
            {
                "use_match_info_color": False,
                "self_color_override": team["match_self_color"],
            }
        ],
        extra_arguments=intra,
    )

    kalman_tracks_adapter_component = ComposableNode(
        package="radar_decision_comm",
        plugin="radar_decision_comm::KalmanTracksAdapterComponent",
        name="kalman_tracks_adapter",
        parameters=[
            {
                "our_color": team["decision_my_color"],
                "input_detect_topic": "/kalman_detect",
                "output_our_tracks_topic": "/radar/perception/our_tracks",
                "output_enemy_tracks_topic": "/radar/perception/enemy_tracks",
            }
        ],
        extra_arguments=intra,
    )

    auto_aim_component = ComposableNode(
        package="auto_aim_fast",
        plugin="auto_aim_fast::AutoAimNode",
        name="auto_aim_node",
        parameters=[
            auto_aim_tracker_cfg,
            auto_aim_camera_cfg,
            auto_aim_detector_cfg,
            auto_aim_qos_cfg,
            auto_aim_info_cfg,
            {
                "detector.target_color": team["aim_target_color"],
                "detector.yolo.swap_rb_input": team["aim_swap_rb_input"],
            },
        ],
        extra_arguments=intra,
    )

    radar_hot_path_container = ComposableNodeContainer(
        name="radar_hot_path_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            hik_camera_component,
            detect_component,
            camera0_rectifier_component,
            camera1_rectifier_component,
            camera2_rectifier_component,
            auto_aim_component,
        ],
        output="both",
        emulate_tty=True,
        on_exit=Shutdown(),
    )

    radar_debug_container = ComposableNodeContainer(
        name="radar_debug_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            localization_component,
            dynamic_cloud_component,
            cluster_component,
            resolve_component,
            kalman_filter_component,
            kalman_tracks_adapter_component,
        ],
        output="both",
        emulate_tty=True,
        on_exit=Shutdown(),
    )

    decision_overrides = {
        "my_color": team["decision_my_color"],
        "derive_ids_from_my_color": True,
        "my_hero_id": team["my_hero_id"],
        "my_sentinel_id": team["my_sentinel_id"],
        "infantry_3_id": team["infantry_3_id"],
        "infantry_4_id": team["infantry_4_id"],
        "enemy_hero_id": team["enemy_hero_id"],
        "hero_lob_receiver_id": team["my_hero_id"],
    }

    radar_decision_comm_container = ComposableNodeContainer(
        name="radar_decision_comm_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::RefereeSerialBridgeComponent",
                name="referee_serial_bridge",
                parameters=[
                    decision_params_file,
                    {
                        "my_color": team["decision_my_color"],
                        "rx_mode": referee_rx_mode,
                        "tx_mode": referee_tx_mode,
                        "device": referee_serial_device,
                        "baudrate": referee_serial_baudrate,
                    },
                ],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::RadarWirelessRxComponent",
                name="radar_wireless_rx",
                parameters=[
                    decision_params_file,
                    {
                        "my_color": team["decision_my_color"],
                    },
                ],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::ClientBridgeComponent",
                name="client_bridge",
                parameters=[decision_params_file],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::PerceptionAdapterComponent",
                name="perception_adapter",
                parameters=[decision_params_file],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::StateBlackboardComponent",
                name="state_blackboard",
                parameters=[decision_params_file],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::FusionComponent",
                name="fusion_engine",
                parameters=[decision_params_file],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::DecisionEngineComponent",
                name="decision_engine",
                parameters=[decision_params_file, decision_overrides],
                extra_arguments=intra,
            ),
            ComposableNode(
                package="radar_decision_comm",
                plugin="radar_decision_comm::TxSchedulerComponent",
                name="tx_scheduler",
                parameters=[decision_params_file],
                extra_arguments=intra,
            ),
        ],
    )

    gimbal_solver_node = Node(
        package="gimbal_aim_solver",
        executable="gimbal_aim_solver_node",
        name="gimbal_aim_solver_node",
        output="screen",
        parameters=[
            gimbal_solver_cfg,
            {"target_side": team["gimbal_target_side"]},
        ],
    )

    laser_aim_node = Node(
        package="laser_aim_controller",
        executable="laser_aim_controller_node",
        name="laser_aim_controller_node",
        output="screen",
        parameters=[laser_aim_cfg],
    )

    gimbal_bridge_node = Node(
        package="gimbal_serial_bridge",
        executable="gimbal_serial_bridge_node",
        name="gimbal_serial_bridge",
        output="screen",
        parameters=[gimbal_bridge_cfg],
    )

    debug_map_node = Node(
        package="debug_map",
        executable="debug_map_homo_node",
        name="debug_map_homo_node",
        output="screen",
        parameters=[
            {
                "map_path": os.path.join(debug_map_share, "config", "jichuzhizao.png"),
                "homography_path": os.path.join(debug_map_share, "config", "debug_map_homography.yaml"),
                "auto_load_homography": True,
                "auto_save_homography": True,
                "target_w": 700,
                "target_h": 375,
                "kalman_topic": "/kalman_detect",
                "resolve_topic": "/resolve_result",
                "uav_topic": "/uav_positions",
                "match_info_topic": "/match_info",
                "debug_image_topic": "/debug_map_homo",
                "draw_timeout_ms": 500,
                "show_window": True,
                "prefer_kalman": True,
                "use_match_info_color": False,
                "self_color_override": team["match_self_color"],
            }
        ],
    )

    livox_driver_node = Node(
        package="livox_ros2_driver",
        executable="livox_ros2_driver_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=[
            {"xfer_format": 0},
            {"multi_topic": 0},
            {"data_src": 0},
            {"publish_freq": 10.0},
            {"output_data_type": 0},
            {"frame_id": "livox_frame"},
            {"lvx_file_path": "/home/livox/livox_test.lvx"},
            {"user_config_path": user_config_path},
            {"cmdline_input_bd_code": "livox0000000001"},
        ],
    )

    cloud_integrator_node = Node(
        package="livox_cloud_integrator",
        executable="cloud_integrator_node",
        name="cloud_integrator",
        output="screen",
        parameters=[
            {"input_topic": "/livox/lidar"},
            {"output_topic": "/livox/lidar_integrated"},
            {"window_sec": 0.2},
            {"use_header_stamp": True},
            {"max_points": 200000},
        ],
    )

    map_pcd_publisher = Node(
        package="pcl_ros",
        executable="pcd_to_pointcloud",
        name="map_pcd_publisher",
        parameters=[
            {
                "file_name": map_pcd_path,
                "frame_id": "map",
                "publish_rate": 0.1,
            }
        ],
        remappings=[("cloud", "/map_point_cloud")],
        output="screen",
    )

    shm_pdu_broadcast_node = Node(
        package="radar_pdu_bridge",
        executable="shm_pdu_broadcast_node",
        name="shm_pdu_broadcast_node",
        output="screen",
        parameters=[
            {
                "poll_period_ms": pdu_poll_period_ms,
            }
        ],
    )

    radar_decision_sync_shm_node = Node(
        package="radar_pdu_bridge",
        executable="radar_decision_sync_shm_node",
        name="radar_decision_sync_shm_node",
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
    )

    return [
        LogInfo(msg=f"radar_launch our_color={team['decision_my_color']}"),
        LogInfo(msg=f"decision params file: {decision_params_file}"),
        LogInfo(msg=f"referee rx_mode={referee_rx_mode} tx_mode={referee_tx_mode}"),
        LogInfo(msg=f"referee serial device={referee_serial_device} baudrate={referee_serial_baudrate}"),
        gimbal_solver_node,
        laser_aim_node,
        gimbal_bridge_node,
        debug_map_node,
        livox_driver_node,
        cloud_integrator_node,
        shm_pdu_broadcast_node,
        radar_decision_sync_shm_node,
        radar_hot_path_container,
        radar_debug_container,
        radar_decision_comm_container,
        map_pcd_publisher,
        rviz_node,
    ]


def generate_launch_description():
    radar_decision_share = get_package_share_directory("radar_decision_comm")
    default_decision_params = os.path.join(
        radar_decision_share, "config", "radar_decision_comm.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "our_color",
                default_value="blue",
                description="our team color: red or blue",
            ),
            DeclareLaunchArgument(
                "referee_rx_mode",
                default_value="serial",
                description="referee_serial_rx mode: mock/topic/serial",
            ),
            DeclareLaunchArgument(
                "referee_tx_mode",
                default_value="serial",
                description="referee_serial_tx mode: topic/serial",
            ),
            DeclareLaunchArgument(
                "referee_serial_device",
                default_value="/dev/ttyUSB20",
                description="referee serial device path, distinct from gimbal serial device",
            ),
            DeclareLaunchArgument(
                "referee_serial_baudrate",
                default_value="115200",
                description="referee serial baudrate",
            ),
            DeclareLaunchArgument(
                "decision_params_file",
                default_value=default_decision_params,
                description="radar_decision_comm parameter file",
            ),
            DeclareLaunchArgument(
                "pdu_poll_period_ms",
                default_value="2",
                description="shared-memory PDU poll period in milliseconds",
            ),
            OpaqueFunction(function=_build_launch),
        ]
    )
