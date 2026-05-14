import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events.process import ShutdownProcess, matches_executable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


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


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _build_launch(context):
    team = _team_config(LaunchConfiguration("our_color").perform(context))
    referee_rx_mode = LaunchConfiguration("referee_rx_mode").perform(context)
    referee_tx_mode = LaunchConfiguration("referee_tx_mode").perform(context)
    referee_serial_device = LaunchConfiguration("referee_serial_device").perform(context)
    referee_serial_baudrate = int(LaunchConfiguration("referee_serial_baudrate").perform(context))
    gimbal_serial_device = LaunchConfiguration("gimbal_serial_device").perform(context)
    referee_publish_raw_debug = ParameterValue(
        LaunchConfiguration("referee_publish_raw_debug"),
        value_type=bool,
    )
    publish_detect_debug_image = ParameterValue(
        LaunchConfiguration("publish_detect_debug_image"),
        value_type=bool,
    )
    decision_params_file = LaunchConfiguration("decision_params_file").perform(context)
    pdu_poll_period_ms = int(LaunchConfiguration("pdu_poll_period_ms").perform(context))
    enable_calibration = _as_bool(LaunchConfiguration("enable_calibration").perform(context))
    show_debug_map = _as_bool(LaunchConfiguration("show_debug_map").perform(context))
    calib_image_file = LaunchConfiguration("calib_image_file").perform(context).strip()

    bringup_share = get_package_share_directory("radar_bringup")
    calib_tool_share = get_package_share_directory("camera_field_calib_tool")
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
    map_pcd_path = os.path.join(localization_share, "config", "RM2026_new.pcd")
    camera_extrinsic_yaml = LaunchConfiguration("calib_out_matrix_yaml").perform(context)
    resolve_camera_params_yaml = os.path.join(bringup_config, "camera_params_rect.yaml")
    resolve_region_points_yaml = os.path.join(bringup_config, "RM2025_Points.yaml")
    resolve_minimap_png = os.path.join(bringup_config, "RM2025.png")
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
    enemy_uav_coord = (26.831, 1.969, 1.5)
    enemy_uav_default = {
        "uav_enemy_default_x": enemy_uav_coord[0],
        "uav_enemy_default_y": enemy_uav_coord[1],
        "uav_enemy_default_z": enemy_uav_coord[2],
        "uav_use_enemy_default_by_identity": True,
        "uav_default_out_blue_x": 0.0,
        "uav_default_out_blue_y": 0.0,
        "uav_default_out_blue_z": 0.0,
        "uav_default_out_red_x": 0.0,
        "uav_default_out_red_y": 0.0,
        "uav_default_out_red_z": 0.0,
    }
    if team["match_self_color"] == 2:
        enemy_uav_default.update({
            "uav_default_out_blue_x": enemy_uav_coord[0],
            "uav_default_out_blue_y": enemy_uav_coord[1],
            "uav_default_out_blue_z": enemy_uav_coord[2],
        })
    else:
        enemy_uav_default.update({
            "uav_default_out_red_x": enemy_uav_coord[0],
            "uav_default_out_red_y": enemy_uav_coord[1],
            "uav_default_out_red_z": enemy_uav_coord[2],
        })

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
        parameters=[
            {
                "image_topic": "/camera0_rect/image",
                "show_debug_window": True,
                "publish_debug_image": publish_detect_debug_image,
                "debug_image_topic": "/detect/debug_image",
            }
        ],
        extra_arguments=intra,
    )

    camera0_rectifier_component = make_rectifier_component("camera0")

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
                "online_initial_y_offset": 15.0,
                "enable_gicp_total_timeout": True,
                "gicp_total_timeout_sec": 30.0,
                "fallback_to_initial_after_timeout": True,
                "gicp_max_iterations": 100,
                "gicp_transformation_epsilon": 1e-8,
                "gicp_euclidean_fitness_epsilon": 1e-6,
                "gicp_success_fitness_threshold": 0.25,
                "accumulate_time": 30,
                "grid_size_degrees": 0.05,
                "downsample_leaf_size": 0.2,
                "roi_min_x": 0.0,
                "roi_max_x": 29.0,
                "roi_min_y": -6.5,
                "roi_max_y": 8.5,
                "roi_max_z": 1.5,
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
                "is_test": False,
                "self_color_override": team["match_self_color"],
                "use_match_info_color": False,
                "use_referee_identity_color": True,
                "referee_identity_topic": "/radar/referee/robot_identity",
                "uav_publish_default_output": True,
                **enemy_uav_default,
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
                "use_referee_identity_color": True,
                "semantic_tracking_enabled": True,
                "semantic_history_size": 20,
                "semantic_confirm_min_hits": 2,
                "semantic_switch_min_hits": 3,
                "fixed_track_confirm_hits": 2,
                "pending_seed_confirm_hits": 2,
                "semantic_vote_history_size": 6,
                "tentative_miss_limit": 2,
                "lost_miss_limit": 100,
                "lost_prediction_publish_ms": 0,
                "camera_publish_hold_ms": 120,
                "lidar_publish_hold_ms": 100,
                "recent_camera_constraint_ms": 500,
                "frame_sync_ms": 50,
                "box_iou_history_size": 6,
                "visual_bootstrap_on_first_frame": True,
                "tentative_accept_same_id_camera": True,
                "car_lidar_update_enabled": False,
                "lidar_match_gate_m": 1.2,
                "lost_lidar_match_gate_m": 2.0,
                "camera_match_gate_m": 1.2,
                "camera_dynamic_speed_gain": 1.0,
                "vision_noise_margin_m": 0.8,
                "box_consistency_gate_relax_m": 2.0,
                "pending_seed_gate_m": 1.2,
                "box_iou_confirm_threshold": 0.25,
                "box_center_diag_ratio": 0.75,
                "publish_lost_prediction_to_decision": False,
                "blind_guess_enabled": False,
                "blind_guess_config_path": os.path.join(bringup_config, "guess_pts.yaml"),
                "blind_guess_after_ms": 250,
                "blind_guess_hold_ms": 1200,
                "kf_predict_hold_ms": 1500,
                "max_predict_step_m": 7.5,
                "max_predict_speed_mps": 5.0,
                "blind_guess_distance_weight": 0.6,
                "blind_guess_direction_weight": 0.4,
                "blind_guess_d_factor": 0.15,
                "blind_guess_min_speed_mps": 0.2,
                "blind_guess_publish_to_detect": False,
                "blind_guess_publish_to_sentry": False,
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
                "use_referee_identity_color": True,
                "input_detect_topic": "/kalman_detect",
                "output_our_tracks_topic": "/radar/perception/our_tracks",
                "output_enemy_tracks_topic": "/radar/perception/enemy_tracks",
            }
        ],
        extra_arguments=intra,
    )

    debug_map_component = ComposableNode(
        package="debug_map",
        plugin="debug_map::DebugMap",
        name="debug_map_node",
        parameters=[
            {
                "map_path": os.path.join(debug_map_share, "config", "RM2026.png"),
                "use_match_info_color": False,
                "self_color_override": team["match_self_color"],
                "use_referee_identity_color": True,
            }
        ],
        extra_arguments=intra,
    )
    debug_map_components = [debug_map_component] if show_debug_map else []

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
            *debug_map_components,
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
                        "publish_raw_debug": referee_publish_raw_debug,
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
            {
                "target_side": team["gimbal_target_side"],
                "camera_to_lidar_yaml": camera_to_lidar_yaml,
            },
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
        parameters=[
            gimbal_bridge_cfg,
            {
                "serial_ports": [gimbal_serial_device],
                "serial_port": gimbal_serial_device,
            },
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
        parameters=[
            {
                "default_identity": team["decision_my_color"].lower(),
            }
        ],
    )

    gnuradio_rx_launcher_node = Node(
        package="radar_pdu_bridge",
        executable="gnuradio_rx_launcher",
        name="gnuradio_rx_launcher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_gnuradio")),
        parameters=[
            {
                "script_path": LaunchConfiguration("gnuradio_script_path"),
                "default_identity": team["decision_my_color"].lower(),
                "allow_direct_13": ParameterValue(
                    LaunchConfiguration("gnuradio_allow_direct_13"),
                    value_type=bool,
                ),
                "access_code0": LaunchConfiguration("gnuradio_access_code0"),
                "access_code1": LaunchConfiguration("gnuradio_access_code1"),
            }
        ],
    )

    referee_link_monitor_node = Node(
        package="radar_bringup",
        executable="referee_link_monitor.py",
        name="referee_link_monitor",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_referee_link_monitor")),
        parameters=[
            {
                "enabled": ParameterValue(
                    LaunchConfiguration("enable_referee_link_monitor"),
                    value_type=bool,
                ),
                "game_status_topic": "/radar/referee/game_status",
                "warn_after_sec": ParameterValue(
                    LaunchConfiguration("referee_link_warn_after_sec"),
                    value_type=float,
                ),
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
    )

    track_status_window_node = Node(
        package="debug_map",
        executable="track_status_window_node",
        name="track_status_window_node",
        output="screen",
        parameters=[
            {
                "detect_topic": "/detect_result",
                "resolve_topic": "/resolve_result",
                "kalman_topic": "/kalman_detect",
                "debug_map_topic": "/debug_map_result",
                "status_topic": "/kalman_track_status",
                "identity_topic": "/radar/referee/robot_identity",
                "self_color_override": team["match_self_color"],
                "out_matrix_path": camera_extrinsic_yaml,
                "camera_params_path": resolve_camera_params_yaml,
                "region_points_path": resolve_region_points_yaml,
                "uav_intersection_debug_topic": "/uav_intersection_debug",
                "uav_cluster_debug_topic": "/uav_cluster_debug",
                "uav_resolve_debug_status_topic": "/uav_resolve_debug_status",
            }
        ],
    )

    referee_status_window_node = Node(
        package="debug_map",
        executable="referee_status_window_node",
        name="referee_status_window_node",
        output="screen",
        parameters=[
            {
                "game_status_topic": "/radar/referee/game_status",
                "identity_topic": "/radar/referee/robot_identity",
                "enemy_hp_topic": "/radar/referee/enemy_hp",
                "dart_status_topic": "/radar/referee/dart_status",
                "mark_progress_topic": "/radar/referee/mark_progress",
                "radar_sync_topic": "/radar/referee/radar_decision_sync",
                "raw_rx_topic": "/radar/debug/referee_raw_bytes",
                "raw_tx_topic": "/radar/debug/referee_tx_raw_bytes",
                "pdu_all_topic": "/radar/all_pdu",
                "pdu_info_topic": "/radar/info_pdu",
                "pdu_jam_topic": "/radar/jam_pdu",
                "wireless_hp_topic": "/radar/wireless/enemy_hp",
                "wireless_ammo_topic": "/radar/wireless/enemy_ammo",
                "wireless_macro_topic": "/radar/wireless/enemy_macro_state",
                "wireless_buffs_topic": "/radar/wireless/enemy_buffs",
                "wireless_key_topic": "/radar/wireless/interference_key",
                "tx_map_topic": "/radar/tx/map_robot_data",
                "tx_interaction_topic": "/radar/tx/robot_interaction",
                "tx_radar_cmd_topic": "/radar/tx/radar_cmd",
            }
        ],
    )

    referee_rosbag_recorder_node = Node(
        package="radar_bringup",
        executable="referee_rosbag_recorder.py",
        name="referee_rosbag_recorder",
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_rosbag")),
        parameters=[
            {
                "enabled": ParameterValue(LaunchConfiguration("record_rosbag"), value_type=bool),
                "output_dir": LaunchConfiguration("rosbag_output_dir"),
                "record_duration_sec": ParameterValue(
                    LaunchConfiguration("rosbag_record_duration_sec"),
                    value_type=float,
                ),
                "game_status_topic": "/radar/referee/game_status",
                "start_game_progress": 4,
                "auto_start_after_sec": ParameterValue(
                    LaunchConfiguration("rosbag_auto_start_after_sec"),
                    value_type=float,
                ),
                "record_topics": [
                    "/camera0/image",
                    "/camera0/camera_info",
                    "/camera2/image",
                    "/camera2/camera_info",
                    "/livox/lidar",
                    "/livox/imu",
                ],
            }
        ],
    )

    detect_debug_image_view_node = Node(
        package="image_view",
        executable="image_view",
        name="detect_debug_image_view",
        output="screen",
        remappings=[("image", "/detect/debug_image")],
        condition=IfCondition(LaunchConfiguration("show_detect_debug_window")),
        parameters=[
            {
                "window_name": "detect",
                "autosize": False,
                "width": 1080,
                "height": 960,
            }
        ],
    )

    referee_startup_actions = [
        radar_decision_comm_container,
        referee_link_monitor_node,
        referee_status_window_node,
    ]

    post_calibration_flow_actions = [
        LogInfo(msg=f"radar_launch our_color={team['decision_my_color']}"),
        LogInfo(msg=f"decision params file: {decision_params_file}"),
        LogInfo(msg=f"referee rx_mode={referee_rx_mode} tx_mode={referee_tx_mode}"),
        LogInfo(msg=f"referee serial device={referee_serial_device} baudrate={referee_serial_baudrate}"),
        LogInfo(msg=f"gimbal serial device={gimbal_serial_device}"),
        LogInfo(msg=f"debug map enabled={show_debug_map}"),
        LogInfo(
            msg=["match rosbag recorder enabled, output_dir=", LaunchConfiguration("rosbag_output_dir")],
            condition=IfCondition(LaunchConfiguration("record_rosbag")),
        ),
        gimbal_solver_node,
        laser_aim_node,
        gimbal_bridge_node,
        livox_driver_node,
        cloud_integrator_node,
        radar_decision_sync_shm_node,
        gnuradio_rx_launcher_node,
        shm_pdu_broadcast_node,
        radar_hot_path_container,
        radar_debug_container,
        map_pcd_publisher,
        referee_rosbag_recorder_node,
        track_status_window_node,
        detect_debug_image_view_node,
        rviz_node,
    ]
    normal_flow_actions = [
        *referee_startup_actions,
        *post_calibration_flow_actions,
    ]

    if not enable_calibration:
        return normal_flow_actions

    calib_launch_file = os.path.join(calib_tool_share, "launch", "pnp_calib_new.launch.py")
    use_static_image_file = bool(calib_image_file)

    pnp_calib_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(calib_launch_file),
        launch_arguments={
            "start_hik_camera": "false" if use_static_image_file else "true",
            "start_static_image_publisher": "true" if use_static_image_file else "false",
            "static_image_file": LaunchConfiguration("calib_image_file"),
            "static_image_rate": LaunchConfiguration("calib_static_image_rate"),
            "static_image_frame": LaunchConfiguration("calib_camera_frame"),
            "image_topic": LaunchConfiguration("calib_image_topic"),
            "intrinsics_yaml": LaunchConfiguration("calib_intrinsics_yaml"),
            "out_matrix_yaml": LaunchConfiguration("calib_out_matrix_yaml"),
            "reference_points_yaml": LaunchConfiguration("calib_reference_points_yaml"),
            "use_refine": LaunchConfiguration("calib_use_refine"),
            "pnp_method": LaunchConfiguration("calib_pnp_method"),
            "min_points": LaunchConfiguration("calib_min_points"),
            "freeze_first_frame": "false" if use_static_image_file else "true",
            "shutdown_on_save": "true",
        }.items(),
    )

    calib_source_executable = (
        "image_publisher_node" if use_static_image_file else "hik_camera_ros2_driver_node"
    )
    stop_calib_image_source = EmitEvent(
        event=ShutdownProcess(
            process_matcher=matches_executable(calib_source_executable),
        )
    )

    start_normal_flow_after_source_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=matches_executable(calib_source_executable),
            on_exit=[
                LogInfo(msg="Calibration image source exited."),
                LogInfo(msg="Starting normal radar flow with the saved out_matrix_gc.yaml."),
                *post_calibration_flow_actions,
            ],
        )
    )

    def on_pnp_calib_exit(event, _context):
        if event.returncode != 0:
            return [
                LogInfo(
                    msg=(
                        "PnP calibration exited without a saved result; "
                        "normal radar flow will not start."
                    )
                ),
                stop_calib_image_source,
                Shutdown(reason="PnP calibration did not save out_matrix_gc.yaml"),
            ]
        return [
            LogInfo(msg="PnP calibration saved and exited; stopping calibration image source."),
            start_normal_flow_after_source_exit,
            stop_calib_image_source,
        ]

    start_normal_flow_after_calib = RegisterEventHandler(
        OnProcessExit(
            target_action=matches_executable("pnp_calib_node_new"),
            on_exit=on_pnp_calib_exit,
        )
    )

    calib_source_msg = (
        f"calibration uses static image file: {calib_image_file}"
        if use_static_image_file
        else "calibration captures the first panoramic camera frame and keeps it still"
    )

    return [
        LogInfo(msg="Starting referee communication and referee status panel before calibration."),
        *referee_startup_actions,
        LogInfo(msg="enable_calibration=true: start PnP calibration first. Press P to solve, S to save and continue."),
        LogInfo(msg=calib_source_msg),
        LogInfo(msg=f"calibration output: {camera_extrinsic_yaml}"),
        start_normal_flow_after_calib,
        pnp_calib_launch,
    ]


def generate_launch_description():
    radar_decision_share = get_package_share_directory("radar_decision_comm")
    calib_tool_share = get_package_share_directory("camera_field_calib_tool")
    bringup_share = get_package_share_directory("radar_bringup")
    default_decision_params = os.path.join(
        radar_decision_share, "config", "radar_decision_comm.yaml"
    )
    default_calib_intrinsics = os.path.join(
        calib_tool_share, "config", "camera_intrinsics.yaml"
    )
    default_calib_reference_points = os.path.join(
        calib_tool_share, "config", "reference_points_RM.yaml"
    )
    default_calib_out_matrix = os.path.join(
        bringup_share, "config", "out_matrix_RM_CS.yaml"
    )

    # WHU
    return LaunchDescription(
        [
            # ==================== MOST IMPORTANT MANUAL PARAMETERS ====================
            # 1) our_color: fallback team color when referee identity is unavailable.
            # 2) referee_serial_device: referee system serial port.
            # 3) gimbal_serial_device: gimbal serial port.
            DeclareLaunchArgument(
                "our_color",
                default_value="blue",
                description="IMPORTANT: fallback team color when referee identity is unavailable: red or blue",
            ),
            DeclareLaunchArgument(
                "referee_serial_device",
                default_value="/dev/ttyUSB0",
                description="IMPORTANT: referee system serial device path, distinct from gimbal serial device",
            ),
            DeclareLaunchArgument(
                "gimbal_serial_device",
                default_value="/dev/ttyACM0",
                description="IMPORTANT: gimbal serial device path",
            ),
            # ==================== Referee Communication ====================
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
                "referee_serial_baudrate",
                default_value="115200",
                description="referee serial baudrate",
            ),
            DeclareLaunchArgument(
                "referee_publish_raw_debug",
                default_value="false",
                description="publish raw referee serial RX bytes to /radar/debug/referee_raw_bytes",
            ),
            DeclareLaunchArgument(
                "enable_referee_link_monitor",
                default_value="true",
                description="monitor referee game_status reception after launch startup",
            ),
            DeclareLaunchArgument(
                "referee_link_warn_after_sec",
                default_value="3.0",
                description="warn if no referee game_status message is received within this many seconds",
            ),
            DeclareLaunchArgument(
                "publish_detect_debug_image",
                default_value="false",
                description="publish detect debug image to /detect/debug_image",
            ),
            DeclareLaunchArgument(
                "show_detect_debug_window",
                default_value="true",
                description="open image_view window for /detect/debug_image",
            ),
            DeclareLaunchArgument(
                "show_debug_map",
                default_value="true",
                description="open debug_map minimap window and publish /map_2d, /debug_map_result",
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
            DeclareLaunchArgument(
                "start_gnuradio",
                default_value="true",
                description="start GNU Radio RX flowgraph together with radar launch",
            ),
            DeclareLaunchArgument(
                "gnuradio_script_path",
                default_value="/home/user/gnuradio_grc/0416/rx_red_gfsk.py",
                description="GNU Radio generated RX python script path",
            ),
            DeclareLaunchArgument(
                "gnuradio_allow_direct_13",
                default_value="true",
                description="allow GNU Radio RX key level to jump directly from 1 to 3",
            ),
            DeclareLaunchArgument(
                "gnuradio_access_code0",
                default_value="2F6F4C74B914492E",
                description="GNU Radio info-wave access code",
            ),
            DeclareLaunchArgument(
                "gnuradio_access_code1",
                default_value="16E8D377151C712D",
                description="GNU Radio jam-wave access code",
            ),
            DeclareLaunchArgument(
                "record_rosbag",
                default_value="false",
                description=(
                    "arm match-triggered rosbag recording; records raw "
                    "camera0/camera2 images, camera_info, and Livox topics when "
                    "referee game_progress enters 4"
                ),
            ),
            DeclareLaunchArgument(
                "rosbag_output_dir",
                default_value="/home/user/DATA/RADAR_ROSBAG",
                description="base directory for match-triggered rosbag recordings",
            ),
            DeclareLaunchArgument(
                "rosbag_record_duration_sec",
                default_value="540.0",
                description="match-triggered rosbag recording duration in seconds",
            ),
            DeclareLaunchArgument(
                "rosbag_auto_start_after_sec",
                default_value="60.0",
                description=(
                    "automatically start rosbag recording this many seconds after "
                    "the recorder is armed if referee game_progress has not triggered it; "
                    "<=0 disables auto-start"
                ),
            ),
            DeclareLaunchArgument(
                "enable_calibration",
                default_value="false",
                description="whether to run pnp_calib_new.launch.py before starting the normal radar flow",
            ),
            DeclareLaunchArgument(
                "calib_image_file",
                default_value="",
                description="optional static panoramic image file; empty captures and freezes the first live camera frame",
            ),
            DeclareLaunchArgument(
                "calib_image_topic",
                default_value="/camera0/image",
                description="image topic used by pnp_calib_node_new during calibration",
            ),
            DeclareLaunchArgument(
                "calib_static_image_rate",
                default_value="5.0",
                description="publish rate when calib_image_file is used",
            ),
            DeclareLaunchArgument(
                "calib_camera_frame",
                default_value="calib_camera",
                description="frame id for static calibration image publishing",
            ),
            DeclareLaunchArgument(
                "calib_intrinsics_yaml",
                default_value=default_calib_intrinsics,
                description="camera intrinsics yaml used by pnp_calib_node_new",
            ),
            DeclareLaunchArgument(
                "calib_out_matrix_yaml",
                default_value=default_calib_out_matrix,
                description="output out_matrix yaml; normal flow reads this after calibration exits",
            ),
            DeclareLaunchArgument(
                "calib_reference_points_yaml",
                default_value=default_calib_reference_points,
                description="reference 3D points yaml used by pnp_calib_node_new",
            ),
            DeclareLaunchArgument(
                "calib_use_refine",
                default_value="true",
                description="whether pnp_calib_node_new refines solvePnP result",
            ),
            DeclareLaunchArgument(
                "calib_pnp_method",
                default_value="0",
                description="OpenCV solvePnP method flag for pnp_calib_node_new",
            ),
            DeclareLaunchArgument(
                "calib_min_points",
                default_value="7",
                description="minimum selected point count before pnp_calib_node_new solves",
            ),
            OpaqueFunction(function=_build_launch),
        ]
    )
