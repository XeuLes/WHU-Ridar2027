import os
import signal

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events.process import ShutdownProcess, matches_executable
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
            "my_hero_id": 101,
            "my_sentinel_id": 107,
            "infantry_3_id": 103,
            "infantry_4_id": 104,
            "enemy_hero_id": 1,
        }
    raise RuntimeError(f"our_color must be 'red' or 'blue', got: {our_color}")


def _build_launch(context):
    team = _team_config(LaunchConfiguration("our_color").perform(context))
    decision_params_file = LaunchConfiguration("decision_params_file").perform(context)
    referee_rx_mode = LaunchConfiguration("referee_rx_mode").perform(context)
    referee_tx_mode = LaunchConfiguration("referee_tx_mode").perform(context)
    referee_serial_device = LaunchConfiguration("referee_serial_device").perform(context)
    referee_serial_baudrate = int(LaunchConfiguration("referee_serial_baudrate").perform(context))
    referee_publish_raw_debug = ParameterValue(
        LaunchConfiguration("referee_publish_raw_debug"),
        value_type=bool,
    )
    pdu_poll_period_ms = int(LaunchConfiguration("pdu_poll_period_ms").perform(context))
    rosbag_file = LaunchConfiguration("rosbag_file").perform(context)
    lidar_input_topic = LaunchConfiguration("lidar_input_topic").perform(context)
    image_input_topic = LaunchConfiguration("image_input_topic").perform(context)
    compressed_image_input_topic = LaunchConfiguration(
        "compressed_image_input_topic"
    ).perform(context)
    calib_first = LaunchConfiguration("calib_first").perform(context).strip().lower()
    calib_first = calib_first in ("1", "true", "yes", "on")

    bringup_share = get_package_share_directory("radar_bringup")
    localization_share = get_package_share_directory("localization")
    debug_map_share = get_package_share_directory("debug_map")

    bringup_config = os.path.join(bringup_share, "config")
    map_pcd_path = os.path.join(localization_share, "config", "RM2025.pcd")
    camera_extrinsic_yaml = LaunchConfiguration("calib_out_matrix_yaml").perform(context)
    camera_to_lidar_yaml = os.path.join(localization_share, "config", "camera_to_lidar.yaml")
    manual_yaml = os.path.join(localization_share, "config", "livox_rm_manual.yaml")
    rviz_config_path = os.path.join(localization_share, "config", "localization_rviz_config.rviz")

    intra = [{"use_intra_process_comms": True}]
    enemy_uav_coord = (27.131, 2.369, 1.7)
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

    def make_rosbag_player_component(extra_params=None, remappings=None):
        params = {
            "rosbag_file": rosbag_file,
            "lidar_input_topic": lidar_input_topic,
            "image_input_topic": image_input_topic,
            "compressed_image_input_topic": compressed_image_input_topic,
        }
        if extra_params:
            params.update(extra_params)
        return ComposableNode(
            package="rosbag_player",
            plugin="RosbagPlayer",
            name="rosbag_player_node",
            parameters=[params],
            remappings=remappings or [],
            extra_arguments=intra,
        )

    rosbag_player_component = make_rosbag_player_component()

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
                "gicp_total_timeout_sec": 20.0,
                "fallback_to_initial_after_timeout": True,
                "gicp_max_iterations": 100,
                "gicp_transformation_epsilon": 1e-8,
                "gicp_euclidean_fitness_epsilon": 1e-6,
                "gicp_success_fitness_threshold": 0.2,
                "accumulate_time": 30,
                "grid_size_degrees": 0.05,
                "downsample_leaf_size": 0.2,
                "roi_min_x": 5.0,
                "roi_max_x": 30.0,
                "roi_min_y": -10.0,
                "roi_max_y": 8.0,
                "roi_max_z": 1.5,
                "tf_publish_rate_hz": 20.0,
                "gicp_worker_rate_hz": 20.0,
                "publish_full_pcd_map": False,
                "publish_live_scan_map_on_gicp_timeout": False,
            }
        ],
        extra_arguments=[{"use_intra_process_comms": False}],
    )

    dynamic_cloud_component = ComposableNode(
        package="dynamic_cloud",
        plugin="tdt_radar::DynamicCloud",
        name="dynamic_cloud_node",
        parameters=[
            {
                "map_pcd_path": map_pcd_path,
                "uav_other_min_x": 0.0,
                "uav_other_max_x": 28.0,
                "uav_other_min_y": 0.0,
                "uav_other_max_y": 15.0,
                "uav_other_min_z": 1.2,
                "uav_other_max_z": 4.5,
            }
        ],
        remappings=[
            ("/livox/lidar_integrated", "/livox/lidar"),
        ],
        extra_arguments=intra,
    )

    cluster_component = ComposableNode(
        package="cluster",
        plugin="tdt_radar::Cluster",
        name="cluster_node",
        extra_arguments=intra,
    )

    detect_component = ComposableNode(
        package="tdt_vision",
        plugin="tdt_radar::Detect",
        name="radar_detect_node",
        parameters=[{"image_topic": "/camera_image"}],
        extra_arguments=intra,
    )

    resolve_component = ComposableNode(
        package="tdt_vision",
        plugin="tdt_radar::Resolve",
        name="radar_resolve_node",
        parameters=[
            {
                "out_matrix_path": camera_extrinsic_yaml,
                # RosbagPlayer decodes /compressed_image directly and does not
                # rectify it, so resolve must use the raw camera model here.
                "camera_params_path": os.path.join(bringup_config, "camera_params.yaml"),
                "region_points_path": os.path.join(bringup_config, "RM2025_Points.yaml"),
                "minimap_path": os.path.join(bringup_config, "RM2025.png"),
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
                "lost_miss_limit": 8,
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
                "blind_guess_enabled": True,
                "blind_guess_config_path": os.path.join(bringup_config, "guess_pts.yaml"),
                "blind_guess_after_ms": 250,
                "blind_guess_hold_ms": 1200,
                "kf_predict_hold_ms": 150,
                "max_predict_step_m": 0.8,
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
                "map_path": os.path.join(debug_map_share, "config", "RM2025.png"),
                "use_match_info_color": False,
                "self_color_override": team["match_self_color"],
                "use_referee_identity_color": True,
            }
        ],
        extra_arguments=intra,
    )

    replay_container = ComposableNodeContainer(
        name="radar_replay_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            rosbag_player_component,
            localization_component,
            dynamic_cloud_component,
            cluster_component,
            detect_component,
            resolve_component,
            kalman_filter_component,
            kalman_tracks_adapter_component,
            debug_map_component,
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
                    {"my_color": team["decision_my_color"]},
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

    shm_pdu_broadcast_node = Node(
        package="radar_pdu_bridge",
        executable="shm_pdu_broadcast_node",
        name="shm_pdu_broadcast_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_pdu_bridge")),
        parameters=[
            {
                "poll_period_ms": pdu_poll_period_ms,
            }
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
                "camera_params_path": os.path.join(bringup_config, "camera_params.yaml"),
                "region_points_path": os.path.join(bringup_config, "RM2025_Points.yaml"),
                "uav_intersection_debug_topic": "/uav_intersection_debug",
                "uav_cluster_debug_topic": "/uav_cluster_debug",
                "uav_resolve_debug_status_topic": "/uav_resolve_debug_status",
            }
        ],
    )

    normal_flow_actions = [
        LogInfo(msg=f"rosbag_radar_launch our_color={team['decision_my_color']}"),
        LogInfo(msg=f"rosbag file: {rosbag_file}"),
        replay_container,
        shm_pdu_broadcast_node,
        radar_decision_comm_container,
        map_pcd_publisher,
        track_status_window_node,
        rviz_node,
    ]

    if not calib_first:
        return normal_flow_actions

    calib_rosbag_container = ComposableNodeContainer(
        name="radar_calib_rosbag_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            make_rosbag_player_component(
                {
                    "static_image_only": True,
                    "static_image_publish_rate": LaunchConfiguration(
                        "calib_static_image_rate"
                    ),
                },
                remappings=[("camera_image", LaunchConfiguration("calib_image_topic"))],
            )
        ],
        output="both",
        emulate_tty=True,
    )

    pnp_calib_node = Node(
        package="camera_field_calib_tool",
        executable="pnp_calib_node_new",
        name="pnp_calib_node_new",
        output="screen",
        parameters=[
            {
                "image_topic": LaunchConfiguration("calib_image_topic"),
                "intrinsics_yaml": LaunchConfiguration("calib_intrinsics_yaml"),
                "out_matrix_yaml": LaunchConfiguration("calib_out_matrix_yaml"),
                "reference_points_yaml": LaunchConfiguration("calib_reference_points_yaml"),
                "use_refine": LaunchConfiguration("calib_use_refine"),
                "pnp_method": LaunchConfiguration("calib_pnp_method"),
                "min_points": LaunchConfiguration("calib_min_points"),
            }
        ],
    )

    stop_calib_rosbag = EmitEvent(
        event=ShutdownProcess(
            process_matcher=matches_executable("component_container_mt"),
        )
    )

    start_normal_flow_after_calib = RegisterEventHandler(
        OnProcessExit(
            target_action=pnp_calib_node,
            on_exit=[
                LogInfo(msg="PnP calibration exited; stopping static rosbag image publisher."),
                stop_calib_rosbag,
            ],
        )
    )

    start_normal_flow_after_calib_rosbag_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=calib_rosbag_container,
            on_exit=[
                LogInfo(msg="Static calibration rosbag publisher exited."),
                LogInfo(msg="Starting normal rosbag radar flow with the saved calibration."),
                *normal_flow_actions,
            ],
        )
    )

    return [
        LogInfo(msg="calib_first=true: start PnP calibration first. Press P to solve, S to save, then Q/Esc to continue."),
        LogInfo(msg="calibration uses the first /compressed_image frame from the rosbag and keeps it still."),
        LogInfo(msg=f"calibration output: {LaunchConfiguration('calib_out_matrix_yaml').perform(context)}"),
        calib_rosbag_container,
        pnp_calib_node,
        start_normal_flow_after_calib,
        start_normal_flow_after_calib_rosbag_exit,
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
        calib_tool_share, "config", "reference_points_RM_guanggu.yaml"
    )
    default_calib_out_matrix = os.path.join(
        bringup_share, "config", "out_matrix_RM_guanggu.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "our_color",
                default_value="blue",
                description="our team color: red or blue",
            ),
            DeclareLaunchArgument(
                "rosbag_file",
                default_value="/home/user/DATA/RADAR_ROSBAG/radar_match_camera_lidar/radar_match_camera_lidar_0.db3",
                description="input rosbag path for replay",
            ),
            DeclareLaunchArgument(
                "lidar_input_topic",
                default_value="/livox/lidar",
                description="input PointCloud2 topic in the rosbag",
            ),
            DeclareLaunchArgument(
                "image_input_topic",
                default_value="/camera0/image",
                description="input raw sensor_msgs/Image topic in the rosbag",
            ),
            DeclareLaunchArgument(
                "compressed_image_input_topic",
                default_value="/compressed_image",
                description="input sensor_msgs/CompressedImage topic in the rosbag",
            ),
            DeclareLaunchArgument(
                "decision_params_file",
                default_value=default_decision_params,
                description="radar_decision_comm parameter file",
            ),
            DeclareLaunchArgument(
                "referee_rx_mode",
                default_value="topic",
                description="referee_serial_rx mode during replay",
            ),
            DeclareLaunchArgument(
                "referee_tx_mode",
                default_value="topic",
                description="referee_serial_tx mode during replay",
            ),
            DeclareLaunchArgument(
                "referee_serial_device",
                default_value="/dev/ttyUSB0",
                description="referee serial device path for bridge mode",
            ),
            DeclareLaunchArgument(
                "referee_serial_baudrate",
                default_value="115200",
                description="referee serial baudrate for bridge mode",
            ),
            DeclareLaunchArgument(
                "referee_publish_raw_debug",
                default_value="false",
                description="publish raw referee serial RX bytes to /radar/debug/referee_raw_bytes",
            ),
            DeclareLaunchArgument(
                "enable_pdu_bridge",
                default_value="true",
                description="start shared-memory PDU bridge during replay",
            ),
            DeclareLaunchArgument(
                "pdu_poll_period_ms",
                default_value="2",
                description="shared-memory PDU poll period in milliseconds",
            ),
            DeclareLaunchArgument(
                "calib_first",
                default_value="false",
                description="run pnp_calib_node_new before starting the normal replay flow",
            ),
            DeclareLaunchArgument(
                "calib_image_topic",
                default_value="/calib_camera_image",
                description="image topic used by pnp_calib_node_new during static rosbag-image calibration",
            ),
            DeclareLaunchArgument(
                "calib_static_image_rate",
                default_value="5.0",
                description="publish rate for the first rosbag image during calibration",
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
                default_value="5",
                description="minimum selected point count before pnp_calib_node_new solves",
            ),
            OpaqueFunction(function=_build_launch),
        ]
    )
