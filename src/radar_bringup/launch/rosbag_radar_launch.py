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
    rosbag_file = LaunchConfiguration("rosbag_file").perform(context)

    bringup_share = get_package_share_directory("radar_bringup")
    localization_share = get_package_share_directory("localization")
    debug_map_share = get_package_share_directory("debug_map")

    bringup_config = os.path.join(bringup_share, "config")
    map_pcd_path = os.path.join(localization_share, "config", "RM2025.pcd")
    rviz_config_path = os.path.join(localization_share, "config", "localization_rviz_config.rviz")

    intra = [{"use_intra_process_comms": True}]

    rosbag_player_component = ComposableNode(
        package="rosbag_player",
        plugin="RosbagPlayer",
        name="rosbag_player_node",
        parameters=[{"rosbag_file": rosbag_file}],
        extra_arguments=intra,
    )

    localization_component = ComposableNode(
        package="localization",
        plugin="tdt_radar::Localization",
        name="localization_node",
        parameters=[
            {
                "use_map": True,
                "map_frame": "rm_frame",
                "target_pcd_path": map_pcd_path,
            }
        ],
        extra_arguments=intra,
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
                "out_matrix_path": os.path.join(bringup_config, "out_matrix.yaml"),
                "camera_params_path": os.path.join(bringup_config, "camera_params_rect.yaml"),
                "region_points_path": os.path.join(bringup_config, "RM2025_Points.yaml"),
                "minimap_path": os.path.join(bringup_config, "RM2025.png"),
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
                "use_referee_identity_color": False,
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

    return [
        LogInfo(msg=f"rosbag_radar_launch our_color={team['decision_my_color']}"),
        LogInfo(msg=f"rosbag file: {rosbag_file}"),
        replay_container,
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
                "rosbag_file",
                default_value="/home/user/DATA/rosbag.db3",
                description="input rosbag path for replay",
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
            OpaqueFunction(function=_build_launch),
        ]
    )
