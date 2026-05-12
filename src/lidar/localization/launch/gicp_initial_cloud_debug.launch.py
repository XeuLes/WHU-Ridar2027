import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def str2bool(value: str) -> bool:
    return value.lower() in ['true', '1', 'yes', 'on']


def generate_launch_description():
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros2_driver')

    default_map_path = os.path.join(localization_share_dir, 'config', 'RM2025.pcd')
    default_manual_yaml = os.path.join(
        localization_share_dir, 'config', 'livox_rm_manual.yaml'
    )
    default_camera_to_lidar_yaml = os.path.join(
        localization_share_dir, 'config', 'camera_to_lidar.yaml'
    )
    default_camera_extrinsic_yaml = (
        '/home/user/code/T-DT_Radar-main/config/out_matrix_guanggu.yaml'
    )
    rviz_config_path = os.path.join(
        localization_share_dir, 'config', 'gicp_initial_cloud_debug.rviz'
    )
    user_config_path = os.path.join(
        livox_share_dir, 'config', 'livox_lidar_config.json'
    )

    declare_use_rosbag = DeclareLaunchArgument('use_rosbag', default_value='false')
    declare_rosbag_file = DeclareLaunchArgument(
        'rosbag_file', default_value='/home/user/DATA/rosbag.db3'
    )
    declare_target_pcd_path = DeclareLaunchArgument(
        'target_pcd_path', default_value=default_map_path
    )
    declare_manual_yaml = DeclareLaunchArgument(
        'manual_yaml', default_value=default_manual_yaml
    )
    declare_initial_pose_mode = DeclareLaunchArgument(
        'initial_pose_mode', default_value='online'
    )
    declare_camera_extrinsic_yaml = DeclareLaunchArgument(
        'camera_extrinsic_yaml', default_value=default_camera_extrinsic_yaml
    )
    declare_camera_to_lidar_yaml = DeclareLaunchArgument(
        'camera_to_lidar_yaml', default_value=default_camera_to_lidar_yaml
    )
    declare_online_initial_y_offset = DeclareLaunchArgument(
        'online_initial_y_offset', default_value='15.0'
    )

    declare_accumulate_time = DeclareLaunchArgument('accumulate_time', default_value='30')
    declare_grid_size_degrees = DeclareLaunchArgument(
        'grid_size_degrees', default_value='0.1'
    )
    declare_downsample_leaf_size = DeclareLaunchArgument(
        'downsample_leaf_size', default_value='0.1'
    )

    declare_roi_min_x = DeclareLaunchArgument('roi_min_x', default_value='5.0')
    declare_roi_max_x = DeclareLaunchArgument('roi_max_x', default_value='29.0')
    declare_roi_min_y = DeclareLaunchArgument('roi_min_y', default_value='-6.5')
    declare_roi_max_y = DeclareLaunchArgument('roi_max_y', default_value='8.5')
    declare_roi_max_z = DeclareLaunchArgument('roi_max_z', default_value='1.5')

    declare_gicp_max_iterations = DeclareLaunchArgument(
        'gicp_max_iterations', default_value='100'
    )
    declare_gicp_transformation_epsilon = DeclareLaunchArgument(
        'gicp_transformation_epsilon', default_value='1e-8'
    )
    declare_gicp_euclidean_fitness_epsilon = DeclareLaunchArgument(
        'gicp_euclidean_fitness_epsilon', default_value='1e-6'
    )
    declare_gicp_worker_rate_hz = DeclareLaunchArgument(
        'gicp_worker_rate_hz', default_value='1.0'
    )
    declare_lock_to_initial = DeclareLaunchArgument(
        'lock_to_initial',
        default_value='false',
        description='Keep /filter_map_rm using the initial transform instead of accepting GICP.',
    )
    declare_gicp_success_fitness_threshold = DeclareLaunchArgument(
        'gicp_success_fitness_threshold', default_value='0.12'
    )
    declare_enable_gicp_total_timeout = DeclareLaunchArgument(
        'enable_gicp_total_timeout', default_value='true'
    )
    declare_gicp_total_timeout_sec = DeclareLaunchArgument(
        'gicp_total_timeout_sec', default_value='30.0'
    )
    declare_tf_publish_rate_hz = DeclareLaunchArgument(
        'tf_publish_rate_hz', default_value='20.0'
    )
    declare_start_rviz = DeclareLaunchArgument('start_rviz', default_value='true')

    declare_xfer_format = DeclareLaunchArgument('xfer_format', default_value='0')
    declare_multi_topic = DeclareLaunchArgument('multi_topic', default_value='0')
    declare_data_src = DeclareLaunchArgument('data_src', default_value='0')
    declare_publish_freq = DeclareLaunchArgument('publish_freq', default_value='10.0')
    declare_output_type = DeclareLaunchArgument('output_data_type', default_value='0')
    declare_frame_id = DeclareLaunchArgument('frame_id', default_value='livox_frame')
    declare_lvx_file_path = DeclareLaunchArgument('lvx_file_path', default_value='')
    declare_cmdline_bd_code = DeclareLaunchArgument(
        'cmdline_input_bd_code', default_value=''
    )

    def get_localization_node(context):
        use_rosbag = str2bool(LaunchConfiguration('use_rosbag').perform(context))
        lock_to_initial = str2bool(
            LaunchConfiguration('lock_to_initial').perform(context)
        )
        success_threshold = (
            -1.0 if lock_to_initial
            else float(
                LaunchConfiguration('gicp_success_fitness_threshold').perform(context)
            )
        )

        params = {
            'use_sim_time': use_rosbag,
            'target_pcd_path': LaunchConfiguration('target_pcd_path').perform(context),
            'initial_pose_mode': LaunchConfiguration('initial_pose_mode').perform(context),
            'camera_extrinsic_yaml': LaunchConfiguration(
                'camera_extrinsic_yaml'
            ).perform(context),
            'camera_to_lidar_yaml': LaunchConfiguration(
                'camera_to_lidar_yaml'
            ).perform(context),
            'online_initial_y_offset': float(
                LaunchConfiguration('online_initial_y_offset').perform(context)
            ),
            'publish_full_pcd_map': False,
            'publish_live_scan_map_on_gicp_timeout': False,
            'enable_gicp_total_timeout': str2bool(
                LaunchConfiguration('enable_gicp_total_timeout').perform(context)
            ),
            'gicp_total_timeout_sec': float(
                LaunchConfiguration('gicp_total_timeout_sec').perform(context)
            ),
            'fallback_to_initial_after_timeout': True,
            'gicp_max_iterations': int(
                LaunchConfiguration('gicp_max_iterations').perform(context)
            ),
            'gicp_transformation_epsilon': float(
                LaunchConfiguration('gicp_transformation_epsilon').perform(context)
            ),
            'gicp_euclidean_fitness_epsilon': float(
                LaunchConfiguration('gicp_euclidean_fitness_epsilon').perform(context)
            ),
            'gicp_success_fitness_threshold': success_threshold,
            'accumulate_time': int(
                LaunchConfiguration('accumulate_time').perform(context)
            ),
            'grid_size_degrees': float(
                LaunchConfiguration('grid_size_degrees').perform(context)
            ),
            'downsample_leaf_size': float(
                LaunchConfiguration('downsample_leaf_size').perform(context)
            ),
            'roi_min_x': float(LaunchConfiguration('roi_min_x').perform(context)),
            'roi_max_x': float(LaunchConfiguration('roi_max_x').perform(context)),
            'roi_min_y': float(LaunchConfiguration('roi_min_y').perform(context)),
            'roi_max_y': float(LaunchConfiguration('roi_max_y').perform(context)),
            'roi_max_z': float(LaunchConfiguration('roi_max_z').perform(context)),
            'tf_publish_rate_hz': float(
                LaunchConfiguration('tf_publish_rate_hz').perform(context)
            ),
            'gicp_worker_rate_hz': float(
                LaunchConfiguration('gicp_worker_rate_hz').perform(context)
            ),
        }

        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[LaunchConfiguration('manual_yaml').perform(context), params],
            extra_arguments=[{'use_intra_process_comms': False}],
        )

    livox_driver_node = Node(
        package='livox_ros2_driver',
        executable='livox_ros2_driver_node',
        output='screen',
        parameters=[{
            'xfer_format': LaunchConfiguration('xfer_format'),
            'multi_topic': LaunchConfiguration('multi_topic'),
            'data_src': LaunchConfiguration('data_src'),
            'publish_freq': LaunchConfiguration('publish_freq'),
            'output_data_type': LaunchConfiguration('output_data_type'),
            'frame_id': LaunchConfiguration('frame_id'),
            'lvx_file_path': LaunchConfiguration('lvx_file_path'),
            'user_config_path': user_config_path,
            'cmdline_input_bd_code': LaunchConfiguration('cmdline_input_bd_code'),
        }],
    )

    def get_rosbag_player(context):
        return ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                LaunchConfiguration('rosbag_file').perform(context),
                '--loop', '--clock',
            ],
            output='screen',
            emulate_tty=True,
            on_exit=Shutdown(),
        )

    def create_nodes(context, *args, **kwargs):
        use_rosbag = str2bool(LaunchConfiguration('use_rosbag').perform(context))
        start_rviz = str2bool(LaunchConfiguration('start_rviz').perform(context))
        lock_to_initial = str2bool(
            LaunchConfiguration('lock_to_initial').perform(context)
        )

        nodes = [
            LogInfo(msg='GICP initial cloud debug launch starting...'),
            LogInfo(msg='target: /livox/map, registered scan source: /filter_map_rm'),
            LogInfo(msg=f'lock_to_initial: {lock_to_initial}'),
        ]

        container = ComposableNodeContainer(
            name='gicp_initial_cloud_debug_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[get_localization_node(context)],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown(),
        )
        nodes.append(container)

        if use_rosbag:
            nodes.append(get_rosbag_player(context))
        else:
            nodes.append(livox_driver_node)

        if start_rviz:
            nodes.append(
                ExecuteProcess(
                    cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
                    output='screen',
                )
            )

        return nodes

    return LaunchDescription([
        declare_use_rosbag,
        declare_rosbag_file,
        declare_target_pcd_path,
        declare_manual_yaml,
        declare_initial_pose_mode,
        declare_camera_extrinsic_yaml,
        declare_camera_to_lidar_yaml,
        declare_online_initial_y_offset,
        declare_accumulate_time,
        declare_grid_size_degrees,
        declare_downsample_leaf_size,
        declare_roi_min_x,
        declare_roi_max_x,
        declare_roi_min_y,
        declare_roi_max_y,
        declare_roi_max_z,
        declare_gicp_max_iterations,
        declare_gicp_transformation_epsilon,
        declare_gicp_euclidean_fitness_epsilon,
        declare_gicp_worker_rate_hz,
        declare_lock_to_initial,
        declare_gicp_success_fitness_threshold,
        declare_enable_gicp_total_timeout,
        declare_gicp_total_timeout_sec,
        declare_tf_publish_rate_hz,
        declare_start_rviz,
        declare_xfer_format,
        declare_multi_topic,
        declare_data_src,
        declare_publish_freq,
        declare_output_type,
        declare_frame_id,
        declare_lvx_file_path,
        declare_cmdline_bd_code,
        OpaqueFunction(function=create_nodes),
    ])
