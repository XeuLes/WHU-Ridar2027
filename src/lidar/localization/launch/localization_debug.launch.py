import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    localization_share_dir = get_package_share_directory('localization')
    livox_share_dir = get_package_share_directory('livox_ros2_driver')

    # -------------------------- Args --------------------------
    declare_use_rosbag = DeclareLaunchArgument(
        'use_rosbag', default_value='false', description='rosbag mode'
    )
    declare_rosbag_file = DeclareLaunchArgument(
        'rosbag_file', default_value='/home/user/DATA/rosbag.db3'
    )
    
    # 路径参数
    declare_map_path = DeclareLaunchArgument(
        'target_pcd_path', 
        default_value='/home/user/code/T-DT_Radar-main/src/lidar/localization/config/基础制造_rm_frame.pcd'
    )
    # 手动参数文件
    declare_manual_yaml = DeclareLaunchArgument(
        'manual_yaml',
        default_value='/home/user/code/T-DT_Radar-main/src/lidar/localization/config/livox_rm_manual.yaml'
    )

    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'livox_lidar_config.json')
    rviz_config_path = os.path.join(localization_share_dir, 'config', 'localization_rviz_config.rviz')

    # Livox args
    declare_xfer_format = DeclareLaunchArgument('xfer_format', default_value='0')
    declare_multi_topic = DeclareLaunchArgument('multi_topic', default_value='0')
    declare_data_src = DeclareLaunchArgument('data_src', default_value='0')
    declare_publish_freq = DeclareLaunchArgument('publish_freq', default_value='10.0')
    declare_output_type = DeclareLaunchArgument('output_data_type', default_value='0')
    declare_frame_id = DeclareLaunchArgument('frame_id', default_value='livox_frame')
    declare_lvx_file_path = DeclareLaunchArgument('lvx_file_path', default_value='')
    declare_cmdline_bd_code = DeclareLaunchArgument('cmdline_input_bd_code', default_value='')

    # -------------------------- Node Definition --------------------------
    def get_localization_node(context):
        pcd_path = LaunchConfiguration('target_pcd_path').perform(context)
        yaml_path = LaunchConfiguration('manual_yaml').perform(context)

        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[
                # 加载 YAML 参数
                yaml_path,
                # 覆盖地图路径
                {'target_pcd_path': pcd_path}
            ],
            extra_arguments=[{'use_intra_process_comms': True}],
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
        }]
    )

    def get_rosbag_player(context):
        bag_path = LaunchConfiguration('rosbag_file').perform(context)
        return ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path, '--loop'],
            output='screen',
            emulate_tty=True,
            on_exit=Shutdown()
        )

    def create_nodes(context, *args, **kwargs):
        use_rosbag = LaunchConfiguration('use_rosbag').perform(context).lower() == 'true'
        manual_yaml = LaunchConfiguration('manual_yaml').perform(context)
        
        nodes = []
        nodes.append(LogInfo(msg=f"===== 自动配准模式启动 (基于初值: {manual_yaml}) ====="))

        container = ComposableNodeContainer(
            name='localization_debug_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[get_localization_node(context)],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown()
        )
        nodes.append(container)

        if use_rosbag:
            nodes.append(get_rosbag_player(context))
        else:
            nodes.append(livox_driver_node)

        nodes.append(ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
            output='screen'
        ))
        return nodes

    return LaunchDescription([
        declare_use_rosbag,
        declare_rosbag_file,
        declare_map_path,
        declare_manual_yaml,
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