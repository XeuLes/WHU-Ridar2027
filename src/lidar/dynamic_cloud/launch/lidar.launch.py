import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # -------------------------- 1. 路径与包获取 --------------------------
    localization_share_dir = get_package_share_directory('localization')
    # 统一使用第一个文件中的驱动包名
    livox_package = 'livox_ros2_driver' 
    livox_share_dir = get_package_share_directory(livox_package)

    # -------------------------- 2. Launch 参数定义 --------------------------
    declare_use_rosbag = DeclareLaunchArgument(
        'use_rosbag', default_value='false',
        description='是否使用rosbag（true=rosbag模式，false=实时雷达模式）'
    )
    declare_rosbag_file = DeclareLaunchArgument(
        'rosbag_file', default_value='/home/user/DATA/rosbag.db3',
        description='rosbag文件路径'
    )
    
    # 雷达驱动参数定义（保持与 debug 脚本一致）
    declare_xfer_format = DeclareLaunchArgument('xfer_format', default_value='0')
    declare_multi_topic = DeclareLaunchArgument('multi_topic', default_value='0')
    declare_data_src = DeclareLaunchArgument('data_src', default_value='0')
    declare_publish_freq = DeclareLaunchArgument('publish_freq', default_value='10.0')
    declare_output_type = DeclareLaunchArgument('output_data_type', default_value='0')
    declare_frame_id = DeclareLaunchArgument('frame_id', default_value='livox_frame')
    declare_user_config_path = DeclareLaunchArgument(
        'user_config_path', 
        default_value=os.path.join(livox_share_dir, 'config', 'livox_lidar_config.json')
    )

    # -------------------------- 3. 算法节点定义 (Composable) --------------------------
    def get_localization_node(use_rosbag: bool):
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                'target_pcd_path': '/home/user/code/T-DT_Radar-main/src/lidar/localization/config/基础制造.pcd',
                'use_rosbag': use_rosbag,
                'rosbag_file_path': LaunchConfiguration('rosbag_file'),
                'manual_mode': False, # 初始设为手动模式
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )

    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
            parameters=[{'input_cloud_topic': '/livox/lidar'}],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_cluster_node():
        return ComposableNode(
            package='cluster',
            plugin='tdt_radar::Cluster',
            name='cluster_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_kalman_filter_node():
        return ComposableNode(
            package='kalman_filter',
            plugin='tdt_radar::KalmanFilter',
            name='kalman_filter_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # -------------------------- 4. 驱动与回放节点定义 --------------------------
    livox_driver_node = Node(
        package=livox_package,
        executable='livox_ros2_driver_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            'xfer_format': LaunchConfiguration('xfer_format'),
            'multi_topic': LaunchConfiguration('multi_topic'),
            'data_src': LaunchConfiguration('data_src'),
            'publish_freq': LaunchConfiguration('publish_freq'),
            'output_data_type': LaunchConfiguration('output_data_type'),
            'frame_id': LaunchConfiguration('frame_id'),
            'user_config_path': LaunchConfiguration('user_config_path'),
        }]
    )

    def get_rosbag_player():
        return ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('rosbag_file')],
            output='screen', on_exit=Shutdown()
        )

    # -------------------------- 5. 动态逻辑选择 (OpaqueFunction) --------------------------
    def create_nodes(context, *args, **kwargs):
        use_rosbag = LaunchConfiguration('use_rosbag').perform(context).lower() == 'true'
        rviz_config_path = os.path.join(localization_share_dir, 'config', 'localization_rviz_config.rviz')
        
        nodes = []
        nodes.append(LogInfo(msg=f"===== 运行模式：{'ROSBAG' if use_rosbag else '实时雷达'} ====="))

        # 算法容器
        container = ComposableNodeContainer(
            name='lidar_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                get_localization_node(use_rosbag),
                get_dynamic_cloud_node(),
                get_cluster_node(),
                get_kalman_filter_node()
            ],
            output='both',
            on_exit=Shutdown()
        )
        
        # 模式逻辑添加
        if use_rosbag:
            nodes.append(get_rosbag_player())
        else:
            nodes.append(livox_driver_node)

        # 延迟启动容器（等待驱动初始化或播放器就绪）
        nodes.append(TimerAction(period=1.5, actions=[container]))

        # 启动 RViz
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
            output='screen', on_exit=Shutdown()
        ))
        
        return nodes

    # -------------------------- 6. 返回 LaunchDescription --------------------------
    return LaunchDescription([
        declare_use_rosbag,
        declare_rosbag_file,
        declare_xfer_format,
        declare_multi_topic,
        declare_data_src,
        declare_publish_freq,
        declare_output_type,
        declare_frame_id,
        declare_user_config_path,
        OpaqueFunction(function=create_nodes),
    ])