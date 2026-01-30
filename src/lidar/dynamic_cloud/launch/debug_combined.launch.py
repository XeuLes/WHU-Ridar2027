import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, Shutdown, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # -------------------------- 1. 路径与包配置 --------------------------
    localization_share_dir = get_package_share_directory('localization')
    livox_package = 'livox_ros2_driver' 
    livox_share_dir = get_package_share_directory(livox_package)

    # -------------------------- 2. 声明模式与驱动参数 --------------------------
    declare_use_rosbag = DeclareLaunchArgument(
        'use_rosbag', default_value='false',
        description='是否使用rosbag（true=rosbag模式，false=实时雷达模式）'
    )
    declare_rosbag_file = DeclareLaunchArgument(
        'rosbag_file', default_value='/home/user/DATA/rosbag.db3',
        description='rosbag文件路径'
    )
    
    # [新增] 地图路径参数
    declare_map_path = DeclareLaunchArgument(
        'target_pcd_path', 
        default_value='/home/user/code/T-DT_Radar-main/src/lidar/localization/config/基础制造_rm_frame.pcd',
        description='目标点云地图路径'
    )
    
    # [新增] 手动配准参数文件
    declare_manual_yaml = DeclareLaunchArgument(
        'manual_yaml',
        default_value='/home/user/code/T-DT_Radar-main/src/lidar/localization/config/livox_rm_manual.yaml',
        description='手动配准生成的参数文件路径'
    )

    # 雷达驱动参数
    declare_xfer_format = DeclareLaunchArgument('xfer_format', default_value='0') # PointCloud2
    declare_multi_topic = DeclareLaunchArgument('multi_topic', default_value='0')
    declare_publish_freq = DeclareLaunchArgument('publish_freq', default_value='10.0')
    declare_frame_id = DeclareLaunchArgument('frame_id', default_value='livox_frame')

    # -------------------------- 3. 节点定义逻辑 --------------------------
    def get_localization_node(context, use_rosbag: bool):
        # 从上下文解析路径
        pcd_path = LaunchConfiguration('target_pcd_path').perform(context)
        yaml_path = LaunchConfiguration('manual_yaml').perform(context)

        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[
                # 1. 加载 YAML 参数文件 (初始位姿)
                yaml_path,
                # 2. 其他覆盖参数
                {
                    'target_pcd_path': pcd_path,
                    'use_rosbag': use_rosbag,
                    'rosbag_file_path': LaunchConfiguration('rosbag_file'),
                }
            ],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_dynamic_cloud_node(context):
        # 也可以把地图路径传给 dynamic_cloud，保持一致
        pcd_path = LaunchConfiguration('target_pcd_path').perform(context)
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
            parameters=[
                {'map_pcd_path': pcd_path}
            ],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # 实时雷达驱动节点
    livox_driver_node = Node(
        package=livox_package,
        executable='livox_ros2_driver_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            'xfer_format': LaunchConfiguration('xfer_format'),
            'multi_topic': LaunchConfiguration('multi_topic'),
            'publish_freq': LaunchConfiguration('publish_freq'),
            'frame_id': LaunchConfiguration('frame_id'),
            'user_config_path': os.path.join(livox_share_dir, 'config', 'livox_lidar_config.json'),
        }]
    )

    # -------------------------- 4. 动态组合逻辑 (OpaqueFunction) --------------------------
    def create_nodes(context, *args, **kwargs):
        use_rosbag = LaunchConfiguration('use_rosbag').perform(context).lower() == 'true'
        manual_yaml = LaunchConfiguration('manual_yaml').perform(context)
        rviz_config_path = os.path.join(localization_share_dir, 'rviz', 'map_visualization.rviz')
        
        nodes = []
        nodes.append(LogInfo(msg=f"===== 联合调试模式 ====="))
        nodes.append(LogInfo(msg=f"数据源: {'ROSBAG' if use_rosbag else '实时雷达'}"))
        nodes.append(LogInfo(msg=f"加载初始位姿: {manual_yaml}"))

        # 核心节点容器
        container = ComposableNodeContainer(
            name='combined_debug_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                get_localization_node(context, use_rosbag),
                get_dynamic_cloud_node(context),
            ],
            output='both',
            on_exit=Shutdown()
        )
        
        # 根据模式选择数据源
        if use_rosbag:
            nodes.append(ExecuteProcess(
                cmd=['ros2', 'bag', 'play', LaunchConfiguration('rosbag_file'), '--loop'], # 建议加上 --loop
                output='screen', on_exit=Shutdown()
            ))
        else:
            nodes.append(livox_driver_node)

        # 容器延迟启动，确保数据源先就绪
        nodes.append(TimerAction(period=1.5, actions=[container]))

        # 启动 RViz
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
            output='screen', on_exit=Shutdown()
        ))
        
        return nodes

    # -------------------------- 5. 返回启动描述 --------------------------
    return LaunchDescription([
        declare_use_rosbag,
        declare_rosbag_file,
        declare_map_path,
        declare_manual_yaml,
        declare_xfer_format,
        declare_multi_topic,
        declare_publish_freq,
        declare_frame_id,
        OpaqueFunction(function=create_nodes),
    ])