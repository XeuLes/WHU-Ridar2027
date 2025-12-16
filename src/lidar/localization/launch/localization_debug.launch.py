import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取功能包路径
    localization_package = 'localization'
    localization_share_dir = get_package_share_directory(localization_package)
    livox_share_dir = get_package_share_directory('livox_ros_driver2')
    
    # -------------------------- 1. 声明模式切换参数 --------------------------
    declare_use_rosbag = DeclareLaunchArgument(
        'use_rosbag',
        default_value='true',  # 默认实时雷达模式
        description='是否使用rosbag（true=rosbag模式，false=实时雷达模式）'
    )
    declare_rosbag_file = DeclareLaunchArgument(
        'rosbag_file',
        default_value="/home/user/DATA/rosbag.db3",
        description="rosbag文件路径（仅rosbag模式生效）"
    )
    
    # -------------------------- 1. 配置文件路径 --------------------------
    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'MID360_config.json')
    if not os.path.exists(user_config_path):
        raise FileNotFoundError(f"Livox 配置文件不存在：{user_config_path}")
    
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'localization_rviz_config.rviz'
    )
    
    # -------------------------- 2. Localization 节点（新增use_rosbag参数） --------------------------
    def get_localization_node(use_rosbag):
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                'map_path': '/home/user/CODE/T-DT_Radar-main/src/lidar/localization/config/aligned.pcd',
                'use_rosbag': use_rosbag,  # 新增：告诉localization节点当前模式
                'rosbag_file_path': LaunchConfiguration('rosbag_file'),  # 传递rosbag路径
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    # -------------------------- 3. Livox 驱动节点（无修改） --------------------------
    livox_driver_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            "xfer_format": 0,
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "frame_id": 'livox_frame',
            "user_config_path": user_config_path,
            "cmdline_input_bd_code": 'livox0000000001',
            "enable_imu": False,
            "enable_gps": False
        }]
    )
    
    # -------------------------- 4. 替换原有rosbag_player_node为官方播放器 --------------------------
    def get_rosbag_player():
        return ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                LaunchConfiguration('rosbag_file'),
            ],
            output='screen',
            emulate_tty=True,
            on_exit=Shutdown()
        )
    
    # -------------------------- 5. 动态选择加载的节点 --------------------------
    def create_nodes(context):
        use_rosbag = LaunchConfiguration('use_rosbag').perform(context).lower() == 'true'
        nodes = []
        
        # 模式提示日志
        nodes.append(LogInfo(msg=f"===== 当前模式：{'ROSBAG模式' if use_rosbag else '实时雷达模式'} ====="))
        
        # 组件容器：根据模式决定是否加载rosbag_player
        container = ComposableNodeContainer(
            name='localization_debug_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                get_localization_node(use_rosbag),  # 传递模式参数
            ],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown()
        )
        nodes.append(container)
        
        # 根据模式加载专属节点
        if use_rosbag:
            nodes.append(get_rosbag_player())  # rosbag模式：加载官方播放器
        else:
            nodes.append(livox_driver_node)    # 实时模式：加载雷达驱动
        
        # 共用RViz节点
        nodes.append(LogInfo(msg=f"RViz2 config path: {rviz_config_path}"))
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
            output='screen',
            on_exit=Shutdown()
        ))
        
        return nodes
    
    # -------------------------- 组装所有节点 --------------------------
    return LaunchDescription([
        declare_use_rosbag,  # 新增：参数声明
        declare_rosbag_file, # 新增：参数声明
        OpaqueFunction(function=create_nodes)  # 新增：动态创建节点
    ])
