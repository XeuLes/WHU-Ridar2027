import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node  # 新增 Node 导入
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取功能包路径
    localization_package = 'localization'
    localization_share_dir = get_package_share_directory(localization_package)
    livox_share_dir = get_package_share_directory('livox_ros_driver2')
    
    # -------------------------- 1. 配置文件路径 --------------------------
    # Livox MID360 配置文件路径
    cur_config_path = os.path.join(livox_share_dir, 'config')
    user_config_path = os.path.join(cur_config_path, 'MID360_config.json')
    # 验证配置文件是否存在（可选，避免路径错误）
    if not os.path.exists(user_config_path):
        raise FileNotFoundError(f"Livox 配置文件不存在：{user_config_path}")
    
    # RViz 配置文件路径
    rviz_config_path = os.path.join(
        localization_share_dir,
        'config',
        'localization_rviz_config.rviz'
    )
    
    # -------------------------- 2. Localization 节点（组件化节点） --------------------------
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',  # 组件化节点用 plugin 指定类
            name='localization_node',
            parameters=[{
                # 地图绝对路径
                'map_path': '/home/kakaka/code/WHU-Radar-2026/src/lidar/localization/config/aligned.pcd',
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    # -------------------------- 3. Livox 驱动节点（独立节点，用 Node 类） --------------------------
    livox_driver_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',  # 独立节点用 executable 指定可执行文件
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            "xfer_format": 0,
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "frame_id": 'livox_frame',  # 点云坐标系关键配置
            "user_config_path": user_config_path,
            "cmdline_input_bd_code": 'livox0000000001',
            "enable_imu": False,
            "enable_gps": False
        }]
    )
    
    # -------------------------- 4. 组件容器（仅加载 Localization 组件） --------------------------
    container = ComposableNodeContainer(
        name='localization_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node()  # 只放组件化节点
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )
    
    # -------------------------- 5. 辅助节点（RViz） --------------------------
    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )
    
    # -------------------------- 6. 组装所有节点 --------------------------
    return LaunchDescription([
        log_rviz_path,
        livox_driver_node,  # 独立节点直接加入 LaunchDescription
        container,          # 组件容器节点
        rviz_node
    ])
