import os
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node  # 新增 Node 导入（关键）
from launch import LaunchDescription
from launch.actions import Shutdown, LogInfo, ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # -------------------------- 新增：获取 Livox 驱动包路径 --------------------------
    livox_package = 'livox_ros_driver2'
    try:
        livox_share_dir = get_package_share_directory(livox_package)
    except:
        raise FileNotFoundError(f"未找到 livox_ros_driver2 包，请先安装！")
    
    # -------------------------- 原有配置保留 --------------------------
    # 获取localization包的路径
    localization_share_dir = get_package_share_directory('localization')
    # 地图点云文件路径
    map_pcd_path = os.path.join(localization_share_dir, 'config', 'aligned.pcd')

    # 定义各个节点（原有逻辑不变）
    def get_localization_node():
        return ComposableNode(
            package='localization',
            plugin='tdt_radar::Localization',
            name='localization_node',
            parameters=[{
                'use_map': True,
                'map_frame': 'livox_frame',
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_dynamic_cloud_node():
        return ComposableNode(
            package='dynamic_cloud',
            plugin='tdt_radar::DynamicCloud',
            name='dynamic_cloud_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_rosbag_player_node():
        return ComposableNode(
            package='rosbag_player',
            plugin='RosbagPlayer',
            name='rosbag_player_node',
            parameters=[{
                'rosbag_file': "/home/kakaka/code/rosbag.db3"
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # （可选）地图点云发布节点（若之前报错，可注释掉，因为localization已加载地图）
    def get_map_pcd_publisher():
        return ComposableNode(
            package='pcl_ros',
            plugin='pcl_ros::PCDToPointCloud',
            name='map_pcd_publisher',
            parameters=[{
                'pcd_file': map_pcd_path,
                'frame_id': 'map',
                'publish_rate': 0.1,
                'repeat_delay': -1
            }],
            remappings=[('cloud', '/map_point_cloud')],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # -------------------------- 新增：Livox 驱动节点配置（核心） --------------------------
    livox_driver_node = Node(
        package=livox_package,  # 驱动包名（固定）
        executable='livox_ros_driver2_node',  # 驱动可执行文件名（固定）
        name='livox_lidar_driver',  # 自定义节点名
        output='screen',  # 日志输出到终端
        parameters=[{
            "xfer_format": 0,  # 数据格式：0=PointCloud2（与你的节点兼容）
            "multi_topic": 0,  # 单话题发布所有点云
            "data_src": 0,  # 数据来源：0=实时雷达（1=PCAP文件）
            "publish_freq": 10.0,  # 发布频率（10Hz，匹配MID360硬件）
            "output_data_type": 0,  # 输出类型：0=点云+强度
            "frame_id": 'livox_frame',  # 点云坐标系（必须与localization一致！）
            "user_config_path": os.path.join(livox_share_dir, 'config', 'MID360_config.json'),  # 雷达配置文件路径
            "cmdline_input_bd_code": 'livox0000000001',  # 雷达绑定码（默认，多雷达需修改）
            "enable_imu": False,  # 关闭IMU（无需则关闭）
            "enable_gps": False   # 关闭GPS（无需则关闭）
        }]
    )

    # 创建节点容器（原有逻辑不变）
    container = ComposableNodeContainer(
        name='combined_debug_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            get_localization_node(),
            get_dynamic_cloud_node(),
            # get_rosbag_player_node(),  # 用实时雷达时，注释rosbag播放
            # get_map_pcd_publisher()  # 若之前加载报错，直接注释
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown()
    )

    # RViz2配置（原有逻辑不变）
    rviz_config_package = 'localization'
    rviz_package_share_dir = get_package_share_directory(rviz_config_package)
    rviz_config_path = os.path.join(rviz_package_share_dir, 'rviz', 'map_visualization.rviz')

    log_rviz_path = LogInfo(msg=f"RViz2 config path: {rviz_config_path}")
    log_map_path = LogInfo(msg=f"Map PCD file path: {map_pcd_path}")
    log_livox_config = LogInfo(msg=f"Livox config path: {os.path.join(livox_share_dir, 'config', 'MID360_config.json')}")  # 新增驱动配置日志

    # 启动RViz2（原有逻辑不变）
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()
    )

    # -------------------------- 新增：将驱动节点加入启动列表 --------------------------
    return LaunchDescription([
        log_rviz_path,
        log_map_path,
        log_livox_config,  # 新增日志
        livox_driver_node,  # 启动Livox驱动（先启动，确保点云先发布）
        container,
        rviz_node
    ])
