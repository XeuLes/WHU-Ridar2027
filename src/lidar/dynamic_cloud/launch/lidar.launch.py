import os
import sys
import yaml
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node  # 新增 Node 导入
from launch.actions import TimerAction, Shutdown, LogInfo, ExecuteProcess  # 新增必要动作
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch
from ament_index_python.packages import get_package_share_directory  # 新增包路径获取

def dump_params(param_file_path, node_name):
    with open(param_file_path, 'r') as file:
        return [yaml.safe_load(file)[node_name]['ros__parameters']]

def generate_launch_description():
    
    # -------------------------- 新增：获取 Livox 驱动包路径 --------------------------
    livox_package = 'livox_ros_driver2'
    try:
        livox_share_dir = get_package_share_directory(livox_package)
    except:
        raise FileNotFoundError(f"未找到 livox_ros_driver2 包，请先安装！")
    
    # -------------------------- 原有节点定义（补充参数确保与Livox兼容） --------------------------
    def get_localization_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='localization_node',
            parameters=[{
                'use_map': True,
                'map_frame': 'livox_frame',  # 与Livox驱动坐标系一致
                'input_topic': '/livox/lidar'  # 订阅Livox点云话题（默认话题）
            }],
            extra_arguments=[{'use_intra_process_comms': True},
                             {'use_multi_threaded_executor': True}],
        )
        
    def get_dynamic_cloud_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='dynamic_cloud_node',
            parameters=[{
                'input_cloud_topic': '/livox/lidar'  # 订阅Livox点云（根据实际需求调整）
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_cluster_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='cluster_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_kalman_filter_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='kalman_filter_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_container(*nodes):
        print(f"当前工作目录: {os.getcwd()}")
        return ComposableNodeContainer(
            name='lidar_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=list(nodes),
            output='both',
            emulate_tty=True,
            on_exit=Shutdown(),
        )
        
    # 初始化原有功能节点
    localization_node = get_localization_node('localization', 'tdt_radar::Localization')
    dynamic_cloud_node = get_dynamic_cloud_node('dynamic_cloud', 'tdt_radar::DynamicCloud')
    cluster_node = get_cluster_node('cluster', 'tdt_radar::Cluster')
    kalman_filter_node = get_kalman_filter_node('kalman_filter', 'tdt_radar::KalmanFilter')

    # 创建节点容器
    lidar_detector = get_container(
        localization_node,
        dynamic_cloud_node,
        cluster_node,
        kalman_filter_node
    )
    
    # -------------------------- 新增：Livox 驱动节点配置（核心） --------------------------
    livox_driver_node = Node(
        package=livox_package,  # 固定驱动包名
        executable='livox_ros_driver2_node',  # 固定可执行文件名
        name='livox_lidar_driver',  # 自定义节点名
        output='screen',  # 日志输出到终端
        parameters=[{
            "xfer_format": 0,  # 0=PointCloud2格式（与你的算法节点兼容）
            "multi_topic": 0,  # 单话题发布所有点云
            "data_src": 0,  # 0=实时雷达数据（1=PCAP回放）
            "publish_freq": 10.0,  # 发布频率（MID360默认10Hz，可根据硬件调整）
            "output_data_type": 0,  # 0=点云+强度（基础数据）
            "frame_id": 'livox_frame',  # 坐标系名称（必须与算法节点一致）
            "user_config_path": os.path.join(livox_share_dir, 'config', 'MID360_config.json'),  # 雷达配置文件
            "cmdline_input_bd_code": 'livox0000000001',  # 雷达绑定码（多雷达时需修改为实际绑定码）
            "enable_imu": False,  # 无需IMU则关闭
            "enable_gps": False,  # 无需GPS则关闭
            "point_cloud_topic": "/livox/lidar"  # 点云发布话题（与算法节点订阅话题一致）
        }]
    )

    # -------------------------- 新增：RViz2 可视化配置 --------------------------
    # 获取RViz配置文件路径（假设配置文件在localization包的rviz目录下）
    rviz_config_package = 'localization'
    try:
        rviz_package_share_dir = get_package_share_directory(rviz_config_package)
    except:
        raise FileNotFoundError(f"未找到 {rviz_config_package} 包，请确认包名正确！")
    
    # RViz配置文件路径（如果没有该文件，需手动创建或调整路径）
    rviz_config_path = os.path.join(rviz_package_share_dir, 'rviz', 'lidar_detection.rviz')
    
    # 日志输出（便于调试路径问题）
    log_livox_config = LogInfo(msg=f"Livox配置文件路径: {os.path.join(livox_share_dir, 'config', 'MID360_config.json')}")
    log_rviz_path = LogInfo(msg=f"RViz配置文件路径: {rviz_config_path}")

    # 启动RViz2（带指定配置文件）
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_config_path],
        output='screen',
        on_exit=Shutdown()  # RViz关闭时退出整个launch
    )

    # -------------------------- 启动顺序配置 --------------------------
    return LaunchDescription([
        # 日志输出
        log_livox_config,
        log_rviz_path,
        # 先启动Livox驱动（确保点云先发布，避免算法节点订阅超时）
        livox_driver_node,
        # 延迟1秒启动算法容器（给驱动初始化时间）
        TimerAction(
            period=1.0,
            actions=[lidar_detector]
        ),
        # 最后启动RViz可视化
        rviz_node
    ])
