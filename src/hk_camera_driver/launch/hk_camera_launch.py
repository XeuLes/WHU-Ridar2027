# 1. 导入需要的工具（类比雷达启动文件的开头）
import os
# 用于找到ROS2包的路径（比如找到config文件夹在哪）
from ament_index_python.packages import get_package_share_directory
# 描述“要启动哪些东西”的工具
from launch import LaunchDescription
# 定义“要启动的节点”的工具
from launch_ros.actions import Node

# 2. 核心函数：生成启动描述
def generate_launch_description():
    # 2.1 找到参数文件的路径（告诉ROS2去哪里找hk_camera_params.yaml）
    config_file_path = os.path.join(
        # 找到hk_camera_driver包的安装路径
        get_package_share_directory("hk_camera_driver"),
        "config",  # 文件夹名
        "hk_camera_params.yaml"  # 文件名
    )

    # 2.2 定义要启动的相机节点（类比雷达的livox_driver节点）
    hk_camera_driver_node = Node(
        package="hk_camera_driver",  # 节点所在的包（就是我们创建的这个包）
        executable="hk_camera_node",  # 节点的可执行文件名（后续编译会生成）
        name="hk_camera_publisher",   # 给节点起个名字（rqt_graph里能看到）
        output="screen",              # 节点的日志输出到终端（方便看报错）
        parameters=[config_file_path] # 加载上面找到的参数文件
    )

    # 2.3 告诉ROS2：启动上面定义的相机节点
    return LaunchDescription([
        hk_camera_driver_node
    ])

