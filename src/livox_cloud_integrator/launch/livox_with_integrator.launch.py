import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch

################### user configure parameters for ros2 start ###################
xfer_format   = 0    # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
multi_topic   = 0    # 0-All LiDARs share the same topic, 1-One LiDAR one topic
data_src      = 0    # 0-lidar,1-hub
publish_freq  = 10.0 # freqency of publish,1.0,2.0,5.0,10.0,etc
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = '/home/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'

# integrator params
input_topic = '/livox/lidar'
output_topic = '/livox/lidar_integrated'
window_sec = 0.5
use_header_stamp = True
max_points = 2000000
################### user configure parameters for ros2 end #####################


def _must_exist(path: str, hint: str):
    if not os.path.exists(path):
        raise RuntimeError(f"[launch] File not found: {path}\nHint: {hint}")
    return path


# ✅ 正确：从目标包 share 里找资源（不依赖 launch 文件相对路径）
livox_share = get_package_share_directory('livox_ros2_driver')
livox_config_dir = os.path.join(livox_share, 'config')

rviz_config_path = _must_exist(
    os.path.join(livox_config_dir, 'livox_lidar.rviz'),
    "Expected in <share>/livox_ros2_driver/config/ . Check install or filenames."
)

user_config_path = _must_exist(
    os.path.join(livox_config_dir, 'livox_lidar_config.json'),
    "Expected in <share>/livox_ros2_driver/config/ . Check install or filenames."
)

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code}
]


def generate_launch_description():
    livox_driver = Node(
        package='livox_ros2_driver',
        executable='livox_ros2_driver_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_ros2_params
    )

    integrator = Node(
        package='livox_cloud_integrator',
        executable='cloud_integrator_node',
        name='cloud_integrator',
        output='screen',
        parameters=[
            {"input_topic": input_topic},
            {"output_topic": output_topic},
            {"window_sec": window_sec},
            {"use_header_stamp": use_header_stamp},
            {"max_points": int(max_points)},
        ],
    )

    livox_rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['--display-config', rviz_config_path]
    )

    return LaunchDescription([
        livox_driver,
        integrator,
        livox_rviz,

        # 可选：rviz 关了就整体 shutdown
        # launch.actions.RegisterEventHandler(
        #     event_handler=launch.event_handlers.OnProcessExit(
        #         target_action=livox_rviz,
        #         on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
        #     )
        # )
    ])
