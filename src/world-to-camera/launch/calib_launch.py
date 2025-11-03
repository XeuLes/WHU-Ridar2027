from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 获取功能包路径
    package_share = FindPackageShare(package='world_to_camera_calib')
    
    # 声明启动参数，默认使用config目录下的文件
    json_path_arg = DeclareLaunchArgument(
        'json_path',
        default_value=PathJoinSubstitution([package_share, 'config', 'calibration_results.json']),
        description='Path to the JSON file containing lidar to camera transform'
    )
    
    output_yaml_path_arg = DeclareLaunchArgument(
        'output_yaml_path',
        default_value=PathJoinSubstitution([package_share, 'config', 'world_to_camera.yaml']),
        description='Path to save the output YAML file'
    )
    
    # 定义节点
    world_to_camera_node = Node(
        package='world_to_camera_calib',
        executable='world_to_camera_node',
        name='world_to_camera_calib_node',
        output='screen',
        parameters=[
            {'json_path': LaunchConfiguration('json_path')},
            {'output_yaml_path': LaunchConfiguration('output_yaml_path')}
        ]
    )
    
    # 返回启动描述
    return LaunchDescription([
        json_path_arg,
        output_yaml_path_arg,
        world_to_camera_node
    ])
    