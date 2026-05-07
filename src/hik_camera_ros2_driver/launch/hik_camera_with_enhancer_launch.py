import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # 获取包目录（动态获取，适配不同安装路径）
    bringup_dir = get_package_share_directory("hik_camera_ros2_driver")
    
    # 声明配置变量
    params_file = LaunchConfiguration("params_file")
    log_level = LaunchConfiguration("log_level")
    enhancer_input_topic = LaunchConfiguration("enhancer_input_topic")
    enhancer_output_topic = LaunchConfiguration("enhancer_output_topic")
    intra = [{"use_intra_process_comms": True}]
    
    # 明确写死RViz配置文件路径（固定为config目录下的camera.rviz）
    rviz_config_path = os.path.join(bringup_dir, "config", "camera.rviz")

    # 设置环境变量（日志相关）
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    # 声明启动参数
    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(bringup_dir, "config", "camera_params.yaml"),
        description="相机参数配置文件路径"
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level", default_value="info", description="日志级别（debug/info/warn/error/fatal）"
    )

    declare_enhancer_input_cmd = DeclareLaunchArgument(
        "enhancer_input_topic", 
        default_value="/camera1/image",
        description="图像增强节点的输入话题"
    )

    declare_enhancer_output_cmd = DeclareLaunchArgument(
        "enhancer_output_topic", 
        default_value="/camera1/enhance",
        description="图像增强节点的输出话题"
    )

    def rectifier_component(camera_name):
        return ComposableNode(
            package="hik_camera_ros2_driver",
            plugin="hik_camera_ros2_driver::HikCameraRectifierNode",
            name=f"{camera_name}_rectifier",
            parameters=[
                {
                    "input_image_topic": f"/{camera_name}/image",
                    "input_camera_info_topic": f"/{camera_name}/camera_info",
                    "output_image_topic": f"/{camera_name}_rect/image",
                    "output_camera_info_topic": f"/{camera_name}_rect/camera_info",
                    "use_sensor_data_qos": True,
                }
            ],
            extra_arguments=intra,
        )

    # 相机驱动+camera0去畸变组件容器
    start_hik_camera_cmd = ComposableNodeContainer(
        name="hik_camera_with_rectifier_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="hik_camera_ros2_driver",
                plugin="hik_camera_ros2_driver::HikCameraRos2DriverNode",
                name="hik_camera_ros2_driver",
                parameters=[params_file],
                extra_arguments=intra,
            ),
            rectifier_component("camera0"),
        ],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
    )

    # 图像增强节点
    start_image_enhancer_cmd = Node(
        name="image_enhancer_node",
        package="hik_camera_ros2_driver",
        executable="image_enhancer_node",
        parameters=[{
            "input_topic": enhancer_input_topic,
            "output_topic": enhancer_output_topic
        }],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
    )

    # RViz节点（路径写死为config/camera.rviz）
    start_rviz_cmd = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],  # 直接使用固定路径
        output="screen",
    )

    # 构建启动描述
    ld = LaunchDescription()
    
    # 添加环境变量设置
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)
    
    # 添加参数声明
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_enhancer_input_cmd)
    ld.add_action(declare_enhancer_output_cmd)
    
    # 添加节点启动动作
    ld.add_action(start_hik_camera_cmd)
    ld.add_action(start_image_enhancer_cmd)
    ld.add_action(start_rviz_cmd)  # 添加RViz节点

    return ld
