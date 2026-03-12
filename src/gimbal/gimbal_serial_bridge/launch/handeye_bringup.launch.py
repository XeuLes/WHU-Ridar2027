#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ----------------------------
    # Global launch args
    # ----------------------------
    log_level = LaunchConfiguration("log_level")
    start_hik = LaunchConfiguration("start_hik")
    start_chessboard = LaunchConfiguration("start_chessboard")
    start_robot_tf = LaunchConfiguration("start_robot_tf")

    # Hik camera args
    hik_params_file = LaunchConfiguration("hik_params_file")

    # Chessboard detector args
    chessboard_width = LaunchConfiguration("chessboard_width")
    chessboard_height = LaunchConfiguration("chessboard_height")
    square_size = LaunchConfiguration("square_size")
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_info_yaml = LaunchConfiguration("camera_info_yaml")
    use_camera_info_topic = LaunchConfiguration("use_camera_info_topic")
    enable_clahe = LaunchConfiguration("enable_clahe")
    clahe_clip_limit = LaunchConfiguration("clahe_clip_limit")
    clahe_tile_grid_size = LaunchConfiguration("clahe_tile_grid_size")
    median_blur_kernel = LaunchConfiguration("median_blur_kernel")
    use_findchessboard_sb = LaunchConfiguration("use_findchessboard_sb")
    publish_tf = LaunchConfiguration("publish_tf")
    publish_pose = LaunchConfiguration("publish_pose")
    publish_debug_image = LaunchConfiguration("publish_debug_image")
    camera_frame = LaunchConfiguration("camera_frame")
    chessboard_frame = LaunchConfiguration("chessboard_frame")

    # ----------------------------
    # Environment
    # ----------------------------
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable(
        "RCUTILS_COLORIZED_OUTPUT", "1"
    )

    # ----------------------------
    # Defaults
    # ----------------------------
    hik_bringup_dir = get_package_share_directory("hik_camera_ros2_driver")
    hik_launch_file = os.path.join(hik_bringup_dir, "launch", "hik_camera_launch.py")
    default_hik_params = os.path.join(hik_bringup_dir, "config", "camera_params.yaml")

    # ----------------------------
    # Declare args
    # ----------------------------
    declare_log_level = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Global ROS log level"
    )

    declare_start_hik = DeclareLaunchArgument(
        "start_hik",
        default_value="true",
        description="Whether to start Hik camera driver"
    )
    declare_start_chessboard = DeclareLaunchArgument(
        "start_chessboard",
        default_value="true",
        description="Whether to start chessboard detector"
    )
    declare_start_robot_tf = DeclareLaunchArgument(
        "start_robot_tf",
        default_value="true",
        description="Whether to start robot_tf_publisher"
    )

    declare_hik_params_file = DeclareLaunchArgument(
        "hik_params_file",
        default_value=default_hik_params,
        description="Hik camera params yaml path"
    )

    declare_chessboard_width = DeclareLaunchArgument(
        "chessboard_width", default_value="10",
        description="Chessboard inner corners width"
    )
    declare_chessboard_height = DeclareLaunchArgument(
        "chessboard_height", default_value="7",
        description="Chessboard inner corners height"
    )
    declare_square_size = DeclareLaunchArgument(
        "square_size", default_value="0.015",
        description="Chessboard square size in meters"
    )
    declare_image_topic = DeclareLaunchArgument(
        "image_topic", default_value="/camera1/image",
        description="Input image topic for chessboard detector"
    )
    declare_camera_info_topic = DeclareLaunchArgument(
        "camera_info_topic", default_value="/camera1/camera_info",
        description="CameraInfo topic for chessboard detector"
    )
    declare_camera_info_yaml = DeclareLaunchArgument(
        "camera_info_yaml", default_value="/home/user/code/handeye_calibration-main/hik_camera_ros2_driver/config/camera_info1.yaml",
        description="Optional camera calibration yaml path"
    )
    declare_use_camera_info_topic = DeclareLaunchArgument(
        "use_camera_info_topic", default_value="true",
        description="Whether to subscribe CameraInfo topic"
    )
    declare_enable_clahe = DeclareLaunchArgument(
        "enable_clahe", default_value="true",
        description="Enable CLAHE preprocessing"
    )
    declare_clahe_clip_limit = DeclareLaunchArgument(
        "clahe_clip_limit", default_value="2.0",
        description="CLAHE clip limit"
    )
    declare_clahe_tile_grid_size = DeclareLaunchArgument(
        "clahe_tile_grid_size", default_value="[8, 8]",
        description="CLAHE tile grid size"
    )
    declare_median_blur_kernel = DeclareLaunchArgument(
        "median_blur_kernel", default_value="5",
        description="Median blur kernel size"
    )
    declare_use_findchessboard_sb = DeclareLaunchArgument(
        "use_findchessboard_sb", default_value="true",
        description="Fallback to findChessboardCornersSB"
    )
    declare_publish_tf = DeclareLaunchArgument(
        "publish_tf", default_value="true",
        description="Publish chessboard TF"
    )
    declare_publish_pose = DeclareLaunchArgument(
        "publish_pose", default_value="true",
        description="Publish chessboard pose"
    )
    declare_publish_debug_image = DeclareLaunchArgument(
        "publish_debug_image", default_value="true",
        description="Publish /chessboard/debug_image"
    )
    declare_camera_frame = DeclareLaunchArgument(
        "camera_frame", default_value="camera_color_optical_frame",
        description="Camera optical frame id"
    )
    declare_chessboard_frame = DeclareLaunchArgument(
        "chessboard_frame", default_value="chessboard",
        description="Chessboard TF child frame id"
    )

    # ----------------------------
    # Include existing Hik launch
    # ----------------------------
    hik_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(hik_launch_file),
        launch_arguments={
            "params_file": hik_params_file,
            "log_level": log_level,
        }.items(),
        condition=IfCondition(start_hik),
    )

    # ----------------------------
    # my_robot_driver nodes
    # ----------------------------
    chessboard_detector = Node(
        package="my_robot_driver",
        executable="chessboard_detector",
        name="chessboard_detector",
        output="screen",
        condition=IfCondition(start_chessboard),
        parameters=[
            {
                "chessboard_width": chessboard_width,
                "chessboard_height": chessboard_height,
                "square_size": square_size,
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "camera_info_yaml": camera_info_yaml,
                "use_camera_info_topic": use_camera_info_topic,
                "enable_clahe": enable_clahe,
                "clahe_clip_limit": clahe_clip_limit,
                "clahe_tile_grid_size": clahe_tile_grid_size,
                "median_blur_kernel": median_blur_kernel,
                "use_findchessboard_sb": use_findchessboard_sb,
                "publish_tf": publish_tf,
                "publish_pose": publish_pose,
                "publish_debug_image": publish_debug_image,
                "camera_frame": camera_frame,
                "chessboard_frame": chessboard_frame,
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    robot_tf_publisher = Node(
        package="my_robot_driver",
        executable="robot_tf_publisher",
        name="robot_tf_publisher",
        output="screen",
        condition=IfCondition(start_robot_tf),
        arguments=["--ros-args", "--log-level", log_level],
    )

    gimbal_serial_bridge = Node(
    package="gimbal_serial_bridge",
    executable="gimbal_serial_bridge_node",
    name="gimbal_serial_bridge",
    output="screen",
    arguments=["--ros-args", "--log-level", log_level],
    )   

    # ----------------------------
    # Helpful runtime hints
    # ----------------------------
    hint1 = LogInfo(msg="Bringup started: Hik camera + chessboard detector + robot TF publisher")
    hint2 = LogInfo(msg="Verify image: ros2 topic echo /chessboard/debug_image --once")
    hint3 = LogInfo(msg="Verify TF: ros2 run tf2_ros tf2_echo robot_base_frame robot_effector_frame")

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)

    ld.add_action(declare_log_level)
    ld.add_action(declare_start_hik)
    ld.add_action(declare_start_chessboard)
    ld.add_action(declare_start_robot_tf)
    ld.add_action(declare_hik_params_file)

    ld.add_action(declare_chessboard_width)
    ld.add_action(declare_chessboard_height)
    ld.add_action(declare_square_size)
    ld.add_action(declare_image_topic)
    ld.add_action(declare_camera_info_topic)
    ld.add_action(declare_camera_info_yaml)
    ld.add_action(declare_use_camera_info_topic)
    ld.add_action(declare_enable_clahe)
    ld.add_action(declare_clahe_clip_limit)
    ld.add_action(declare_clahe_tile_grid_size)
    ld.add_action(declare_median_blur_kernel)
    ld.add_action(declare_use_findchessboard_sb)
    ld.add_action(declare_publish_tf)
    ld.add_action(declare_publish_pose)
    ld.add_action(declare_publish_debug_image)
    ld.add_action(declare_camera_frame)
    ld.add_action(declare_chessboard_frame)

    ld.add_action(hint1)
    ld.add_action(hint2)
    ld.add_action(hint3)

    ld.add_action(hik_camera)
    ld.add_action(chessboard_detector)
    ld.add_action(robot_tf_publisher)
    ld.add_action(gimbal_serial_bridge)

    return ld