# ros2 launch camera_field_calib_tool plane_polygon.launch.py \
#   image_topic:=/camera/image \
#   out_matrix_yaml:=/home/user/code/T-DT_Radar-main/src/camera_field_calib_tool/config/out_matrix.yaml \
#   field_points_yaml:=/home/user/code/T-DT_Radar-main/src/camera_field_calib_tool/config/Field_Points.yaml


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ============================================================
    # Hik camera driver defaults
    # ============================================================
    hik_pkg_share = get_package_share_directory("hik_camera_ros2_driver")
    hik_default_params = os.path.join(
        hik_pkg_share, "config", "camera_params.yaml"
    )

    params_file = LaunchConfiguration("params_file")
    log_level = LaunchConfiguration("log_level")

    # ============================================================
    # Plane polygon tool defaults
    # ============================================================
    tool_pkg_share = FindPackageShare("camera_field_calib_tool")

    default_intrinsics = PathJoinSubstitution(
        [tool_pkg_share, "config", "camera_intrinsics.yaml"]
    )
    default_out_matrix = PathJoinSubstitution(
        [tool_pkg_share, "config", "out_matrix_gc.yaml"]
    )
    default_field_points = PathJoinSubstitution(
        [tool_pkg_share, "config", "Field_Points.yaml"]
    )

    # ============================================================
    # Environment variables (same as camera driver launch)
    # ============================================================
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    colorized_output_envvar = SetEnvironmentVariable(
        "RCUTILS_COLORIZED_OUTPUT", "1"
    )

    # ============================================================
    # Launch arguments — Hik camera
    # ============================================================
    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=hik_default_params,
        description="Hik camera driver parameter file path",
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level for hik_camera_ros2_driver",
    )

    # ============================================================
    # Launch arguments — Plane polygon node
    # ============================================================
    declare_image_topic_cmd = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera0/image",
        description="Input image topic for plane_polygon_node",
    )

    declare_intrinsics_yaml_cmd = DeclareLaunchArgument(
        "intrinsics_yaml",
        default_value=default_intrinsics,
        description="Camera intrinsics YAML",
    )

    declare_out_matrix_yaml_cmd = DeclareLaunchArgument(
        "out_matrix_yaml",
        default_value=default_out_matrix,
        description="Extrinsics YAML (from PnP calibration)",
    )

    declare_field_points_yaml_cmd = DeclareLaunchArgument(
        "field_points_yaml",
        default_value=default_field_points,
        description="Output Field_Points.yaml (plane polygons)",
    )

    declare_default_plane_z_cmd = DeclareLaunchArgument(
        "default_plane_z",
        default_value="0.0",
        description="Default plane height Z",
    )

    declare_start_plane_index_cmd = DeclareLaunchArgument(
        "start_plane_index",
        default_value="0",
        description="Start plane index (plane0, plane1, ...)",
    )

    # ============================================================
    # Nodes
    # ============================================================
    start_hik_camera_cmd = Node(
        name="hik_camera_ros2_driver",
        package="hik_camera_ros2_driver",
        executable="hik_camera_ros2_driver_node",
        parameters=[params_file],
        arguments=["--ros-args", "--log-level", log_level],
        output="screen",
    )

    start_plane_polygon_cmd = Node(
        name="plane_polygon_node",
        package="camera_field_calib_tool",
        executable="plane_polygon_node",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "intrinsics_yaml": LaunchConfiguration("intrinsics_yaml"),
            "out_matrix_yaml": LaunchConfiguration("out_matrix_yaml"),
            "field_points_yaml": LaunchConfiguration("field_points_yaml"),
            "default_plane_z": LaunchConfiguration("default_plane_z"),
            "start_plane_index": LaunchConfiguration("start_plane_index"),
        }],
    )

    # ============================================================
    # Launch description
    # ============================================================
    ld = LaunchDescription()

    # env
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(colorized_output_envvar)

    # args
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_log_level_cmd)

    ld.add_action(declare_image_topic_cmd)
    ld.add_action(declare_intrinsics_yaml_cmd)
    ld.add_action(declare_out_matrix_yaml_cmd)
    ld.add_action(declare_field_points_yaml_cmd)
    ld.add_action(declare_default_plane_z_cmd)
    ld.add_action(declare_start_plane_index_cmd)

    # nodes
    ld.add_action(start_hik_camera_cmd)
    ld.add_action(start_plane_polygon_cmd)

    return ld
