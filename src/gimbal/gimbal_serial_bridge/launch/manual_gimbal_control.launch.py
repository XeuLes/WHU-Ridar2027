from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("gimbal_serial_bridge")
    hik_share = get_package_share_directory("hik_camera_ros2_driver")
    aim_share = get_package_share_directory("auto_aim_fast")
    solver_share = get_package_share_directory("gimbal_aim_solver")
    laser_share = get_package_share_directory("laser_aim_controller")
    default_bridge_params = os.path.join(pkg_share, "config", "gimbal_serial_bridge.yaml")
    default_hik_params = os.path.join(hik_share, "config", "camera_params.yaml")
    default_solver_params = os.path.join(solver_share, "config", "gimbal_aim_solver.yaml")
    default_laser_params = os.path.join(laser_share, "config", "laser_aim_controller.yaml")
    default_tracker_params = os.path.join(aim_share, "config", "tracker.yaml")
    default_camera_params = os.path.join(aim_share, "config", "camera.yaml")
    default_detector_params = os.path.join(aim_share, "config", "detector.yaml")
    default_qos_params = os.path.join(aim_share, "config", "qos.yaml")
    default_info_params = os.path.join(aim_share, "config", "camera_info.yaml")
    default_yolo_engine = os.path.join(aim_share, "model", "auto_aim.engine")

    start_bridge = LaunchConfiguration("start_bridge")
    start_camera = LaunchConfiguration("start_camera")
    start_auto_aim = LaunchConfiguration("start_auto_aim")
    start_laser_aim = LaunchConfiguration("start_laser_aim")
    start_gimbal_solver = LaunchConfiguration("start_gimbal_solver")
    control_mode = LaunchConfiguration("control_mode")
    aim_mode = LaunchConfiguration("aim_mode")
    bridge_params_file = LaunchConfiguration("bridge_params_file")
    hik_params_file = LaunchConfiguration("hik_params_file")
    gimbal_solver_params_file = LaunchConfiguration("gimbal_solver_params_file")
    laser_params_file = LaunchConfiguration("laser_params_file")
    cmd_topic = LaunchConfiguration("cmd_topic")
    enable_topic = LaunchConfiguration("enable_topic")
    manual_command_topic = LaunchConfiguration("manual_command_topic")
    final_cmd_topic = LaunchConfiguration("final_cmd_topic")
    laser_target_topic = LaunchConfiguration("laser_target_topic")
    disabled_target_topic = LaunchConfiguration("disabled_target_topic")
    gimbal_feedback_topic = LaunchConfiguration("gimbal_feedback_topic")
    gimbal_quaternion_topic = LaunchConfiguration("gimbal_quaternion_topic")
    yaw_deg = LaunchConfiguration("yaw_deg")
    pitch_deg = LaunchConfiguration("pitch_deg")
    step_deg = LaunchConfiguration("step_deg")
    publish_hz = LaunchConfiguration("publish_hz")
    publish_enable = LaunchConfiguration("publish_enable")
    enable_value = LaunchConfiguration("enable_value")
    yaw_index = LaunchConfiguration("yaw_index")
    pitch_index = LaunchConfiguration("pitch_index")
    publish_dim = LaunchConfiguration("publish_dim")
    tracker_params = LaunchConfiguration("tracker_params")
    camera_params = LaunchConfiguration("camera_params")
    detector_params = LaunchConfiguration("detector_params")
    qos_params = LaunchConfiguration("qos_params")
    camera_info_params = LaunchConfiguration("camera_info_params")
    detector_mode = LaunchConfiguration("detector_mode")
    laser_increment_mode = LaunchConfiguration("laser_increment_mode")
    yolo_engine_path = LaunchConfiguration("yolo_engine_path")
    yolo_input_w = LaunchConfiguration("yolo_input_w")
    yolo_input_h = LaunchConfiguration("yolo_input_h")
    yolo_conf_thresh = LaunchConfiguration("yolo_conf_thresh")
    yolo_nms_thresh = LaunchConfiguration("yolo_nms_thresh")
    yolo_kpt_conf_thresh = LaunchConfiguration("yolo_kpt_conf_thresh")
    yolo_max_det = LaunchConfiguration("yolo_max_det")
    yolo_use_fp16 = LaunchConfiguration("yolo_use_fp16")
    yolo_class_id = LaunchConfiguration("yolo_class_id")
    yolo_swap_rb_input = LaunchConfiguration("yolo_swap_rb_input")

    intra = [{"use_intra_process_comms": True}]
    fine_modes = "['fine', 'full', 'auto']"
    all_vision_modes = (
        "['fine', 'full', 'auto', 'vision_only', 'recognition_only', "
        "'detect_only', 'direct_vision', 'direct_detect', 'direct_recognition']"
    )
    all_direct_modes = (
        "['direct', 'direct_manual', 'direct_vision', 'direct_detect', "
        "'direct_recognition']"
    )

    start_camera_effective = PythonExpression([
        "'", start_camera, "'.lower() == 'true'",
    ])
    start_auto_aim_effective = PythonExpression([
        "'", start_auto_aim, "'.lower() == 'true' and '",
        control_mode, "'.lower() in ", all_vision_modes,
    ])
    start_laser_aim_effective = PythonExpression([
        "'", start_laser_aim, "'.lower() == 'true' and '",
        control_mode, "'.lower() not in ", all_direct_modes,
    ])
    rough_output_topic = PythonExpression([
        "('", control_mode, "'.lower() in ", all_direct_modes, ") and '",
        final_cmd_topic, "' or '", cmd_topic, "'",
    ])
    laser_target_topic_effective = PythonExpression([
        "('", control_mode, "'.lower() in ", fine_modes, ") and '",
        laser_target_topic, "' or '", disabled_target_topic, "'",
    ])
    manual_mode = IfCondition(
        PythonExpression(["'", aim_mode, "'.lower() == 'manual'"])
    )
    scan_solver_mode = IfCondition(
        PythonExpression([
            "'", aim_mode, "'.lower() == 'scan' and '",
            start_gimbal_solver,
            "'.lower() == 'true'",
        ])
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_bridge",
            default_value="true",
            description="Whether to start the gimbal_serial_bridge_node together with manual control",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="true",
            description="Whether to start the Hik camera driver and camera1 rectifier",
        ),
        DeclareLaunchArgument(
            "start_auto_aim",
            default_value="true",
            description="Whether to allow auto_aim_fast when control_mode uses recognition",
        ),
        DeclareLaunchArgument(
            "start_laser_aim",
            default_value="true",
            description="Whether to allow laser_aim_controller when control_mode is not direct",
        ),
        DeclareLaunchArgument(
            "start_gimbal_solver",
            default_value="true",
            description="Whether to start gimbal_aim_solver when aim_mode is scan",
        ),
        DeclareLaunchArgument(
            "aim_mode",
            default_value="manual",
            description="manual: keyboard artificial rough aim; scan: radar rough aim plus scan recovery",
        ),
        DeclareLaunchArgument(
            "control_mode",
            default_value="fine",
            description=(
                "fine/full/auto: recognition + fine aim; "
                "vision_only/recognition_only/detect_only: keep recognition but disable fine aim; "
                "rough_only/rough/manual_only: rough manual/scan only, no recognition; camera still follows start_camera; "
                "direct/direct_manual: rough source directly publishes final command, no recognition or laser; camera still follows start_camera; "
                "direct_vision/direct_detect/direct_recognition: keep recognition and publish rough source directly"
            ),
        ),
        DeclareLaunchArgument(
            "bridge_params_file",
            default_value=default_bridge_params,
            description="Parameter file for gimbal_serial_bridge_node",
        ),
        DeclareLaunchArgument(
            "hik_params_file",
            default_value=default_hik_params,
            description="Hik camera parameter file",
        ),
        DeclareLaunchArgument(
            "gimbal_solver_params_file",
            default_value=default_solver_params,
            description="Parameter file for gimbal_aim_solver_node",
        ),
        DeclareLaunchArgument(
            "laser_params_file",
            default_value=default_laser_params,
            description="Parameter file for laser_aim_controller_node",
        ),
        DeclareLaunchArgument(
            "cmd_topic",
            default_value="/gimbal/rough_command_angles",
            description="Manual artificial rough command topic for accumulated absolute yaw/pitch angles",
        ),
        DeclareLaunchArgument(
            "enable_topic",
            default_value="/gimbal/enable",
            description="Enable topic for gimbal output",
        ),
        DeclareLaunchArgument(
            "manual_command_topic",
            default_value="/gimbal/command_angle_manual",
            description="Optional fine manual offset topic consumed by laser_aim_controller",
        ),
        DeclareLaunchArgument(
            "final_cmd_topic",
            default_value="/gimbal/command_angles",
            description="Final command topic consumed by gimbal_serial_bridge",
        ),
        DeclareLaunchArgument(
            "laser_target_topic",
            default_value="/auto_aim/target",
            description="auto_aim target topic consumed by laser_aim_controller in fine modes",
        ),
        DeclareLaunchArgument(
            "disabled_target_topic",
            default_value="/auto_aim/target_disabled",
            description="Unused target topic used to keep recognition running while laser fine aim is disabled",
        ),
        DeclareLaunchArgument(
            "gimbal_feedback_topic",
            default_value="/gimbal/angles_true",
            description="Feedback yaw/pitch topic used by solver/controller/manual initialization",
        ),
        DeclareLaunchArgument(
            "gimbal_quaternion_topic",
            default_value="/gimbal/quaternion_true",
            description="Feedback quaternion topic used by laser_aim_controller",
        ),
        DeclareLaunchArgument(
            "yaw_deg",
            default_value="0.0",
            description="Initial absolute yaw command in degrees",
        ),
        DeclareLaunchArgument(
            "pitch_deg",
            default_value="0.0",
            description="Initial absolute pitch command in degrees",
        ),
        DeclareLaunchArgument(
            "step_deg",
            default_value="0.05",
            description="Keyboard increment step in degrees",
        ),
        DeclareLaunchArgument(
            "publish_hz",
            default_value="200.0",
            description="Periodic publish rate",
        ),
        DeclareLaunchArgument(
            "publish_enable",
            default_value="true",
            description="Whether to publish /gimbal/enable together with angle commands",
        ),
        DeclareLaunchArgument(
            "enable_value",
            default_value="true",
            description="Value published to the enable topic",
        ),
        DeclareLaunchArgument(
            "yaw_index",
            default_value="0",
            description="Yaw index in the Float64MultiArray command",
        ),
        DeclareLaunchArgument(
            "pitch_index",
            default_value="1",
            description="Pitch index in the Float64MultiArray command",
        ),
        DeclareLaunchArgument(
            "publish_dim",
            default_value="2",
            description="Float64MultiArray size",
        ),
        DeclareLaunchArgument(
            "tracker_params",
            default_value=default_tracker_params,
            description="auto_aim tracker params",
        ),
        DeclareLaunchArgument(
            "camera_params",
            default_value=default_camera_params,
            description="auto_aim camera intrinsics fallback",
        ),
        DeclareLaunchArgument(
            "detector_params",
            default_value=default_detector_params,
            description="auto_aim detector params",
        ),
        DeclareLaunchArgument(
            "qos_params",
            default_value=default_qos_params,
            description="auto_aim qos/debug params",
        ),
        DeclareLaunchArgument(
            "camera_info_params",
            default_value=default_info_params,
            description="auto_aim camera_info params",
        ),
        DeclareLaunchArgument(
            "detector_mode",
            default_value="yolo",
            description="auto_aim detector mode: legacy or yolo",
        ),
        DeclareLaunchArgument(
            "laser_increment_mode",
            default_value="pixel",
            description="laser_aim_controller increment mode: auto, pnp, or pixel",
        ),
        DeclareLaunchArgument(
            "yolo_engine_path",
            default_value=default_yolo_engine,
            description="TensorRT engine path for YOLO pose detector",
        ),
        DeclareLaunchArgument(
            "yolo_input_w",
            default_value="640",
            description="YOLO TensorRT input width",
        ),
        DeclareLaunchArgument(
            "yolo_input_h",
            default_value="640",
            description="YOLO TensorRT input height",
        ),
        DeclareLaunchArgument(
            "yolo_conf_thresh",
            default_value="0.25",
            description="YOLO object confidence threshold",
        ),
        DeclareLaunchArgument(
            "yolo_nms_thresh",
            default_value="0.45",
            description="YOLO NMS IoU threshold for raw pose output",
        ),
        DeclareLaunchArgument(
            "yolo_kpt_conf_thresh",
            default_value="0.25",
            description="YOLO keypoint confidence threshold",
        ),
        DeclareLaunchArgument(
            "yolo_max_det",
            default_value="300",
            description="YOLO max detections per frame",
        ),
        DeclareLaunchArgument(
            "yolo_use_fp16",
            default_value="true",
            description="Use FP16 TensorRT inference",
        ),
        DeclareLaunchArgument(
            "yolo_class_id",
            default_value="0",
            description="Target class id for YOLO detector",
        ),
        DeclareLaunchArgument(
            "yolo_swap_rb_input",
            default_value="true",
            description="Swap R/B channels before YOLO inference",
        ),
        LogInfo(msg=[
            "Gimbal aim bringup: control_mode=", control_mode,
            ", aim_mode=", aim_mode,
            ", rough output topic=", rough_output_topic,
            ", final command topic=", final_cmd_topic,
        ]),
        LogInfo(msg="Manual mode keyboard: W/S pitch, A/D yaw, Q quit keyboard thread"),
        Node(
            package="gimbal_serial_bridge",
            executable="gimbal_serial_bridge_node",
            name="gimbal_serial_bridge",
            output="screen",
            condition=IfCondition(start_bridge),
            parameters=[bridge_params_file],
        ),
        Node(
            package="laser_aim_controller",
            executable="laser_aim_controller_node",
            name="laser_aim_controller_node",
            output="screen",
            condition=IfCondition(start_laser_aim_effective),
            parameters=[
                laser_params_file,
                {
                    "target_topic": laser_target_topic_effective,
                    "search_stage_mode": aim_mode,
                    "increment_mode": laser_increment_mode,
                    "rough_command_topic": cmd_topic,
                    "manual_command_topic": manual_command_topic,
                    "command_topic": final_cmd_topic,
                    "gimbal_angles_topic": gimbal_feedback_topic,
                    "gimbal_quaternion_topic": gimbal_quaternion_topic,
                    "manual_always_add": False,
                },
            ],
        ),
        Node(
            package="gimbal_aim_solver",
            executable="gimbal_aim_solver_node",
            name="gimbal_aim_solver_node",
            output="screen",
            condition=scan_solver_mode,
            parameters=[
                gimbal_solver_params_file,
                {
                    "target_angles_topic": rough_output_topic,
                    "gimbal_angles_topic": gimbal_feedback_topic,
                    "use_increment_output": False,
                },
            ],
        ),
        Node(
            package="gimbal_serial_bridge",
            executable="manual_angle_cmd_node",
            name="manual_angle_cmd_node",
            output="screen",
            emulate_tty=True,
            condition=manual_mode,
            parameters=[{
                "cmd_topic": rough_output_topic,
                "enable_topic": enable_topic,
                "feedback_topic": gimbal_feedback_topic,
                "initialize_from_feedback": True,
                "feedback_timeout_sec": 1.0,
                "yaw_deg": ParameterValue(yaw_deg, value_type=float),
                "pitch_deg": ParameterValue(pitch_deg, value_type=float),
                "step_deg": ParameterValue(step_deg, value_type=float),
                "publish_hz": ParameterValue(publish_hz, value_type=float),
                "publish_enable": ParameterValue(publish_enable, value_type=bool),
                "enable_value": ParameterValue(enable_value, value_type=bool),
                "yaw_index": ParameterValue(yaw_index, value_type=int),
                "pitch_index": ParameterValue(pitch_index, value_type=int),
                "feedback_yaw_index": ParameterValue(yaw_index, value_type=int),
                "feedback_pitch_index": ParameterValue(pitch_index, value_type=int),
                "publish_dim": ParameterValue(publish_dim, value_type=int),
            }],
        ),
        ComposableNodeContainer(
            name="manual_control_vision_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container_mt",
            output="screen",
            condition=IfCondition(start_camera_effective),
            composable_node_descriptions=[
                ComposableNode(
                    package="hik_camera_ros2_driver",
                    plugin="hik_camera_ros2_driver::HikCameraRos2DriverNode",
                    name="hik_camera_ros2_driver",
                    parameters=[hik_params_file],
                    extra_arguments=intra,
                ),
                ComposableNode(
                    package="hik_camera_ros2_driver",
                    plugin="hik_camera_ros2_driver::HikCameraRectifierNode",
                    name="camera1_rectifier",
                    parameters=[
                        {
                            "input_image_topic": "/camera1/image",
                            "input_camera_info_topic": "/camera1/camera_info",
                            "output_image_topic": "/camera1_rect/image",
                            "output_camera_info_topic": "/camera1_rect/camera_info",
                            "use_sensor_data_qos": True,
                        }
                    ],
                    extra_arguments=intra,
                ),
                ComposableNode(
                    package="auto_aim_fast",
                    plugin="auto_aim_fast::AutoAimNode",
                    name="auto_aim_node",
                    condition=IfCondition(start_auto_aim_effective),
                    parameters=[
                        tracker_params,
                        camera_params,
                        detector_params,
                        qos_params,
                        camera_info_params,
                        {
                            "detector.mode": detector_mode,
                            "detector.yolo.engine_path": yolo_engine_path,
                            "detector.yolo.input_w": yolo_input_w,
                            "detector.yolo.input_h": yolo_input_h,
                            "detector.yolo.conf_thresh": yolo_conf_thresh,
                            "detector.yolo.nms_thresh": yolo_nms_thresh,
                            "detector.yolo.kpt_conf_thresh": yolo_kpt_conf_thresh,
                            "detector.yolo.max_det": yolo_max_det,
                            "detector.yolo.use_fp16_input": yolo_use_fp16,
                            "detector.yolo.target_class_id": yolo_class_id,
                            "detector.yolo.swap_rb_input": yolo_swap_rb_input,
                            "debug.publish_debug_image": True,
                            "debug.decimation": 1,
                        },
                    ],
                    extra_arguments=intra,
                ),
            ],
        ),
    ])
