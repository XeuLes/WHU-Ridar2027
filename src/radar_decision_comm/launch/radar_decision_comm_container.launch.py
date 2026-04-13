from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    intra = [{'use_intra_process_comms': True}]

    container = ComposableNodeContainer(
        name='radar_decision_comm_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::RefereeSerialRxComponent',
                name='referee_serial_rx',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::RadarWirelessRxComponent',
                name='radar_wireless_rx',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::ClientBridgeComponent',
                name='client_bridge',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::PerceptionAdapterComponent',
                name='perception_adapter',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::StateBlackboardComponent',
                name='state_blackboard',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::FusionComponent',
                name='fusion_engine',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::DecisionEngineComponent',
                name='decision_engine',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::TxSchedulerComponent',
                name='tx_scheduler',
                parameters=[params_file],
                extra_arguments=intra
            ),
            ComposableNode(
                package='radar_decision_comm',
                plugin='radar_decision_comm::RefereeSerialTxComponent',
                name='referee_serial_tx',
                parameters=[params_file],
                extra_arguments=intra
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('radar_decision_comm'),
                'config',
                'radar_decision_comm.yaml'
            ])
        ),
        container
    ])
