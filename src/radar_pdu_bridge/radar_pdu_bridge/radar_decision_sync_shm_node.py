#!/usr/bin/env python3
from multiprocessing import shared_memory

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from radar_decision_comm.msg import RadarDecisionSync
from radar_decision_comm.msg import RefereeRobotIdentity


class RadarDecisionSyncShmNode(Node):
    def __init__(self):
        super().__init__('radar_decision_sync_shm_node')

        self.declare_parameter('input_topic', '/radar/referee/radar_decision_sync')
        self.declare_parameter('identity_topic', '/radar/referee/robot_identity')
        self.declare_parameter('shm_name', 'radar_info_shm')
        self.declare_parameter('zero_on_startup', True)
        self.declare_parameter('verbose', False)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.identity_topic = self.get_parameter('identity_topic').get_parameter_value().string_value
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        self.zero_on_startup = self.get_parameter('zero_on_startup').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.shm = self._open_or_create_shm(self.shm_name, 1)
        self.buf = self.shm.buf
        initial_radar_info = int(self.buf[0])
        self.sync_bits = initial_radar_info & 0x3F
        self.identity_bits = initial_radar_info & 0xC0

        if self.zero_on_startup:
            self.sync_bits = 0
            self.identity_bits = 0
            self._write_current_radar_info()

        self.sub = self.create_subscription(
            RadarDecisionSync,
            self.input_topic,
            self._on_sync,
            10,
        )
        self.identity_sub = self.create_subscription(
            RefereeRobotIdentity,
            self.identity_topic,
            self._on_identity,
            10,
        )

        self.get_logger().info(
            'RadarDecisionSync SHM bridge started. '
            f'sync_topic={self.input_topic}, identity_topic={self.identity_topic}, '
            f'shm={self.shm_name}'
        )

    def _open_or_create_shm(self, shm_name: str, size: int) -> shared_memory.SharedMemory:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self.get_logger().info(f'Opened existing shared memory: {shm_name}')
            return shm
        except FileNotFoundError:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
            self.get_logger().info(f'Created shared memory: {shm_name} size={size}')
            return shm

    def _encode_sync_bits(self, msg: RadarDecisionSync) -> int:
        sync_bits = 0
        sync_bits |= int(msg.double_effect_chance) & 0x03
        sync_bits |= (1 if msg.is_double_effect_active else 0) << 2
        sync_bits |= (int(msg.enemy_key_level) & 0x03) << 3
        sync_bits |= (1 if msg.can_change_key else 0) << 5
        return sync_bits & 0x3F

    def _encode_identity_bits(self, msg: RefereeRobotIdentity):
        if not msg.valid:
            return None

        self_color = int(msg.self_color)
        if self_color not in (0, 2):
            self.get_logger().warning(
                f'Ignore invalid robot identity self_color={self_color} robot_id={msg.robot_id}'
            )
            return None

        # Store RefereeRobotIdentity.self_color in bit6-bit7: 0=Blue, 2=Red.
        return (self_color & 0x03) << 6

    def _write_radar_info(self, radar_info: int):
        self.buf[0] = radar_info & 0xFF

    def _write_current_radar_info(self):
        self._write_radar_info(self.sync_bits | self.identity_bits)

    def _on_sync(self, msg: RadarDecisionSync):
        self.sync_bits = self._encode_sync_bits(msg)
        self._write_current_radar_info()
        radar_info = self.sync_bits | self.identity_bits

        if self.verbose:
            self.get_logger().info(
                'Write radar_info=0x%02X chance=%u active=%s key_level=%u can_change=%s '
                'identity_bits=0x%02X'
                % (
                    radar_info,
                    int(msg.double_effect_chance),
                    str(bool(msg.is_double_effect_active)).lower(),
                    int(msg.enemy_key_level),
                    str(bool(msg.can_change_key)).lower(),
                    self.identity_bits,
                )
            )

    def _on_identity(self, msg: RefereeRobotIdentity):
        identity_bits = self._encode_identity_bits(msg)
        if identity_bits is None:
            return

        self.identity_bits = identity_bits
        self._write_current_radar_info()

        if self.verbose:
            self.get_logger().info(
                'Write radar_info=0x%02X robot_id=%u self_color=%d identity_bits=0x%02X'
                % (
                    self.sync_bits | self.identity_bits,
                    int(msg.robot_id),
                    int(msg.self_color),
                    self.identity_bits,
                )
            )

    def destroy_node(self):
        try:
            if hasattr(self, 'shm') and self.shm is not None:
                self.shm.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RadarDecisionSyncShmNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
