#!/usr/bin/env python3
from multiprocessing import shared_memory

import rclpy
from rclpy.node import Node

from radar_decision_comm.msg import RadarDecisionSync


class RadarDecisionSyncShmNode(Node):
    def __init__(self):
        super().__init__('radar_decision_sync_shm_node')

        self.declare_parameter('input_topic', '/radar/referee/radar_decision_sync')
        self.declare_parameter('shm_name', 'radar_info_shm')
        self.declare_parameter('zero_on_startup', True)
        self.declare_parameter('verbose', False)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        self.zero_on_startup = self.get_parameter('zero_on_startup').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.shm = self._open_or_create_shm(self.shm_name, 1)
        self.buf = self.shm.buf

        if self.zero_on_startup:
            self._write_radar_info(0)

        self.sub = self.create_subscription(
            RadarDecisionSync,
            self.input_topic,
            self._on_sync,
            10,
        )

        self.get_logger().info(
            f'RadarDecisionSync SHM bridge started. topic={self.input_topic}, shm={self.shm_name}'
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

    def _encode_radar_info(self, msg: RadarDecisionSync) -> int:
        radar_info = 0
        radar_info |= int(msg.double_effect_chance) & 0x03
        radar_info |= (1 if msg.is_double_effect_active else 0) << 2
        radar_info |= (int(msg.enemy_key_level) & 0x03) << 3
        radar_info |= (1 if msg.can_change_key else 0) << 5
        return radar_info & 0xFF

    def _write_radar_info(self, radar_info: int):
        self.buf[0] = radar_info & 0xFF

    def _on_sync(self, msg: RadarDecisionSync):
        radar_info = self._encode_radar_info(msg)
        self._write_radar_info(radar_info)

        if self.verbose:
            self.get_logger().info(
                'Write radar_info=0x%02X chance=%u active=%s key_level=%u can_change=%s'
                % (
                    radar_info,
                    int(msg.double_effect_chance),
                    str(bool(msg.is_double_effect_active)).lower(),
                    int(msg.enemy_key_level),
                    str(bool(msg.can_change_key)).lower(),
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
    node = RadarDecisionSyncShmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
