#!/usr/bin/env python3
import json
import struct
import zlib
from multiprocessing import shared_memory

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ShmRingReader:
    RING_MAGIC = 0x52494E47  # 'RING'
    SLOT_MAGIC = 0x534C4F54  # 'SLOT'
    VERSION = 1

    STATE_EMPTY = 0
    STATE_WRITING = 1
    STATE_READY = 2

    RING_HEADER_FMT = "<IHHIQQ"
    RING_HEADER_SIZE = 64

    SLOT_HEADER_FMT = "<IIHHHBBIIQQII"
    SLOT_HEADER_SIZE = 64

    def __init__(self, shm_name: str, expected_source_type: int, verbose: bool = False):
        self.shm_name = shm_name
        self.expected_source_type = int(expected_source_type)
        self.verbose = bool(verbose)

        self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
        self.buf = self.shm.buf

        self.ring_size = None
        self.slot_size = None
        self.frame_max = None

        self.read_index = 0
        self.last_overwrite_count = 0

        self._init_from_ring_header()

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    def _init_from_ring_header(self):
        hdr_size = struct.calcsize(self.RING_HEADER_FMT)
        raw = self.buf[0:hdr_size]
        magic, version, ring_size, slot_size, write_index, overwrite_count = struct.unpack(self.RING_HEADER_FMT, raw)

        if magic != self.RING_MAGIC:
            raise RuntimeError(f"{self.shm_name}: bad ring magic: 0x{magic:08X}")
        if version != self.VERSION:
            raise RuntimeError(f"{self.shm_name}: bad ring version: {version}")

        self.ring_size = int(ring_size)
        self.slot_size = int(slot_size)
        self.frame_max = self.slot_size - self.SLOT_HEADER_SIZE
        self.read_index = int(write_index)
        self.last_overwrite_count = int(overwrite_count)

    def _read_ring_header(self):
        hdr_size = struct.calcsize(self.RING_HEADER_FMT)
        raw = self.buf[0:hdr_size]
        return struct.unpack(self.RING_HEADER_FMT, raw)

    def _slot_offset(self, slot_idx: int) -> int:
        return self.RING_HEADER_SIZE + slot_idx * self.slot_size

    def _slot_header_offset(self, slot_idx: int) -> int:
        return self._slot_offset(slot_idx)

    def _slot_payload_offset(self, slot_idx: int) -> int:
        return self._slot_offset(slot_idx) + self.SLOT_HEADER_SIZE

    def _read_slot_header(self, slot_idx: int):
        off = self._slot_header_offset(slot_idx)
        hdr_size = struct.calcsize(self.SLOT_HEADER_FMT)
        raw = self.buf[off:off + hdr_size]
        return struct.unpack(self.SLOT_HEADER_FMT, raw)

    def read_available(self):
        msgs = []

        ring_magic, ring_ver, ring_size, slot_size, write_index, overwrite_count = self._read_ring_header()

        if ring_magic != self.RING_MAGIC or ring_ver != self.VERSION:
            raise RuntimeError(f"{self.shm_name}: ring header changed/corrupted")

        write_index = int(write_index)
        overwrite_count = int(overwrite_count)

        self.last_overwrite_count = overwrite_count

        while self.read_index < write_index:
            slot_idx = self.read_index % self.ring_size

            hdr = self._read_slot_header(slot_idx)
            (
                state,
                slot_magic,
                version,
                source_type,
                cmd_id,
                seq,
                reserved0,
                frame_len,
                data_len,
                stamp_ns,
                msg_index,
                crc32,
                reserved1,
            ) = hdr

            if state != self.STATE_READY:
                break

            if slot_magic != self.SLOT_MAGIC or version != self.VERSION:
                self.read_index += 1
                continue

            if source_type != self.expected_source_type:
                self.read_index += 1
                continue

            if frame_len < 0 or frame_len > self.frame_max:
                self.read_index += 1
                continue

            payload_off = self._slot_payload_offset(slot_idx)
            frame_bytes = bytes(self.buf[payload_off:payload_off + frame_len])

            crc_calc = zlib.crc32(frame_bytes) & 0xFFFFFFFF
            if crc_calc != crc32:
                self.read_index += 1
                continue

            msg = {
                "source_type_num": int(source_type),
                "cmd_id": int(cmd_id),
                "seq": int(seq),
                "packet_len": int(frame_len),
                "data_len": int(data_len),
                "stamp_ns": int(stamp_ns),
                "msg_index": int(msg_index),
                "crc32": int(crc32),
                "frame_bytes": frame_bytes,
            }
            msgs.append(msg)
            self.read_index += 1

        return msgs


class ShmPduBroadcastNode(Node):
    CMD_NAMES = {
        0x0A01: "robot_position",
        0x0A02: "robot_hp",
        0x0A03: "robot_bullet",
        0x0A04: "team_macro_status",
        0x0A05: "robot_gain",
        0x0A06: "jam_key",
    }

    def __init__(self):
        super().__init__('shm_pdu_broadcast_node')

        self.declare_parameter('info_shm_name', 'radar_info_pdu_shm')
        self.declare_parameter('jam_shm_name', 'radar_jam_pdu_shm')
        self.declare_parameter('poll_period_ms', 5)
        self.declare_parameter('verbose', False)

        self.info_shm_name = self.get_parameter('info_shm_name').get_parameter_value().string_value
        self.jam_shm_name = self.get_parameter('jam_shm_name').get_parameter_value().string_value
        self.poll_period_ms = self.get_parameter('poll_period_ms').get_parameter_value().integer_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.pub_all = self.create_publisher(String, '/radar/all_pdu', 10)
        self.pub_info = self.create_publisher(String, '/radar/info_pdu', 10)
        self.pub_jam = self.create_publisher(String, '/radar/jam_pdu', 10)

        self.info_reader = None
        self.jam_reader = None

        self._try_open_readers()
        self.timer = self.create_timer(self.poll_period_ms / 1000.0, self._poll_once)

        self.get_logger().info(
            f"SHM readers initialized. info_shm={self.info_shm_name}, "
            f"jam_shm={self.jam_shm_name}, poll={self.poll_period_ms} ms"
        )

    def _try_open_readers(self):
        if self.info_reader is None:
            try:
                self.info_reader = ShmRingReader(self.info_shm_name, 1, self.verbose)
                self.get_logger().info(f'Opened info shared memory: {self.info_shm_name}')
            except FileNotFoundError:
                self.get_logger().warning(f'Info shared memory not found yet: {self.info_shm_name}')
            except Exception as e:
                self.get_logger().error(f'Failed to open info shared memory: {e}')

        if self.jam_reader is None:
            try:
                self.jam_reader = ShmRingReader(self.jam_shm_name, 2, self.verbose)
                self.get_logger().info(f'Opened jam shared memory: {self.jam_shm_name}')
            except FileNotFoundError:
                self.get_logger().warning(f'Jam shared memory not found yet: {self.jam_shm_name}')
            except Exception as e:
                self.get_logger().error(f'Failed to open jam shared memory: {e}')

    def _make_json_msg(self, item: dict, source_name: str) -> String:
        frame_bytes = item["frame_bytes"]
        data_hex = frame_bytes[7:-2].hex() if len(frame_bytes) >= 9 else ""

        obj = {
            "source_type": source_name,
            "cmd_id": item["cmd_id"],
            "seq": item["seq"],
            "packet_len": item["packet_len"],
            "data_len": item["data_len"],
            "cmd_name": self.CMD_NAMES.get(item["cmd_id"], "unknown"),
            "data_hex": data_hex,
            "frame_hex": frame_bytes.hex(),
            "stamp_ns": item["stamp_ns"],
            "msg_index": item["msg_index"],
            "crc32": item["crc32"],
        }

        msg = String()
        msg.data = json.dumps(obj, ensure_ascii=False)
        return msg

    def _publish_items(self, items, source_name: str):
        for item in items:
            msg = self._make_json_msg(item, source_name)
            self.pub_all.publish(msg)

            if source_name == 'info' or (0x0A01 <= item["cmd_id"] <= 0x0A05):
                self.pub_info.publish(msg)

            if source_name == 'jam' or item["cmd_id"] == 0x0A06:
                self.pub_jam.publish(msg)

    def _poll_once(self):
        if self.info_reader is None or self.jam_reader is None:
            self._try_open_readers()

        if self.info_reader is not None:
            try:
                items = self.info_reader.read_available()
                if items:
                    self._publish_items(items, 'info')
            except Exception as e:
                self.get_logger().error(f'Info SHM read failed: {e}')
                try:
                    self.info_reader.close()
                except Exception:
                    pass
                self.info_reader = None

        if self.jam_reader is not None:
            try:
                items = self.jam_reader.read_available()
                if items:
                    self._publish_items(items, 'jam')
            except Exception as e:
                self.get_logger().error(f'Jam SHM read failed: {e}')
                try:
                    self.jam_reader.close()
                except Exception:
                    pass
                self.jam_reader = None

    def destroy_node(self):
        try:
            if self.info_reader is not None:
                self.info_reader.close()
        except Exception:
            pass

        try:
            if self.jam_reader is not None:
                self.jam_reader.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ShmPduBroadcastNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
