#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PduSubscriberNode(Node):
    def __init__(self):
        super().__init__('pdu_subscriber_node')

        self.declare_parameter('topic_name', '/radar/all_pdu')
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.sub = self.create_subscription(
            String,
            self.topic_name,
            self.callback,
            10
        )

        self.get_logger().info(f'Subscribing topic: {self.topic_name}')

    def callback(self, msg: String):
        try:
            obj = json.loads(msg.data)
            source_type = obj.get('source_type', 'unknown')
            cmd_id = int(obj.get('cmd_id', 0))
            seq = int(obj.get('seq', 0))
            cmd_name = obj.get('cmd_name', '')
            data_hex = obj.get('data_hex', '')

            self.get_logger().info(
                f'[RX] source={source_type} '
                f'cmd_id=0x{cmd_id:04X} '
                f'cmd_name={cmd_name} '
                f'seq={seq} '
                f'data_hex={data_hex}'
            )
        except Exception as e:
            self.get_logger().warning(f'Failed to parse message: {e}, raw={msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = PduSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
