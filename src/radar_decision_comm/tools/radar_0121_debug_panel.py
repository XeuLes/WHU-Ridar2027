#!/usr/bin/env python3

import sys
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from rclpy.node import Node

from radar_decision_comm.msg import RadarDecisionCommand


class Radar0121DebugPanel(Node):
    def __init__(self):
        super().__init__("radar_0121_debug_panel")

        self.declare_parameter("output_topic", "/radar/debug/tx/radar_cmd")
        self.declare_parameter("default_radar_cmd", 1)
        self.declare_parameter("default_password_cmd", 2)
        self.declare_parameter("default_verify_key", "RM2026")

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.default_radar_cmd = int(self.get_parameter("default_radar_cmd").value)
        self.default_password_cmd = int(self.get_parameter("default_password_cmd").value)
        self.default_verify_key = str(self.get_parameter("default_verify_key").value)

        self.publisher = self.create_publisher(RadarDecisionCommand, self.output_topic, 10)

    @staticmethod
    def key_bytes(text):
        if len(text) != 6 or not text.isascii() or not text.isalnum():
            raise ValueError("秘钥必须刚好 6 位，只能包含 ASCII 字母或数字")
        return [ord(ch) for ch in text]

    def publish_command(self, radar_cmd, password_cmd, verify_key):
        msg = RadarDecisionCommand()
        msg.command_type = 1
        msg.command_value = radar_cmd
        msg.radar_cmd = radar_cmd
        msg.password_cmd = password_cmd
        msg.password = self.key_bytes(verify_key)
        msg.note = (
            f"debug_0121_panel|radar_cmd={radar_cmd}|"
            f"password_cmd={password_cmd}|verify={verify_key}"
        )
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published 0x0121 debug command: radar_cmd={radar_cmd} "
            f"password_cmd={password_cmd} verify={verify_key}"
        )


class PanelApp:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("0x0121 Radar Decision Debug")
        self.root.resizable(False, False)

        self.verify_key_var = tk.StringVar(value=node.default_verify_key)
        self.radar_cmd_var = tk.StringVar(value=str(node.default_radar_cmd))
        self.password_cmd_var = tk.StringVar(value=str(node.default_password_cmd))
        self.status_var = tk.StringVar(value=f"发布 topic: {node.output_topic}")

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="双倍易伤 radar_cmd").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, width=12, textvariable=self.radar_cmd_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="秘钥命令 password_cmd").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, width=12, textvariable=self.password_cmd_var).grid(
            row=1, column=1, sticky="ew", pady=(8, 0)
        )

        ttk.Label(frame, text="验证秘钥").grid(row=2, column=0, sticky="w", pady=(8, 0))
        key_entry = ttk.Entry(frame, width=12, textvariable=self.verify_key_var)
        key_entry.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        key_entry.focus_set()

        submit = ttk.Button(frame, text="发送 0x0121", command=self.submit)
        submit.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        self.root.bind("<Return>", lambda _event: self.submit())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def submit(self):
        try:
            radar_cmd = int(self.radar_cmd_var.get())
            password_cmd = int(self.password_cmd_var.get())
            verify_key = self.verify_key_var.get().strip()
            if not 0 <= radar_cmd <= 255:
                raise ValueError("radar_cmd 必须在 0..255")
            if not 0 <= password_cmd <= 255:
                raise ValueError("password_cmd 必须在 0..255")

            self.node.publish_command(radar_cmd, password_cmd, verify_key)
            self.status_var.set(
                f"已发送: radar_cmd={radar_cmd}, password_cmd={password_cmd}, verify={verify_key}"
            )
        except Exception as exc:
            messagebox.showerror("发送失败", str(exc))

    def close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = Radar0121DebugPanel()
    app = PanelApp(node)
    try:
        app.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
