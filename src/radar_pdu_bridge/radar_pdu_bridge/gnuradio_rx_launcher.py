#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def _normalize_identity(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in ("red", "r", "2"):
        return "red"
    if normalized in ("blue", "b", "0"):
        return "blue"
    raise ValueError(f"default_identity must be red/blue/2/0, got: {value!r}")


class GnuradioRxLauncher(Node):
    def __init__(self):
        super().__init__("gnuradio_rx_launcher")

        self.declare_parameter(
            "script_path",
            "/home/user/gnuradio_grc/0416/rx_red_gfsk.py",
        )
        self.declare_parameter("default_identity", "red")
        self.declare_parameter("allow_direct_13", True)
        self.declare_parameter("access_code0", "2F6F4C74B914492E")
        self.declare_parameter("access_code1", "16E8D377151C712D")

        self.script_path = Path(
            self.get_parameter("script_path").get_parameter_value().string_value
        ).expanduser()
        self.default_identity = _normalize_identity(
            self.get_parameter("default_identity").get_parameter_value().string_value
        )
        self.allow_direct_13 = self.get_parameter("allow_direct_13").get_parameter_value().bool_value
        self.access_code0 = self.get_parameter("access_code0").get_parameter_value().string_value
        self.access_code1 = self.get_parameter("access_code1").get_parameter_value().string_value

        self.tb = None
        self._start_flowgraph()

    def _load_flowgraph_module(self):
        if not self.script_path.is_file():
            raise RuntimeError(f"GNU Radio script not found: {self.script_path}")

        script_dir = str(self.script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        spec = importlib.util.spec_from_file_location(
            "radar_external_rx_red_gfsk",
            str(self.script_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load GNU Radio script: {self.script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _configure_level_controller(self):
        level_ctrl = getattr(self.tb, "epy_block_0", None)
        if level_ctrl is None:
            self.get_logger().warning(
                "GNU Radio flowgraph has no epy_block_0 level controller; "
                "identity and key-level transition options cannot be injected."
            )
            return

        if hasattr(level_ctrl, "default_identity"):
            level_ctrl.default_identity = self.default_identity
        if hasattr(level_ctrl, "current_identity"):
            level_ctrl.current_identity = self.default_identity
        if hasattr(level_ctrl, "allow_direct_13"):
            level_ctrl.allow_direct_13 = self.allow_direct_13

    def _start_flowgraph(self):
        module = self._load_flowgraph_module()
        top_block_cls = getattr(module, "rx_red_gfsk", None)
        if top_block_cls is None:
            raise RuntimeError(f"{self.script_path} does not define rx_red_gfsk")

        self.tb = top_block_cls(
            access_code0=self.access_code0,
            access_code1=self.access_code1,
        )
        self._configure_level_controller()

        snippets = getattr(module, "snippets_main_after_init", None)
        if callable(snippets):
            snippets(self.tb)

        self.tb.start()
        self.get_logger().info(
            "GNU Radio RX started: script=%s default_identity=%s allow_direct_13=%s "
            "access_code0=%s access_code1=%s"
            % (
                str(self.script_path),
                self.default_identity,
                str(self.allow_direct_13).lower(),
                self.access_code0,
                self.access_code1,
            )
        )

    def destroy_node(self):
        if self.tb is not None:
            try:
                self.tb.stop()
                self.tb.wait()
            except Exception as exc:
                self.get_logger().warning(f"Failed to stop GNU Radio RX cleanly: {exc}")
            self.tb = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GnuradioRxLauncher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
