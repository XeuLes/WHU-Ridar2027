#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import transforms3d
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import yaml
import time
import os

class UltimateCalibrator(Node):
    def __init__(self):
        super().__init__('manual_calibrator_ultimate')
        
        self.declare_parameter('output_topic', '/livox/manual_preview')
        self.declare_parameter('target_frame', 'rm_frame')
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.sub = None
        self.current_topic = ""
        self.qos_mode = "MAP" 
        
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        
        self.preview_points = None # 降采样用于UI显示
        self.full_points = None    # 全量数据用于保存PCD
        self.latest_header = None
        self.ui_callback = None 

        self.params = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.get_logger().info("校准工具 V3 (PCD保存版) 已启动...")

    def get_topics(self):
        tps = self.get_topic_names_and_types()
        return sorted([name for name, types in tps if 'sensor_msgs/msg/PointCloud2' in types])

    def set_qos_mode(self, mode):
        self.qos_mode = mode
        if self.current_topic:
            self.update_subscription(self.current_topic)

    def update_subscription(self, topic):
        if not topic: return
        
        if self.sub:
            self.destroy_subscription(self.sub)
            
        self.current_topic = topic
        
        if self.qos_mode == "MAP":
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE, 
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.get_logger().info(f"订阅 {topic} [模式: 地图/Reliable]")
        else: 
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            self.get_logger().info(f"订阅 {topic} [模式: 实时/BestEffort]")

        try:
            self.sub = self.create_subscription(PointCloud2, topic, self.callback, qos)
            if self.ui_callback: self.ui_callback("waiting")
        except Exception as e:
            self.get_logger().error(f"订阅失败: {e}")

    def callback(self, msg):
        self.latest_header = msg.header
        
        try:
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            raw_list = list(gen)
            
            if not raw_list:
                return

            raw_arr = np.array(raw_list)
            
            if raw_arr.dtype.names:
                x = raw_arr['x']
                y = raw_arr['y']
                z = raw_arr['z']
                raw = np.column_stack((x, y, z)).astype(np.float32)
            else:
                raw = raw_arr.astype(np.float32)

            if raw.ndim == 1:
                raw = raw.reshape(-1, 3)
            
            # --- 关键修改：保存全量数据 ---
            self.full_points = raw
            
            # 智能降采样 (仅用于UI预览)
            limit = 10000
            if len(raw) > limit:
                step = len(raw) // limit
                self.preview_points = raw[::step]
            else:
                self.preview_points = raw

            if self.ui_callback: 
                self.ui_callback("received", len(raw))
            
            self.publish_preview()
            
        except Exception as e:
            self.get_logger().error(f"解析错误: {e}")

    def get_current_transform_matrix(self):
        T = np.eye(4)
        T[0:3, 3] = [self.params['x'], self.params['y'], self.params['z']]
        R = transforms3d.euler.euler2mat(
            np.radians(self.params['roll']),
            np.radians(self.params['pitch']),
            np.radians(self.params['yaw']), axes='sxyz'
        )
        T[0:3, 0:3] = R
        return T

    def publish_preview(self):
        if self.preview_points is None: return
        
        try:
            pts = self.preview_points
            if pts.ndim == 1: pts = pts.reshape(-1, 3)

            T = self.get_current_transform_matrix()
            
            ones = np.ones((pts.shape[0], 1))
            pts_hom = np.hstack((pts, ones))
            trans_pts = (T @ pts_hom.T).T[:, :3]

            header = self.latest_header
            header.frame_id = self.target_frame
            header.stamp = self.get_clock().now().to_msg()
            
            out_msg = pc2.create_cloud_xyz32(header, trans_pts)
            self.pub.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"发布错误: {e}")

    def update_param(self, k, v):
        self.params[k] = float(v)
        if self.preview_points is not None:
            self.publish_preview()

    def save_yaml(self, path):
        # 强制转换为标准float，修复yaml保存报错问题
        params_dict = {}
        for k, v in self.params.items():
            if 'roll' in k or 'pitch' in k or 'yaw' in k:
                val = float(np.radians(v)) 
            else:
                val = float(v)
            params_dict[f'initial_{k}'] = val

        data = {
            '/**': {
                'ros__parameters': params_dict
            }
        }
        try:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=None, sort_keys=False)
            self.get_logger().info(f"✅ YAML 参数已保存: {path}")
            return True
        except Exception as e:
            self.get_logger().error(f"YAML 保存失败: {e}")
            return False

    def save_pcd(self, path):
        if self.full_points is None:
            self.get_logger().warn("没有点云数据，无法保存 PCD")
            return False
            
        try:
            self.get_logger().info(f"正在转换并保存 PCD ({len(self.full_points)} points)...")
            
            # 1. 应用当前变换到全量点云
            pts = self.full_points
            if pts.ndim == 1: pts = pts.reshape(-1, 3)
            
            T = self.get_current_transform_matrix()
            ones = np.ones((pts.shape[0], 1))
            pts_hom = np.hstack((pts, ones))
            trans_pts = (T @ pts_hom.T).T[:, :3]
            
            # 2. 写入 PCD 文件 (ASCII 格式，兼容性最好)
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(trans_pts)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(trans_pts)}
DATA ascii
"""
            with open(path, 'w') as f:
                f.write(header)
                # 使用 numpy 快速写入
                np.savetxt(f, trans_pts, fmt='%.6f %.6f %.6f')
            
            self.get_logger().info(f"✅ PCD 已保存: {path}")
            return True
        except Exception as e:
            self.get_logger().error(f"PCD 保存失败: {e}")
            return False

# --- UI ---
def run_ui(node):
    root = tk.Tk()
    root.title("Calibrator V3 (Save PCD)")
    root.geometry("450x850")

    # 1. 话题与模式
    f_top = tk.LabelFrame(root, text="1. Input Source", padx=5, pady=5, fg="blue")
    f_top.pack(fill='x', padx=10, pady=5)
    
    qos_var = tk.StringVar(value="MAP")
    def on_mode_change():
        node.set_qos_mode(qos_var.get())
        
    f_mode = tk.Frame(f_top)
    f_mode.pack(fill='x', pady=2)
    tk.Label(f_mode, text="Mode: ").pack(side='left')
    tk.Radiobutton(f_mode, text="Map (Reliable)", variable=qos_var, value="MAP", 
                   command=on_mode_change).pack(side='left')
    tk.Radiobutton(f_mode, text="Lidar (Best Effort)", variable=qos_var, value="LIVE", 
                   command=on_mode_change).pack(side='left')

    combo = ttk.Combobox(f_top)
    combo.pack(fill='x', pady=5)
    
    def refresh():
        ts = node.get_topics()
        combo['values'] = ts
        if not ts: combo.set("Scanning...")
        else: combo.set("Select Topic...")
    
    tk.Button(f_top, text="↻ Refresh Topics", command=refresh).pack(anchor='e')
    combo.bind("<<ComboboxSelected>>", lambda e: node.update_subscription(combo.get()))

    # 状态栏
    lbl_stat = tk.Label(root, text="Status: Idle", bg="#ddd")
    lbl_stat.pack(fill='x', padx=10)
    
    def ui_update(s, n=0):
        if s=="waiting": lbl_stat.config(text="⏳ Waiting for data...", bg="orange", fg="white")
        elif s=="received": lbl_stat.config(text=f"✅ Received {n} points", bg="#4CAF50", fg="white")
    node.ui_callback = ui_update

    # 2. 坐标系
    f_fr = tk.LabelFrame(root, text="2. Output Frame", padx=5, pady=5)
    f_fr.pack(fill='x', padx=10)
    e_fr = tk.Entry(f_fr)
    e_fr.insert(0, node.target_frame)
    e_fr.pack(fill='x')
    def on_fr(e): 
        node.target_frame = e_fr.get()
        node.publish_preview()
    e_fr.bind('<Return>', on_fr)

    # 3. 滑块
    f_sl = tk.LabelFrame(root, text="3. Transform", padx=5, pady=5)
    f_sl.pack(fill='both', expand=True, padx=10)
    
    def mk_sl(lbl, key, minv, maxv):
        f = tk.Frame(f_sl)
        f.pack(fill='x')
        tk.Label(f, text=lbl, width=5).pack(side='left')
        tk.Scale(f, from_=minv, to=maxv, orient='horizontal', resolution=0.01, 
                 command=lambda v: node.update_param(key, v), showvalue=1).pack(side='right', fill='x', expand=True)

    mk_sl("X", 'x', -100, 100)
    mk_sl("Y", 'y', -100, 100)
    mk_sl("Z", 'z', -20, 20)
    tk.Frame(f_sl, height=1, bg="#999").pack(fill='x', pady=5)
    mk_sl("R", 'roll', -180, 180)
    mk_sl("P", 'pitch', -180, 180)
    mk_sl("Y", 'yaw', -180, 180)

    # 4. 保存区域 (分开两个按钮)
    f_save = tk.LabelFrame(root, text="4. Save Options", padx=5, pady=5, fg="green")
    f_save.pack(fill='x', padx=10, pady=10)

    def btn_save_yaml():
        fn = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".yaml",
            filetypes=[("YAML Config", "*.yaml")]
        )
        if fn:
            if node.save_yaml(fn):
                messagebox.showinfo("Success", f"Parameters saved to:\n{fn}")

    def btn_save_pcd():
        fn = filedialog.asksaveasfilename(
            title="Save Transformed PointCloud",
            defaultextension=".pcd",
            filetypes=[("Point Cloud Data", "*.pcd")]
        )
        if fn:
            # 稍微卡顿一下是正常的，因为在写入文件
            if node.save_pcd(fn):
                messagebox.showinfo("Success", f"PCD saved to:\n{fn}")
            else:
                messagebox.showerror("Error", "Failed to save PCD (No data?)")

    tk.Button(f_save, text="💾 Save Parameters (.yaml)", bg="#008CBA", fg="white", height=2,
              command=btn_save_yaml).pack(fill='x', pady=2)
              
    tk.Button(f_save, text="☁️ Save PointCloud (.pcd)", bg="#4CAF50", fg="white", height=2,
              command=btn_save_pcd).pack(fill='x', pady=2)

    root.after(500, refresh)
    root.mainloop()

def main():
    rclpy.init()
    node = UltimateCalibrator()
    Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    run_ui(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()