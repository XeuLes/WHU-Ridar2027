#!/usr/bin/env python3
import argparse
import ctypes
import math
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOWER_EXPOSURE_ALPHA = 0.55


class CudaRuntime:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, what):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code)
            text = msg.decode("utf-8", "replace") if msg else str(code)
            raise RuntimeError(f"{what} failed: cuda error {code}: {text}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
            "cudaMalloc",
        )
        return ptr

    def free(self, ptr):
        if ptr:
            self.lib.cudaFree(ptr)

    def memcpy_h2d(self, dst, src_np):
        src = np.ascontiguousarray(src_np)
        self._check(
            self.lib.cudaMemcpy(
                dst,
                ctypes.c_void_p(src.ctypes.data),
                ctypes.c_size_t(src.nbytes),
                self.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy H2D",
        )

    def memcpy_d2h(self, dst_np, src):
        dst = np.ascontiguousarray(dst_np)
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst.ctypes.data),
                src,
                ctypes.c_size_t(dst.nbytes),
                self.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy D2H",
        )
        return dst


def load_config(path):
    engine = "model/TensorRT/uav_train7_best_fp16.engine"
    conf = 0.35
    if not path or not os.path.exists(path):
        return engine, conf

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return engine, conf

    node = fs.getNode("uav_path")
    if not node.empty():
        engine = node.string()
    node = fs.getNode("uav_conf_threshold")
    if not node.empty():
        conf = float(node.real())
    fs.release()
    return engine, conf


def letterbox_bgr_to_rgb_chw(frame, input_w, input_h, dtype):
    h, w = frame.shape[:2]
    scale = min(input_w / float(w), input_h / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    if dtype == np.float16:
        chw = chw.astype(np.float16)
    return np.ascontiguousarray(chw[None]), scale, pad_x, pad_y


def tensor_dtype_to_np(dtype):
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.float32:
        return np.float32
    if dtype == trt.int32:
        return np.int32
    if dtype == trt.int8:
        return np.int8
    raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")


def volume(shape):
    n = 1
    for d in shape:
        if d < 0:
            raise RuntimeError(f"Dynamic shape not resolved: {shape}")
        n *= int(d)
    return n


def nms(boxes, iou_thr):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        bx1, by1, bx2, by2 = best[:4]
        barea = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        rest = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            inter_w = max(0.0, min(bx2, x2) - max(bx1, x1))
            inter_h = max(0.0, min(by2, y2) - max(by1, y1))
            inter = inter_w * inter_h
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = barea + area - inter
            if union <= 0.0 or inter / union <= iou_thr:
                rest.append(box)
        boxes = rest
    return keep


def decode_output(
    output,
    conf_thr,
    nms_thr,
    frame_w,
    frame_h,
    scale,
    pad_x,
    pad_y,
    layout,
):
    arr = np.asarray(output)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Unsupported output shape after squeeze: {arr.shape}")

    if layout == "auto":
        if arr.shape[0] >= 5 and arr.shape[0] <= 128 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        elif arr.shape[1] >= 5:
            pass
        else:
            raise RuntimeError(f"Cannot infer output layout: {arr.shape}")
    elif layout == "cxn":
        arr = arr.T
    elif layout == "nxc":
        pass
    else:
        raise RuntimeError(f"Unknown layout: {layout}")

    boxes = []
    for row in arr:
        if row.shape[0] < 5:
            continue
        conf = float(row[4])
        if not math.isfinite(conf) or conf < conf_thr:
            continue

        a, b, c, d = map(float, row[:4])
        if c > a and d > b:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1 = a - c * 0.5
            y1 = b - d * 0.5
            x2 = a + c * 0.5
            y2 = b + d * 0.5

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1 = max(0.0, min(float(frame_w - 1), x1))
        y1 = max(0.0, min(float(frame_h - 1), y1))
        x2 = max(0.0, min(float(frame_w - 1), x2))
        y2 = max(0.0, min(float(frame_h - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2, conf])

    return nms(boxes, nms_thr)


def draw_boxes(frame, boxes):
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(
            frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"UAV {conf:.2f}",
            (int(x1), max(20, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


def lower_exposure(frame):
    return cv2.convertScaleAbs(frame, alpha=LOWER_EXPOSURE_ALPHA, beta=0.0)


def infer_frame(
    context,
    cuda,
    d_input,
    d_output,
    output_shape,
    output_np_dtype,
    frame,
    input_w,
    input_h,
    input_np_dtype,
    conf_thr,
    nms_thr,
    layout,
):
    inp, scale, pad_x, pad_y = letterbox_bgr_to_rgb_chw(
        frame, input_w, input_h, input_np_dtype
    )
    t0 = time.time()
    cuda.memcpy_h2d(d_input, inp)
    if not context.execute_async_v3(0):
        raise RuntimeError("TensorRT execute_async_v3 failed")

    out = np.empty(output_shape, dtype=output_np_dtype)
    out = cuda.memcpy_d2h(out, d_output)
    infer_ms = (time.time() - t0) * 1000.0
    boxes = decode_output(
        out,
        conf_thr,
        nms_thr,
        frame.shape[1],
        frame.shape[0],
        scale,
        pad_x,
        pad_y,
        layout,
    )
    return boxes, infer_ms


def main():
    parser = argparse.ArgumentParser(
        description="Run a TensorRT UAV engine on a video or image."
    )
    parser.add_argument("--video", required=True, help="input video or image path")
    parser.add_argument("--config", default="config/detect_params.yaml")
    parser.add_argument("--engine", default="", help="override engine path")
    parser.add_argument("--conf", type=float, default=-1.0)
    parser.add_argument("--nms", type=float, default=0.25)
    parser.add_argument("--output", default="", help="optional annotated video output")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--lower-exposure",
        action="store_true",
        help="darken frames before inference/display/output",
    )
    parser.add_argument("--layout", choices=["auto", "cxn", "nxc"], default="auto")
    args = parser.parse_args()

    cfg_engine, cfg_conf = load_config(args.config)
    engine_path = args.engine or cfg_engine
    conf_thr = args.conf if args.conf > 0 else cfg_conf
    args.stride = max(1, args.stride)

    if not os.path.isabs(engine_path):
        engine_path = os.path.abspath(engine_path)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(engine_path)

    logger = trt.Logger(trt.Logger.INFO)
    print(f"engine: {engine_path}")
    print(f"input:  {args.video}")
    print(f"conf:   {conf_thr}")
    print(f"lower_exposure: {args.lower_exposure}")

    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context failed")

    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
        print(
            f"tensor[{i}] {name} {mode} "
            f"{engine.get_tensor_dtype(name)} {engine.get_tensor_shape(name)}"
        )

    if len(input_names) != 1 or len(output_names) < 1:
        raise RuntimeError(
            f"Expected 1 input and >=1 output, got {input_names}, {output_names}"
        )

    input_name = input_names[0]
    output_name = output_names[0]
    input_shape = tuple(engine.get_tensor_shape(input_name))
    if any(d < 0 for d in input_shape):
        input_shape = (1, 3, 1280, 1280)
        context.set_input_shape(input_name, input_shape)
    if len(input_shape) != 4:
        raise RuntimeError(f"Unsupported input shape: {input_shape}")

    _, c, input_h, input_w = input_shape
    if c != 3:
        raise RuntimeError(f"Expected NCHW 3-channel input, got {input_shape}")

    input_np_dtype = tensor_dtype_to_np(engine.get_tensor_dtype(input_name))
    output_shape = tuple(context.get_tensor_shape(output_name))
    output_np_dtype = tensor_dtype_to_np(engine.get_tensor_dtype(output_name))

    cuda = CudaRuntime()
    d_input = cuda.malloc(volume(input_shape) * np.dtype(input_np_dtype).itemsize)
    d_output = cuda.malloc(volume(output_shape) * np.dtype(output_np_dtype).itemsize)

    try:
        context.set_tensor_address(input_name, int(d_input.value))
        context.set_tensor_address(output_name, int(d_output.value))

        input_ext = os.path.splitext(args.video)[1].lower()
        if input_ext in IMAGE_EXTS:
            frame = cv2.imread(args.video, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to open image: {args.video}")
            if args.lower_exposure:
                frame = lower_exposure(frame)

            boxes, infer_ms = infer_frame(
                context,
                cuda,
                d_input,
                d_output,
                output_shape,
                output_np_dtype,
                frame,
                input_w,
                input_h,
                input_np_dtype,
                conf_thr,
                args.nms,
                args.layout,
            )
            print(f"image boxes={len(boxes)} infer_ms={infer_ms:.2f}", end="")
            for x1, y1, x2, y2, conf in boxes:
                print(
                    f" [conf={conf:.3f} "
                    f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})]",
                    end="",
                )
            print()

            draw_boxes(frame, boxes)
            if args.output:
                if not cv2.imwrite(args.output, frame):
                    raise RuntimeError(f"failed to write output image: {args.output}")
            if args.show:
                cv2.imshow("uav_engine_image_test", frame)
                cv2.waitKey(0)
            print(f"summary: images=1 hit_images={1 if boxes else 0}")
            return

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {args.video}")

        writer = None
        if args.output:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open output: {args.output}")

        frame_idx = 0
        inferred = 0
        hit_frames = 0
        last_boxes = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if args.lower_exposure:
                frame = lower_exposure(frame)

            if frame_idx % args.stride == 0:
                last_boxes, infer_ms = infer_frame(
                    context,
                    cuda,
                    d_input,
                    d_output,
                    output_shape,
                    output_np_dtype,
                    frame,
                    input_w,
                    input_h,
                    input_np_dtype,
                    conf_thr,
                    args.nms,
                    args.layout,
                )

                inferred += 1
                if last_boxes:
                    hit_frames += 1
                print(
                    f"frame={frame_idx} boxes={len(last_boxes)} "
                    f"infer_ms={infer_ms:.2f}",
                    end="",
                )
                for x1, y1, x2, y2, conf in last_boxes:
                    print(
                        f" [conf={conf:.3f} "
                        f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})]",
                        end="",
                    )
                print()

            draw_boxes(frame, last_boxes)

            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("uav_engine_video_test", frame)
                key = cv2.waitKey(1)
                if key in (27, ord("q"), ord("Q")):
                    break

            frame_idx += 1

        print(
            f"summary: frames={frame_idx} inferred={inferred} "
            f"hit_frames={hit_frames}"
        )
    finally:
        cuda.free(d_input)
        cuda.free(d_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[1;31mERROR: {exc}\033[0m", file=sys.stderr)
        sys.exit(1)


# cd /home/user/code/T-DT_Radar-main

# python3 src/tdt_vision/tools/uav_engine_video_test.py \
#   --video /home/user/下载/VID_20260513_231650.mp4 \
#   --engine model/TensorRT/uav_train7_best_fp16.engine \
#   --conf 0.01 \
#   --show \
#   --lower-exposure