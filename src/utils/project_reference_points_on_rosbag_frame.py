#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def load_opencv_matrix_yaml(path):
    try:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            try:
                world_rvec = fs.getNode("world_rvec").mat()
                world_tvec = fs.getNode("world_tvec").mat()
            finally:
                fs.release()

            if world_rvec is not None and world_tvec is not None:
                return (
                    world_rvec.reshape(3, 1).astype(np.float64),
                    world_tvec.reshape(3, 1).astype(np.float64),
                )
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("%YAML:")
    )
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "world_rvec" not in data or "world_tvec" not in data:
        raise RuntimeError(f"missing world_rvec/world_tvec in {path}")

    return (
        np.asarray(data["world_rvec"], dtype=np.float64).reshape(3, 1),
        np.asarray(data["world_tvec"], dtype=np.float64).reshape(3, 1),
    )


def load_camera_params(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open yaml: {path}")

    try:
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        if dist_coeffs is None:
            dist_coeffs = fs.getNode("dist_coeff").mat()
    finally:
        fs.release()

    if camera_matrix is None:
        raise RuntimeError(f"missing camera_matrix in {path}")
    if dist_coeffs is None:
        dist_coeffs = np.zeros((1, 5), dtype=np.float64)

    return camera_matrix.astype(np.float64), dist_coeffs.reshape(1, -1).astype(np.float64)


def load_reference_points(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # PyYAML does not accept OpenCV's "%YAML:1.0" directive.
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("%YAML:")
    )
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "reference_points" not in data:
        raise RuntimeError(f"missing reference_points in {path}")

    names = []
    points = []
    for i, item in enumerate(data["reference_points"]):
        if not isinstance(item, dict) or "name" not in item or "pos" not in item:
            raise RuntimeError(f"invalid reference_points[{i}] in {path}")
        pos = item["pos"]
        if len(pos) != 3:
            raise RuntimeError(f"reference_points[{i}].pos must have 3 values")
        names.append(str(item["name"]))
        points.append([float(pos[0]), float(pos[1]), float(pos[2])])

    return names, np.asarray(points, dtype=np.float32).reshape(-1, 1, 3)


def open_reader(rosbag_path):
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def read_compressed_frame(rosbag_path, topic, frame_index):
    reader = open_reader(rosbag_path)
    seen = 0

    while reader.has_next():
        bag_topic, data, timestamp = reader.read_next()
        if bag_topic != topic:
            continue
        if seen != frame_index:
            seen += 1
            continue

        msg = deserialize_message(data, CompressedImage)
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {topic} frame_index={frame_index}")
        return image, timestamp

    raise RuntimeError(
        f"could not find frame_index={frame_index} on topic {topic}; found {seen} matching frames"
    )


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def draw_points(image, names, image_points):
    output = image.copy()
    h, w = output.shape[:2]

    for name, point in zip(names, image_points.reshape(-1, 2)):
        x, y = float(point[0]), float(point[1])
        inside = 0 <= x < w and 0 <= y < h
        color = (0, 255, 0) if inside else (0, 0, 255)
        center = (int(round(x)), int(round(y)))

        cv2.drawMarker(
            output,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(output, center, 6, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            output,
            name,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project reference_points_RM.yaml world points onto a static rosbag frame "
            "using out_matrix.yaml and camera_params.yaml."
        )
    )
    parser.add_argument("--rosbag", help="rosbag path, db3 file or bag directory")
    parser.add_argument("--image", help="static image path; skips rosbag reading when set")
    parser.add_argument("--topic", default="/compressed_image", help="compressed image topic in the bag")
    parser.add_argument("--frame-index", type=int, default=0, help="0-based compressed image frame index")
    parser.add_argument("--out-matrix", default="config/out_matrix.yaml", help="out_matrix yaml path")
    parser.add_argument("--camera-params", default="config/camera_params.yaml", help="camera params yaml path")
    parser.add_argument(
        "--reference-points",
        default="src/camera_field_calib_tool/config/reference_points_RM.yaml",
        help="reference_points_RM yaml path",
    )
    parser.add_argument(
        "--output",
        default="/tmp/reference_points_projection.png",
        help="output debug image path",
    )
    parser.add_argument("--show", action="store_true", help="show result with cv2.imshow")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_index < 0:
        raise RuntimeError("--frame-index must be >= 0")
    if not args.image and not args.rosbag:
        raise RuntimeError("provide either --image or --rosbag")

    rvec, tvec = load_opencv_matrix_yaml(args.out_matrix)
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    names, world_points = load_reference_points(args.reference_points)
    if args.image:
        frame = read_image(args.image)
        timestamp = "image-file"
    else:
        frame, timestamp = read_compressed_frame(args.rosbag, args.topic, args.frame_index)

    image_points, _ = cv2.projectPoints(
        world_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    output = draw_points(frame, names, image_points)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(args.output, output):
        raise RuntimeError(f"failed to write output image: {args.output}")

    if args.image:
        print(f"image: {args.image}")
    else:
        print(f"rosbag: {args.rosbag}")
        print(f"topic: {args.topic}")
        print(f"frame_index: {args.frame_index}")
    print(f"timestamp: {timestamp}")
    print(f"out_matrix: {args.out_matrix}")
    print(f"camera_params: {args.camera_params}")
    print(f"reference_points: {args.reference_points}")
    print(f"output: {args.output}")
    print("projected pixels:")
    h, w = frame.shape[:2]
    for name, point in zip(names, image_points.reshape(-1, 2)):
        x, y = float(point[0]), float(point[1])
        inside = 0 <= x < w and 0 <= y < h
        print(f"  {name}: ({x:.2f}, {y:.2f}) inside={inside}")

    if args.show:
        cv2.imshow("reference point projection", output)
        cv2.waitKey(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
