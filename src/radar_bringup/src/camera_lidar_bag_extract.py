#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path

import rosbag2_py


DEFAULT_CAMERA_TOPIC = "/camera0/image"
DEFAULT_LIDAR_TOPIC = "/livox/lidar_integrated"
SERIALIZATION_FORMAT = "cdr"
NS_PER_SEC = 1_000_000_000


def _expand_path(path_text):
    return Path(path_text).expanduser().resolve()


def _resolve_input_bag_dir(path_text):
    path = _expand_path(path_text)
    if path.is_file():
        if path.suffix == ".db3":
            path = path.parent
        else:
            raise RuntimeError(f"input path is a file but not a .db3 file: {path}")

    if not path.is_dir():
        raise RuntimeError(f"input bag directory does not exist: {path}")

    metadata = path / "metadata.yaml"
    if not metadata.is_file():
        db3_files = sorted(path.glob("*.db3"))
        if db3_files:
            raise RuntimeError(
                f"found db3 files in {path}, but metadata.yaml is missing; "
                "run `ros2 bag reindex <bag_dir>` first"
            )
        raise RuntimeError(
            f"{path} is not a rosbag2 directory; expected metadata.yaml inside it"
        )

    return path


def _default_output_bag_dir(input_bag_dir):
    return input_bag_dir.with_name(f"{input_bag_dir.name}_camera_lidar")


def _alternate_topic_names(topic_name):
    stripped = topic_name.strip()
    if not stripped:
        return []

    names = [stripped]
    without_slash = stripped.lstrip("/")
    with_slash = f"/{without_slash}"
    for name in (with_slash, without_slash):
        if name not in names:
            names.append(name)
    return names


def _select_available_topics(available_topics, requested_topics):
    selected = []
    missing = []

    for requested in requested_topics:
        match = None
        for candidate in _alternate_topic_names(requested):
            if candidate in available_topics:
                match = candidate
                break
        if match is None:
            missing.append(requested)
        elif match not in selected:
            selected.append(match)

    return selected, missing


def _topic_metadata_copy(topic_metadata):
    return rosbag2_py.TopicMetadata(
        name=topic_metadata.name,
        type=topic_metadata.type,
        serialization_format=getattr(
            topic_metadata, "serialization_format", SERIALIZATION_FORMAT
        ) or SERIALIZATION_FORMAT,
        offered_qos_profiles=getattr(topic_metadata, "offered_qos_profiles", ""),
    )


def _seconds_to_nanoseconds(seconds):
    return int(round(seconds * NS_PER_SEC))


def _timestamp_to_nanoseconds(timestamp):
    if isinstance(timestamp, int):
        return timestamp
    if hasattr(timestamp, "nanoseconds"):
        return int(timestamp.nanoseconds)
    if hasattr(timestamp, "timestamp"):
        return int(timestamp.timestamp() * NS_PER_SEC)
    raise TypeError(f"unsupported timestamp type: {type(timestamp)!r}")


def _read_bag_start_timestamp_ns(input_bag_dir, storage_id):
    try:
        metadata = rosbag2_py.Info().read_metadata(str(input_bag_dir), storage_id)
        return _timestamp_to_nanoseconds(metadata.starting_time)
    except Exception as exc:
        print(
            "warning: failed to read bag starting_time from metadata; "
            f"using first selected message as t=0: {exc}",
            file=sys.stderr,
        )
        return None


def _normalize_time_bounds(start_sec, end_sec):
    start_sec = float(start_sec)
    end_sec = float(end_sec)

    if start_sec < 0.0:
        raise RuntimeError("--start-sec must be >= 0")

    if end_sec <= 0.0:
        return start_sec, None

    if end_sec <= start_sec:
        raise RuntimeError("--end-sec must be greater than --start-sec, or <= 0 for bag end")

    return start_sec, end_sec


def convert_bag(
    input_bag_dir,
    output_bag_dir,
    requested_topics,
    input_storage_id,
    output_storage_id,
    overwrite,
    allow_missing,
    start_sec,
    end_sec,
):
    input_bag_dir = _resolve_input_bag_dir(input_bag_dir)
    start_sec, end_sec = _normalize_time_bounds(start_sec, end_sec)
    if output_bag_dir:
        output_bag_dir = _expand_path(output_bag_dir)
    else:
        output_bag_dir = _default_output_bag_dir(input_bag_dir)

    if output_bag_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"output bag directory already exists: {output_bag_dir}; "
                "use --overwrite to replace it"
            )
        if output_bag_dir == input_bag_dir:
            raise RuntimeError("output bag directory must not equal input bag directory")
        shutil.rmtree(output_bag_dir)

    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=SERIALIZATION_FORMAT,
        output_serialization_format=SERIALIZATION_FORMAT,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag_dir), storage_id=input_storage_id),
        converter_options,
    )

    available_metadata = {
        topic.name: topic for topic in reader.get_all_topics_and_types()
    }
    selected_topics, missing_topics = _select_available_topics(
        available_metadata.keys(), requested_topics
    )

    if missing_topics and not allow_missing:
        available = "\n  ".join(sorted(available_metadata.keys()))
        raise RuntimeError(
            "input bag is missing required topic(s): "
            + ", ".join(missing_topics)
            + "\navailable topics:\n  "
            + available
        )
    if not selected_topics:
        raise RuntimeError("no requested topics are available in the input bag")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag_dir), storage_id=output_storage_id),
        converter_options,
    )

    for topic_name in selected_topics:
        writer.create_topic(_topic_metadata_copy(available_metadata[topic_name]))

    reader.set_filter(rosbag2_py.StorageFilter(topics=selected_topics))

    bag_start_timestamp_ns = _read_bag_start_timestamp_ns(input_bag_dir, input_storage_id)
    segment_start_ns = None
    segment_end_ns = None
    if bag_start_timestamp_ns is not None:
        segment_start_ns = bag_start_timestamp_ns + _seconds_to_nanoseconds(start_sec)
        if end_sec is not None:
            segment_end_ns = bag_start_timestamp_ns + _seconds_to_nanoseconds(end_sec)

    counts = {topic_name: 0 for topic_name in selected_topics}
    skipped_before = 0
    total_written = 0
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name not in counts:
            continue
        if segment_start_ns is None:
            bag_start_timestamp_ns = timestamp
            segment_start_ns = bag_start_timestamp_ns + _seconds_to_nanoseconds(start_sec)
            if end_sec is not None:
                segment_end_ns = bag_start_timestamp_ns + _seconds_to_nanoseconds(end_sec)
        if timestamp < segment_start_ns:
            skipped_before += 1
            continue
        if segment_end_ns is not None and timestamp > segment_end_ns:
            break
        writer.write(topic_name, data, timestamp)
        counts[topic_name] += 1
        total_written += 1
        if total_written % 5000 == 0:
            print(f"written {total_written} messages...", flush=True)

    print(f"input bag: {input_bag_dir}")
    print(f"output bag: {output_bag_dir}")
    if end_sec is None:
        print(f"time window: {start_sec:.3f}s -> end")
    else:
        print(f"time window: {start_sec:.3f}s -> {end_sec:.3f}s")
    for topic_name in selected_topics:
        print(f"{topic_name}: {counts[topic_name]} messages")
    if missing_topics:
        print("missing topics skipped: " + ", ".join(missing_topics))
    print(f"skipped before window: {skipped_before}")
    print(f"total written: {total_written}")

    return 0 if total_written > 0 else 2


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Extract camera0 image and integrated Livox lidar topics from a rosbag2 bag."
        )
    )
    parser.add_argument(
        "--input-bag-dir",
        required=True,
        help="input rosbag2 directory, or a .db3 file inside that directory",
    )
    parser.add_argument(
        "--output-bag-dir",
        default="",
        help="output rosbag2 directory; default is <input>_camera_lidar",
    )
    parser.add_argument(
        "--camera-topic",
        default=DEFAULT_CAMERA_TOPIC,
        help=f"camera topic to extract, default: {DEFAULT_CAMERA_TOPIC}",
    )
    parser.add_argument(
        "--lidar-topic",
        default=DEFAULT_LIDAR_TOPIC,
        help=f"lidar topic to extract, default: {DEFAULT_LIDAR_TOPIC}",
    )
    parser.add_argument(
        "--input-storage-id",
        default="sqlite3",
        help="input rosbag2 storage id, default: sqlite3",
    )
    parser.add_argument(
        "--output-storage-id",
        default="sqlite3",
        help="output rosbag2 storage id, default: sqlite3",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output bag directory if it already exists",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write available requested topics instead of failing on missing topics",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="relative start time in seconds from bag beginning, default: 0",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=-1.0,
        help="relative end time in seconds from bag beginning; <=0 means bag end",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return convert_bag(
            input_bag_dir=args.input_bag_dir,
            output_bag_dir=args.output_bag_dir,
            requested_topics=[args.camera_topic, args.lidar_topic],
            input_storage_id=args.input_storage_id,
            output_storage_id=args.output_storage_id,
            overwrite=args.overwrite,
            allow_missing=args.allow_missing,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        )
    except Exception as exc:
        print(f"\033[1;31mcamera_lidar_bag_extract failed: {exc}\033[0m", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
