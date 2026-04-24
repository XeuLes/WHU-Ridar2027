import argparse
import subprocess
from pathlib import Path


def make_nchw_shape(batch: str, hw_shape: str) -> str:
    return f"{batch}x3x{hw_shape}"


def build_trtexec_command(args: argparse.Namespace) -> list[str]:
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.saveEngine}",
    ]

    if args.static:
        infer_shape = args.optShape or make_nchw_shape(args.optBatch, args.Shape)
        command.append(f"--shapes={args.input_name}:{infer_shape}")
    else:
        min_shape = args.minShape or make_nchw_shape(args.minBatch, args.Shape)
        opt_shape = args.optShape or make_nchw_shape(args.optBatch, args.Shape)
        max_shape = args.maxShape or make_nchw_shape(args.maxBatch, args.Shape)
        command.extend(
            [
                f"--minShapes={args.input_name}:{min_shape}",
                f"--optShapes={args.input_name}:{opt_shape}",
                f"--maxShapes={args.input_name}:{max_shape}",
            ]
        )

    if args.fp16:
        command.append("--fp16")

    command.extend(args.extraArg)
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from ONNX with optional dynamic profiles."
    )
    parser.add_argument("--onnx", default="", help="Path to the ONNX file.")
    parser.add_argument("--saveEngine", default="", help="Path to save the TensorRT engine file.")
    parser.add_argument("--Shape", default="224x224", help="Input HxW, for example 1280x1280.")
    parser.add_argument("--minBatch", default="1", help="Minimum batch size for dynamic profile.")
    parser.add_argument("--optBatch", default="10", help="Optimal batch size for dynamic profile.")
    parser.add_argument("--maxBatch", default="20", help="Maximum batch size for dynamic profile.")
    parser.add_argument("--input_name", default="input", help="Name of the input tensor.")
    parser.add_argument("--minShape", default="", help="Explicit min NCHW shape, overrides --minBatch/--Shape.")
    parser.add_argument("--optShape", default="", help="Explicit opt NCHW shape, overrides --optBatch/--Shape.")
    parser.add_argument("--maxShape", default="", help="Explicit max NCHW shape, overrides --maxBatch/--Shape.")
    parser.add_argument("--static", action="store_true", help="Build a static-shape engine and skip min/opt/max profiles.")
    parser.add_argument("--fp16", action="store_true", help="Enable TensorRT FP16 build.")
    parser.add_argument(
        "--extraArg",
        action="append",
        default=[],
        help="Extra trtexec argument, may be repeated.",
    )
    args = parser.parse_args()

    save_engine = Path(args.saveEngine)
    save_engine.parent.mkdir(parents=True, exist_ok=True)

    command = build_trtexec_command(args)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)
