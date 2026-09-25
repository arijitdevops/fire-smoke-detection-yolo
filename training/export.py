"""Export trained YOLO weights to a deployment format.

Supported targets are the Ultralytics formats this project has been exercised
with: ``onnx``, ``engine`` (TensorRT), ``torchscript`` and ``openvino``.  Export
backends are optional dependencies, so a missing backend is reported as a clear,
actionable message rather than a traceback.

Example::

    python training/export.py --weights runs/detect/fire_smoke/weights/best.pt --format onnx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Final, Sequence

LOGGER: Final = logging.getLogger("export")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Export format -> the pip extra that provides it.
BACKEND_HINTS: Final[dict[str, str]] = {
    "onnx": "pip install onnx onnxruntime onnxslim",
    "engine": "install TensorRT (NVIDIA GPU + 'pip install tensorrt') and export on the target GPU",
    "torchscript": "torchscript ships with torch; check your torch installation",
    "openvino": "pip install openvino",
}


class ExportError(RuntimeError):
    """Raised when export cannot be completed."""


def build_export_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Assemble ``YOLO.export`` arguments, passing ONNX-only options only to ONNX.

    Recent Ultralytics releases reject arguments a format does not understand
    (e.g. ``opset`` for TorchScript), so they are added conditionally.
    """
    kwargs: dict[str, object] = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "device": args.device,
        "batch": args.batch,
    }
    if args.format == "onnx":
        kwargs.update(dynamic=args.dynamic, simplify=args.simplify, opset=args.opset)
    elif args.format == "openvino":
        kwargs.update(dynamic=args.dynamic)
    return kwargs


def export_weights(args: argparse.Namespace) -> Path:
    """Export ``args.weights`` to ``args.format`` and return the produced file.

    Raises:
        ExportError: If the weights are missing, Ultralytics is absent, or the
            export backend is unavailable.
    """
    weights = Path(args.weights).expanduser()
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    if not weights.is_file():
        raise ExportError(
            f"weights not found: {weights}. Train first with 'python training/train.py'."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExportError("ultralytics is not installed; pip install ultralytics") from exc

    LOGGER.info("exporting %s to %s", weights.name, args.format)
    try:
        model = YOLO(str(weights))
        exported = model.export(**build_export_kwargs(args))
    except ImportError as exc:
        hint = BACKEND_HINTS.get(args.format, "install the matching export backend")
        raise ExportError(
            f"export backend for '{args.format}' is unavailable ({exc}). Try: {hint}"
        ) from exc
    except Exception as exc:
        hint = BACKEND_HINTS.get(args.format, "")
        suffix = f" Hint: {hint}" if hint else ""
        raise ExportError(f"export to '{args.format}' failed: {exc}.{suffix}") from exc

    produced = Path(str(exported))
    if not produced.exists():
        raise ExportError(
            f"Ultralytics reported success but {produced} does not exist; "
            "check the export log above"
        )
    LOGGER.info("wrote %s (%.1f MB)", produced, produced.stat().st_size / (1024 * 1024))
    return produced


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Export trained YOLO weights.")
    parser.add_argument(
        "--weights",
        default=os.environ.get("MODEL_PATH", "runs/detect/fire_smoke/weights/best.pt"),
        help="Checkpoint to export (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        default="onnx",
        choices=tuple(BACKEND_HINTS),
        help="Export format (default: %(default)s).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (default: %(default)s).")
    parser.add_argument("--batch", type=int, default=1, help="Static batch size (default: %(default)s).")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: %(default)s).")
    parser.add_argument("--half", action="store_true", help="Export in FP16 (GPU targets only).")
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic input shapes (ONNX).")
    parser.add_argument("--simplify", action="store_true", help="Run the ONNX graph simplifier.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""), help="Torch device (default: auto).")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Logging level.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    if args.half and args.format == "onnx" and not args.device:
        LOGGER.warning("--half with ONNX requires a CUDA device; pass --device 0")
    try:
        export_weights(args)
    except ExportError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
