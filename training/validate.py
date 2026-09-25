"""Evaluate trained weights and write ``reports/metrics.json``.

The JSON written here is consumed by nothing but humans and CI; it deliberately
records the exact weights, split and thresholds used so that a number can never
be quoted without its provenance.

Example::

    python training/validate.py --weights runs/detect/fire_smoke/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from prepare_data import CLASS_NAMES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT, DatasetError
from prepare_data import _resolve_data_dir, prepare

LOGGER: Final = logging.getLogger("validate")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


class ValidationError(RuntimeError):
    """Raised when evaluation cannot run."""


def _as_float(value: Any) -> float | None:
    """Coerce a numpy scalar / tensor / python number to ``float``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return float(item())
            except (TypeError, ValueError):
                return None
        return None


def _per_class_metrics(results: Any) -> dict[str, dict[str, float | None]]:
    """Extract per-class precision/recall/mAP from an Ultralytics result object."""
    box = getattr(results, "box", None)
    per_class: dict[str, dict[str, float | None]] = {}
    if box is None:
        return per_class
    for index, name in enumerate(CLASS_NAMES):
        entry: dict[str, float | None] = {}
        for key, attribute in (
            ("precision", "p"),
            ("recall", "r"),
            ("map50", "ap50"),
            ("map50_95", "ap"),
        ):
            series = getattr(box, attribute, None)
            try:
                entry[key] = _as_float(series[index]) if series is not None else None
            except (IndexError, TypeError):
                entry[key] = None
        per_class[name] = entry
    return per_class


def _ensure_data_yaml(data_dir: str, data_yaml: str) -> Path:
    """Return the dataset descriptor, generating it if it is absent."""
    output = Path(data_yaml).expanduser()
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    if output.is_file():
        return output
    try:
        return prepare(_resolve_data_dir(data_dir), output)
    except DatasetError as exc:
        raise ValidationError(str(exc)) from exc


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run ``model.val()`` and return a JSON-serialisable metrics payload.

    Raises:
        ValidationError: If the weights are missing or Ultralytics fails.
    """
    weights = Path(args.weights).expanduser()
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    if not weights.is_file():
        raise ValidationError(
            f"weights not found: {weights}. Train first with 'python training/train.py'."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValidationError("ultralytics is not installed; pip install ultralytics") from exc

    data_yaml = _ensure_data_yaml(args.data_dir, args.data_yaml)
    LOGGER.info("evaluating %s on split=%s", weights.name, args.split)

    try:
        model = YOLO(str(weights))
        results = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            plots=args.plots,
            project=str(REPO_ROOT / "runs" / "detect"),
            name=f"val_{args.split}",
            exist_ok=True,
            verbose=True,
        )
    except Exception as exc:
        raise ValidationError(f"evaluation failed: {exc}") from exc

    box = getattr(results, "box", None)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "class_names": list(CLASS_NAMES),
        "overall": {
            "precision": _as_float(getattr(box, "mp", None)),
            "recall": _as_float(getattr(box, "mr", None)),
            "map50": _as_float(getattr(box, "map50", None)),
            "map50_95": _as_float(getattr(box, "map", None)),
        },
        "per_class": _per_class_metrics(results),
        "speed_ms": {
            key: _as_float(value)
            for key, value in (getattr(results, "speed", {}) or {}).items()
        },
    }
    return payload


def write_metrics(payload: dict[str, Any], output: Path) -> Path:
    """Write the metrics payload as pretty-printed JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"could not write {output}: {exc}") from exc
    LOGGER.info("wrote metrics to %s", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Validate trained YOLO weights and dump metrics.")
    parser.add_argument(
        "--weights",
        default=os.environ.get("MODEL_PATH", "runs/detect/fire_smoke/weights/best.pt"),
        help="Checkpoint to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate (default: %(default)s).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: %(default)s).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: %(default)s).")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: %(default)s).")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (default: %(default)s).")
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""), help="Torch device (default: auto).")
    parser.add_argument("--plots", action="store_true", help="Also write confusion-matrix / PR plots.")
    parser.add_argument("--output", default="reports/metrics.json", help="Metrics destination (default: %(default)s).")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR), help="Dataset root.")
    parser.add_argument("--data-yaml", default=os.environ.get("DATA_YAML", DEFAULT_OUTPUT), help="Dataset descriptor.")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Logging level.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        payload = evaluate(args)
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = (REPO_ROOT / output).resolve()
        write_metrics(payload, output)
    except ValidationError as exc:
        LOGGER.error("%s", exc)
        return 1
    overall = payload["overall"]
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
