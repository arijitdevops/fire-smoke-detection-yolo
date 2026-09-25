"""Train a YOLO fire/smoke detector with Ultralytics.

This is a thin, explicit wrapper around ``ultralytics.YOLO.train`` so that
training runs are reproducible from the command line and from CI.  It always
trains against the descriptor produced by :mod:`prepare_data`, which is
regenerated automatically when it is missing or stale.

Examples::

    python training/train.py --model yolo11n --epochs 100 --imgsz 640
    python training/train.py --model yolov8s --batch 32 --device 0 --name run2
    python training/train.py --resume --name run2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Final, Sequence

from prepare_data import DEFAULT_DATA_DIR, DEFAULT_OUTPUT, DatasetError, _resolve_data_dir, prepare

LOGGER: Final = logging.getLogger("train")

#: Model identifiers Ultralytics can download pretrained COCO weights for.
SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolov8n",
    "yolov8s",
    "yolov8m",
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


class TrainingError(RuntimeError):
    """Raised when training cannot start or the Ultralytics call fails."""


def _points_at(data_yaml: Path, data_dir: str) -> bool:
    """Return ``True`` when ``data_yaml``'s ``path`` key refers to ``data_dir``."""
    try:
        wanted = _resolve_data_dir(data_dir).as_posix()
    except DatasetError:
        return True  # let the existing descriptor be used; training will report problems
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            return line.split(":", 1)[1].strip() == wanted
    return False


def ensure_data_yaml(data_dir: str, data_yaml: str, regenerate: bool) -> Path:
    """Return a usable ``data.yaml`` path, generating it when needed.

    Args:
        data_dir: Dataset root passed through to :func:`prepare_data.prepare`.
        data_yaml: Destination descriptor path.
        regenerate: Force regeneration even if the file already exists.

    Raises:
        TrainingError: If the descriptor cannot be produced.
    """
    output = Path(data_yaml).expanduser()
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    if output.is_file() and not regenerate and _points_at(output, data_dir):
        LOGGER.info("using existing dataset descriptor %s", output)
        return output
    LOGGER.info("generating dataset descriptor at %s", output)
    try:
        return prepare(_resolve_data_dir(data_dir), output)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc


def load_overrides(config_path: str | None) -> dict[str, Any]:
    """Load hyper-parameter overrides from a YAML file.

    Args:
        config_path: Path to a YAML mapping, or ``None``.

    Returns:
        A dictionary of Ultralytics training arguments (empty when no config).

    Raises:
        TrainingError: If the file is missing, unreadable or not a mapping.
    """
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise TrainingError(f"config file not found: {path}")
    try:
        import yaml  # provided by the ultralytics dependency chain
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TrainingError("PyYAML is required to read --config files") from exc
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingError(f"could not parse {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TrainingError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    LOGGER.info("loaded %d hyper-parameter override(s) from %s", len(loaded), path)
    return dict(loaded)


def resolve_project(project: str) -> Path:
    """Return ``project`` as an absolute path (relative paths are anchored at the repo root).

    Ultralytics nests *relative* ``project`` values under its own ``runs/<task>``
    directory, which would produce ``runs/detect/runs/detect/<name>``.  Passing an
    absolute path keeps artefacts exactly where the README says they are.
    """
    path = Path(project).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def build_train_kwargs(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    """Assemble the keyword arguments handed to ``YOLO.train``."""
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(resolve_project(args.project)),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "exist_ok": True,
        "resume": args.resume,
        "pretrained": not args.scratch,
        "val": True,
        "plots": True,
    }
    kwargs.update(load_overrides(args.config))
    return kwargs


def train(args: argparse.Namespace) -> Path:
    """Run a training session and return the directory holding its artefacts.

    Raises:
        TrainingError: If Ultralytics is unavailable or training fails.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TrainingError(
            "ultralytics is not installed; run 'pip install -r backend/requirements.txt' "
            "or 'pip install ultralytics'"
        ) from exc

    data_yaml = ensure_data_yaml(args.data_dir, args.data_yaml, args.regenerate_data_yaml)

    if args.resume:
        checkpoint = resolve_project(args.project) / args.name / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise TrainingError(
                f"--resume requested but no checkpoint at {checkpoint}; "
                "start a fresh run or pass the correct --project/--name"
            )
        weights = str(checkpoint)
        LOGGER.info("resuming from %s", weights)
    else:
        weights = f"{args.model}.pt"
        LOGGER.info("starting from pretrained COCO weights %s", weights)

    try:
        model = YOLO(weights)
        results = model.train(**build_train_kwargs(args, data_yaml))
    except Exception as exc:  # Ultralytics raises a wide variety of errors
        raise TrainingError(f"training failed: {exc}") from exc

    save_dir = Path(getattr(results, "save_dir", resolve_project(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    LOGGER.info("training finished; artefacts in %s", save_dir)
    if best.is_file():
        LOGGER.info("best weights: %s", best)
        LOGGER.info("point the API at them with MODEL_PATH=%s", best)
    return save_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Train a YOLO fire/smoke detector.")
    parser.add_argument(
        "--model",
        default="yolo11n",
        choices=SUPPORTED_MODELS,
        help="Pretrained checkpoint to fine-tune (default: %(default)s).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: %(default)s).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: %(default)s).")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; -1 lets Ultralytics auto-size it (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", ""),
        help="Torch device: '' (auto), 'cpu', '0', '0,1' (default: auto).",
    )
    parser.add_argument("--project", default="runs/detect", help="Output root (default: %(default)s).")
    parser.add_argument("--name", default="fire_smoke", help="Run name (default: %(default)s).")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience in epochs; 0 disables it (default: %(default)s).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: %(default)s).")
    parser.add_argument("--resume", action="store_true", help="Resume the run named by --project/--name.")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train from randomly initialised weights instead of the COCO checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML file of extra Ultralytics hyper-parameters, e.g. training/configs/yolo11n.yaml.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--data-yaml",
        default=os.environ.get("DATA_YAML", DEFAULT_OUTPUT),
        help="Generated dataset descriptor (default: %(default)s).",
    )
    parser.add_argument(
        "--regenerate-data-yaml",
        action="store_true",
        help="Re-run dataset validation even if the descriptor already exists.",
    )
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
        train(args)
    except TrainingError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
