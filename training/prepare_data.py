"""Validate the D-Fire style dataset and generate a portable ``data.yaml``.

The dataset that ships with this project contains a ``data.yaml`` whose ``path``
key points at ``/kaggle/working/D Fire Dataset`` -- an absolute path from the
machine the dataset was originally exported on.  Ultralytics resolves the
``train``/``val``/``test`` keys relative to that ``path``, so training fails
immediately on any other machine.

This module regenerates a correct ``data.yaml`` from the ``DATA_DIR``
environment variable (or ``--data-dir``), verifies that every split has a
matching number of images and label files, and prints a class-frequency
histogram built by parsing the YOLO label files.

Usage::

    python training/prepare_data.py --data-dir ../_datasets/smoke_fire_detection
    python training/prepare_data.py --output training/data.generated.yaml --strict
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

LOGGER: Final = logging.getLogger("prepare_data")

#: Class index -> class name.  Index 0 is ``smoke``, index 1 is ``fire``.
CLASS_NAMES: Final[tuple[str, ...]] = ("smoke", "fire")

#: Split directory names, in the order they are reported.
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

#: Image extensions Ultralytics accepts and that this dataset uses.
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

DEFAULT_DATA_DIR: Final[str] = "../_datasets/smoke_fire_detection"
DEFAULT_OUTPUT: Final[str] = "training/data.generated.yaml"


class DatasetError(RuntimeError):
    """Raised when the dataset layout is missing or internally inconsistent."""


@dataclass(slots=True)
class SplitReport:
    """Validation result for a single dataset split."""

    name: str
    images_dir: Path
    labels_dir: Path
    image_count: int = 0
    label_count: int = 0
    empty_label_count: int = 0
    box_count: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        """``True`` when every image has a label file and nothing is malformed."""
        return not (self.missing_labels or self.orphan_labels or self.malformed)


def _resolve_data_dir(raw: str) -> Path:
    """Return an existing, absolute dataset root.

    Args:
        raw: Path as given on the command line or in ``DATA_DIR``.  Relative
            paths are resolved against the repository root (the parent of the
            ``training`` directory) so that the documented default works from
            anywhere.

    Raises:
        DatasetError: If the directory does not exist.
    """
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise DatasetError(
            f"dataset root not found: {candidate}. "
            "Set DATA_DIR or pass --data-dir to point at the extracted dataset."
        )
    return candidate


def _iter_images(images_dir: Path) -> Iterable[Path]:
    """Yield image files inside ``images_dir``, sorted for deterministic output."""
    for entry in sorted(images_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
            yield entry


def parse_label_file(path: Path) -> tuple[list[int], list[str]]:
    """Parse one YOLO label file.

    Each non-empty line must be ``class_index cx cy w h`` with normalised
    coordinates in ``[0, 1]``.  Empty files are legal: they mark background
    images, of which this dataset has many.

    Args:
        path: Label file to read.

    Returns:
        A tuple of ``(class_indices, problems)`` where ``problems`` describes
        every line that could not be parsed.
    """
    class_indices: list[int] = []
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # unreadable file, permissions, truncated mount
        return [], [f"{path.name}: cannot read ({exc})"]

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            problems.append(f"{path.name}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_index = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append(f"{path.name}:{lineno}: non-numeric field in {stripped!r}")
            continue
        if not 0 <= class_index < len(CLASS_NAMES):
            problems.append(f"{path.name}:{lineno}: class index {class_index} out of range")
            continue
        if any(not 0.0 <= value <= 1.0 for value in coords):
            problems.append(f"{path.name}:{lineno}: coordinates outside [0, 1]: {coords}")
            continue
        class_indices.append(class_index)
    return class_indices, problems


def validate_split(data_dir: Path, split: str) -> SplitReport:
    """Validate one split and collect its class frequencies.

    Args:
        data_dir: Dataset root that contains ``data/<split>/images``.
        split: One of :data:`SPLITS`.

    Raises:
        DatasetError: If the split's ``images`` or ``labels`` directory is absent.
    """
    images_dir = data_dir / "data" / split / "images"
    labels_dir = data_dir / "data" / split / "labels"
    for directory in (images_dir, labels_dir):
        if not directory.is_dir():
            raise DatasetError(f"missing directory for split {split!r}: {directory}")

    report = SplitReport(name=split, images_dir=images_dir, labels_dir=labels_dir)
    label_stems: set[str] = {
        entry.stem for entry in labels_dir.iterdir() if entry.suffix == ".txt"
    }
    report.label_count = len(label_stems)

    seen_stems: set[str] = set()
    for image_path in _iter_images(images_dir):
        report.image_count += 1
        seen_stems.add(image_path.stem)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            report.missing_labels.append(image_path.name)
            continue
        class_indices, problems = parse_label_file(label_path)
        report.malformed.extend(problems)
        if not class_indices and not problems:
            report.empty_label_count += 1
        report.box_count += len(class_indices)
        report.class_counts.update(class_indices)

    report.orphan_labels = sorted(label_stems - seen_stems)
    return report


def build_data_yaml(data_dir: Path, reports: Sequence[SplitReport]) -> str:
    """Render the Ultralytics dataset descriptor as YAML text.

    The ``path`` key is written as an absolute POSIX-style path so the file is
    valid on Windows (where ``C:/...`` works fine) and on Linux alike.
    """
    counts = {report.name: report.image_count for report in reports}
    names_block = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)
    )
    return (
        "# Generated by training/prepare_data.py -- do not edit by hand.\n"
        "# Regenerate after moving the dataset:\n"
        "#   python training/prepare_data.py --data-dir <path>\n"
        f"path: {data_dir.as_posix()}\n"
        "train: data/train/images\n"
        "val: data/val/images\n"
        "test: data/test/images\n"
        "\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
        "\n"
        "# Image counts observed at generation time (informational only).\n"
        f"train_count: {counts.get('train', 0)}\n"
        f"val_count: {counts.get('val', 0)}\n"
        f"test_count: {counts.get('test', 0)}\n"
    )


def _histogram_line(label: str, count: int, total: int, width: int = 40) -> str:
    """Return one ``label | bar | count (pct)`` row for the console histogram."""
    share = (count / total) if total else 0.0
    filled = int(round(share * width))
    bar = "#" * filled + "." * (width - filled)
    return f"  {label:<14} {bar} {count:>7,d}  ({share * 100:5.1f}%)"


def print_report(reports: Sequence[SplitReport]) -> None:
    """Print per-split counts and a class-frequency histogram."""
    print("\nDataset validation")
    print("=" * 72)
    header = f"  {'split':<8}{'images':>10}{'labels':>10}{'boxes':>10}{'background':>12}"
    print(header)
    for report in reports:
        print(
            f"  {report.name:<8}{report.image_count:>10,d}{report.label_count:>10,d}"
            f"{report.box_count:>10,d}{report.empty_label_count:>12,d}"
        )

    print("\nClass frequency (boxes)")
    print("=" * 72)
    grand_total = sum(report.box_count for report in reports)
    for report in reports:
        print(f"\n{report.name} -- {report.box_count:,d} boxes")
        if report.box_count == 0:
            print("  (no annotated objects)")
            continue
        for index, name in enumerate(CLASS_NAMES):
            print(_histogram_line(name, report.class_counts[index], report.box_count))
    print(f"\nTotal annotated boxes across all splits: {grand_total:,d}")


def _log_problems(reports: Sequence[SplitReport], max_examples: int = 5) -> bool:
    """Log inconsistencies. Returns ``True`` when every split is consistent."""
    all_ok = True
    for report in reports:
        if report.image_count != report.label_count:
            all_ok = False
            LOGGER.warning(
                "split %s: %d images but %d label files",
                report.name,
                report.image_count,
                report.label_count,
            )
        for kind, items in (
            ("images without labels", report.missing_labels),
            ("labels without images", report.orphan_labels),
            ("malformed label lines", report.malformed),
        ):
            if items:
                all_ok = False
                LOGGER.warning(
                    "split %s: %d %s; first %d: %s",
                    report.name,
                    len(items),
                    kind,
                    min(max_examples, len(items)),
                    ", ".join(items[:max_examples]),
                )
    return all_ok


def prepare(data_dir: Path, output: Path, strict: bool = False) -> Path:
    """Validate the dataset and write the generated ``data.yaml``.

    Args:
        data_dir: Dataset root.
        output: Destination YAML path.
        strict: When ``True``, raise :class:`DatasetError` on any inconsistency
            instead of only warning.

    Returns:
        The path the YAML was written to.
    """
    reports = [validate_split(data_dir, split) for split in SPLITS]
    print_report(reports)
    consistent = _log_problems(reports)
    if strict and not consistent:
        raise DatasetError("dataset validation failed; see warnings above")

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(build_data_yaml(data_dir, reports), encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"could not write {output}: {exc}") from exc
    LOGGER.info("wrote dataset descriptor to %s", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate the fire/smoke dataset and generate a portable data.yaml.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Dataset root containing data/{train,val,test} (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("DATA_YAML", DEFAULT_OUTPUT),
        help="Where to write the generated data.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any split is inconsistent.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Python logging level (default: %(default)s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        data_dir = _resolve_data_dir(args.data_dir)
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = (Path(__file__).resolve().parent.parent / output).resolve()
        prepare(data_dir, output, strict=args.strict)
    except DatasetError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
