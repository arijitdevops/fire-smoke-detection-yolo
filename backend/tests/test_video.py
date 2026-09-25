"""Unit tests for the drawing helpers and the video summary builder."""

from __future__ import annotations

import numpy as np
import pytest
from app.schemas import Detection
from app.services.video import CLASS_COLORS, color_for, draw_detections


def _frame(width: int = 200, height: int = 120) -> np.ndarray:
    """Return a black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_color_for_known_and_unknown_classes() -> None:
    """Known classes get their palette colour; unknown ones get the default."""
    assert color_for("fire") == CLASS_COLORS["fire"]
    assert color_for("SMOKE") == CLASS_COLORS["smoke"]
    assert color_for("aeroplane") not in CLASS_COLORS.values()


def test_draw_detections_does_not_mutate_the_source_frame() -> None:
    """With copy=True the caller's frame is untouched."""
    frame = _frame()
    detections = [Detection(class_id=1, class_name="fire", confidence=0.8, xyxy=[10, 10, 80, 60])]

    annotated = draw_detections(frame, detections)

    assert annotated is not frame
    assert frame.sum() == 0
    assert annotated.sum() > 0


def test_draw_detections_clips_boxes_to_the_frame() -> None:
    """Boxes reaching outside the frame are clipped instead of raising."""
    frame = _frame()
    detections = [
        Detection(class_id=0, class_name="smoke", confidence=0.5, xyxy=[-50, -50, 5000, 5000])
    ]

    annotated = draw_detections(frame, detections)

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0


def test_draw_detections_skips_degenerate_boxes() -> None:
    """A zero-area box produces no drawing at all."""
    frame = _frame()
    detections = [Detection(class_id=1, class_name="fire", confidence=0.9, xyxy=[30, 30, 30, 30])]

    annotated = draw_detections(frame, detections)

    assert annotated.sum() == 0


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_draw_detections_handles_confidence_extremes(confidence: float) -> None:
    """The confidence bar renders across the whole [0, 1] range."""
    annotated = draw_detections(
        _frame(),
        [Detection(class_id=1, class_name="fire", confidence=confidence, xyxy=[5, 20, 90, 70])],
    )

    assert annotated.shape == (120, 200, 3)
