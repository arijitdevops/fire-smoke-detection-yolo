"""Video annotation pipeline built on OpenCV.

The pipeline reads the uploaded clip, runs the detector on every ``frame_stride``
-th frame (carrying the previous boxes on skipped frames so the overlay does not
flicker), draws the boxes, and writes an annotated H.264 MP4 through
:class:`~app.services.encoder.AnnotatedVideoWriter` so the result plays in browsers.

Everything in this module is synchronous and CPU bound: it must be called from a
worker thread, never from the event loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np

from ..errors import MediaProcessingError
from ..schemas import ClassSummary, Detection, DetectionSummary, FrameDetection
from .encoder import AnnotatedVideoWriter

LOGGER: Final = logging.getLogger(__name__)

#: BGR colours per class name; anything unknown falls back to ``_DEFAULT_COLOR``.
CLASS_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "smoke": (196, 174, 120),  # muted steel blue
    "fire": (54, 106, 240),  # warm orange-red
}
_DEFAULT_COLOR: Final[tuple[int, int, int]] = (120, 200, 120)

#: Used when the container does not report a usable frame rate.
FALLBACK_FPS: Final[float] = 25.0

#: Upper bound on entries kept in :attr:`DetectionSummary.timeline`.
MAX_TIMELINE_ENTRIES: Final[int] = 250

#: Callback signature: ``(percent, processed_frames, total_frames)``.
ProgressCallback = Callable[[float, int, int], None]


class FrameDetector(Protocol):
    """Minimal detector interface the video pipeline depends on."""

    def detect_frame(
        self, frame: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        """Return detections for a single BGR frame."""


@dataclass(slots=True, frozen=True)
class VideoMetadata:
    """Container properties read from the source clip."""

    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Clip duration derived from the frame count and frame rate."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(slots=True)
class _ClassAccumulator:
    """Running statistics for one class while a clip is processed."""

    frames: int = 0
    detections: int = 0
    peak_confidence: float = 0.0
    first_seen_seconds: float | None = None


@dataclass(slots=True)
class VideoProcessingResult:
    """Outcome of :func:`process_video`."""

    output_path: Path
    metadata: VideoMetadata
    summary: DetectionSummary
    timeline: list[FrameDetection] = field(default_factory=list)


def _sanitise_fps(raw: float) -> float:
    """Clamp a container frame rate into a plausible range."""
    if not np.isfinite(raw) or raw <= 0.0 or raw > 240.0:
        LOGGER.warning("implausible source fps %.3f; using %.1f", raw, FALLBACK_FPS)
        return FALLBACK_FPS
    return float(raw)


def probe_video(path: Path) -> VideoMetadata:
    """Read fps, resolution and frame count from a video file.

    Args:
        path: Video file to inspect.

    Raises:
        MediaProcessingError: If OpenCV cannot open the file or it has no frames.

    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaProcessingError(
                "could not decode the uploaded video; is it a valid, non-empty file?",
                {"filename": path.name},
            )
        fps = _sanitise_fps(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise MediaProcessingError(
                "the uploaded video reports a zero frame size and cannot be processed",
                {"width": width, "height": height},
            )
        return VideoMetadata(fps=fps, width=width, height=height, frame_count=max(frame_count, 0))
    finally:
        capture.release()


def color_for(class_name: str) -> tuple[int, int, int]:
    """Return the BGR colour used to draw ``class_name``."""
    return CLASS_COLORS.get(class_name.lower(), _DEFAULT_COLOR)


def draw_detections(
    frame: np.ndarray, detections: Sequence[Detection], copy: bool = True
) -> np.ndarray:
    """Draw boxes, labels and per-detection confidence bars onto a frame.

    Args:
        frame: HxWx3 BGR image.
        detections: Boxes to draw, in absolute pixel coordinates.
        copy: Draw on a copy (default) instead of mutating ``frame`` in place.

    Returns:
        The annotated image.

    """
    canvas = frame.copy() if copy else frame
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(width, height) / 400))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(width, height) / 1400)

    for detection in detections:
        color = color_for(detection.class_name)
        x1 = int(max(0, min(detection.xyxy[0], width - 1)))
        y1 = int(max(0, min(detection.xyxy[1], height - 1)))
        x2 = int(max(0, min(detection.xyxy[2], width - 1)))
        y2 = int(max(0, min(detection.xyxy[3], height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"{detection.class_name} {detection.confidence * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, max(1, thickness)
        )
        bar_height = max(3, thickness * 2)
        box_height = text_height + baseline + bar_height + 6
        label_top = y1 - box_height
        if label_top < 0:  # label would fall off the top edge: draw it inside
            label_top = y1
        label_bottom = min(height - 1, label_top + box_height)
        label_right = min(width - 1, x1 + text_width + 10)

        cv2.rectangle(canvas, (x1, label_top), (label_right, label_bottom), color, cv2.FILLED)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, label_top + text_height + 3),
            font,
            font_scale,
            (16, 16, 16),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

        # Confidence bar underneath the label text.
        bar_top = label_bottom - bar_height - 2
        bar_full = max(1, label_right - x1 - 10)
        bar_fill = int(bar_full * min(max(detection.confidence, 0.0), 1.0))
        cv2.rectangle(
            canvas,
            (x1 + 5, bar_top),
            (x1 + 5 + bar_full, bar_top + bar_height),
            (32, 32, 32),
            cv2.FILLED,
        )
        if bar_fill > 0:
            cv2.rectangle(
                canvas,
                (x1 + 5, bar_top),
                (x1 + 5 + bar_fill, bar_top + bar_height),
                (250, 250, 250),
                cv2.FILLED,
            )
    return canvas


def _build_summary(
    metadata: VideoMetadata,
    frame_stride: int,
    processed_frames: int,
    written_frames: int,
    frames_with_detections: int,
    total_detections: int,
    peak: tuple[str | None, float],
    first_detection_seconds: float | None,
    accumulators: dict[str, _ClassAccumulator],
    timeline: list[FrameDetection],
    timeline_truncated: bool,
    codec: str,
) -> DetectionSummary:
    """Assemble the :class:`DetectionSummary` returned to the API layer."""
    peak_class, peak_confidence = peak
    per_class = [
        ClassSummary(
            class_name=name,
            frames=accumulator.frames,
            detections=accumulator.detections,
            peak_confidence=accumulator.peak_confidence,
            first_seen_seconds=accumulator.first_seen_seconds,
        )
        for name, accumulator in sorted(accumulators.items())
    ]
    fire = accumulators.get("fire")
    smoke = accumulators.get("smoke")
    return DetectionSummary(
        total_frames=max(metadata.frame_count, written_frames),
        processed_frames=processed_frames,
        frames_with_detections=frames_with_detections,
        frames_with_fire=fire.frames if fire else 0,
        frames_with_smoke=smoke.frames if smoke else 0,
        total_detections=total_detections,
        peak_confidence=peak_confidence,
        peak_confidence_class=peak_class,
        first_detection_seconds=first_detection_seconds,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        duration_seconds=(written_frames / metadata.fps) if metadata.fps > 0 else 0.0,
        frame_stride=frame_stride,
        per_class=per_class,
        timeline=timeline,
        timeline_truncated=timeline_truncated,
        codec=codec,
    )


def process_video(
    source: Path,
    destination: Path,
    detector: FrameDetector,
    frame_stride: int = 3,
    conf: float | None = None,
    iou: float | None = None,
    max_frames: int = 18_000,
    progress: ProgressCallback | None = None,
    prefer_h264: bool = True,
) -> VideoProcessingResult:
    """Annotate ``source`` into ``destination`` and summarise what was found.

    Detection runs on every ``frame_stride``-th frame; skipped frames reuse the
    most recent boxes so the overlay stays stable. Per-class frame counts are
    therefore counted over *sampled* frames, not over every frame of the clip.

    Args:
        source: Uploaded video file.
        destination: Path of the annotated MP4 to write.
        detector: Object exposing :meth:`FrameDetector.detect_frame`.
        frame_stride: Run the model every N frames (``1`` = every frame).
        conf: Confidence threshold override.
        iou: NMS IoU override.
        max_frames: Safety cap; processing stops after this many frames.
        progress: Optional callback invoked as the clip is processed.
        prefer_h264: Encode with ffmpeg/libx264 (default); ``False`` forces mp4v.

    Returns:
        A :class:`VideoProcessingResult` with the output path and summary.

    Raises:
        MediaProcessingError: If the clip cannot be decoded or the output cannot
            be written.

    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    metadata = probe_video(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MediaProcessingError("could not reopen the uploaded video for processing")

    try:
        writer = AnnotatedVideoWriter(
            destination, metadata.width, metadata.height, metadata.fps, prefer_h264=prefer_h264
        )
    except MediaProcessingError:
        capture.release()
        raise

    accumulators: dict[str, _ClassAccumulator] = {}
    timeline: list[FrameDetection] = []
    timeline_truncated = False
    last_detections: list[Detection] = []
    processed_frames = 0
    written_frames = 0
    frames_with_detections = 0
    total_detections = 0
    peak: tuple[str | None, float] = (None, 0.0)
    first_detection_seconds: float | None = None
    last_reported = -1.0
    expected_frames = metadata.frame_count if metadata.frame_count > 0 else 0

    try:
        while written_frames < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if written_frames % frame_stride == 0:
                processed_frames += 1
                timestamp = written_frames / metadata.fps if metadata.fps > 0 else 0.0
                last_detections = detector.detect_frame(frame, conf=conf, iou=iou)
                if last_detections:
                    frames_with_detections += 1
                    total_detections += len(last_detections)
                    if first_detection_seconds is None:
                        first_detection_seconds = round(timestamp, 3)
                    if len(timeline) < MAX_TIMELINE_ENTRIES:
                        timeline.append(
                            FrameDetection(
                                frame_index=written_frames,
                                timestamp_seconds=round(timestamp, 3),
                                detections=list(last_detections),
                            )
                        )
                    else:
                        timeline_truncated = True
                    seen_classes: set[str] = set()
                    for detection in last_detections:
                        accumulator = accumulators.setdefault(
                            detection.class_name, _ClassAccumulator()
                        )
                        accumulator.detections += 1
                        if detection.class_name not in seen_classes:
                            seen_classes.add(detection.class_name)
                            accumulator.frames += 1
                        if detection.confidence > accumulator.peak_confidence:
                            accumulator.peak_confidence = detection.confidence
                        if accumulator.first_seen_seconds is None:
                            accumulator.first_seen_seconds = round(timestamp, 3)
                        if detection.confidence > peak[1]:
                            peak = (detection.class_name, detection.confidence)

            writer.write(draw_detections(frame, last_detections, copy=False))
            written_frames += 1

            if progress is not None and expected_frames > 0:
                percent = min(99.0, written_frames / expected_frames * 100.0)
                if percent - last_reported >= 1.0:
                    last_reported = percent
                    progress(percent, written_frames, expected_frames)
            elif progress is not None and written_frames % 50 == 0:
                progress(0.0, written_frames, 0)
        else:
            LOGGER.warning("stopped after the %d frame safety cap for %s", max_frames, source.name)
    except MediaProcessingError:
        raise
    except Exception as exc:  # OpenCV and torch raise a wide variety of errors
        raise MediaProcessingError(f"video processing failed: {exc}") from exc
    finally:
        capture.release()
        writer.close()

    if written_frames == 0:
        raise MediaProcessingError("the uploaded video contains no readable frames")

    if progress is not None:
        progress(100.0, written_frames, expected_frames or written_frames)

    summary = _build_summary(
        metadata=metadata,
        frame_stride=frame_stride,
        processed_frames=processed_frames,
        written_frames=written_frames,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        peak=peak,
        first_detection_seconds=first_detection_seconds,
        accumulators=accumulators,
        timeline=timeline,
        timeline_truncated=timeline_truncated,
        codec=writer.codec,
    )
    LOGGER.info(
        "annotated %s -> %s [%s] (%d frames written, %d sampled, %d detections)",
        source.name,
        destination.name,
        writer.codec,
        written_frames,
        processed_frames,
        total_detections,
    )
    return VideoProcessingResult(
        output_path=destination, metadata=metadata, summary=summary, timeline=timeline
    )
