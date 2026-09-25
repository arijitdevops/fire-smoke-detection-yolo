"""Lazy-loading YOLO detector shared by the whole process.

The Ultralytics model is expensive to construct, so a single instance is created
on first use (or warmed at startup by the lifespan handler) and reused for every
request.  Inference itself is CPU/GPU bound and is always called from a worker
thread, never from the event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..config import Settings, get_settings
from ..errors import ModelNotAvailableError
from ..schemas import Detection

LOGGER: Final = logging.getLogger(__name__)

#: Class names of the fire/smoke dataset, used when the checkpoint has no names.
DEFAULT_CLASS_NAMES: Final[tuple[str, ...]] = ("smoke", "fire")

#: Reported by ``/api/health`` so the UI can warn about a non-fire model.
SOURCE_TRAINED: Final[str] = "trained"
SOURCE_COCO_FALLBACK: Final[str] = "coco-fallback"

_NO_WEIGHTS_MESSAGE: Final[str] = "no weights found - train first or set MODEL_PATH"


class Detector:
    """Thread-safe wrapper around an Ultralytics YOLO model.

    Args:
        settings: Effective application settings.

    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._model_source: str | None = None
        self._class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
        self._load_error: str | None = None
        self._lock = threading.Lock()
        # Ultralytics predictors keep per-call state, so concurrent jobs must not
        # share one predictor at the same time.
        self._predict_lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def is_loaded(self) -> bool:
        """``True`` when a model is in memory and ready for inference."""
        return self._model is not None

    @property
    def model_source(self) -> str | None:
        """``'trained'``, ``'coco-fallback'`` or ``None`` when nothing is loaded."""
        return self._model_source

    @property
    def class_names(self) -> list[str]:
        """Class names reported by the loaded checkpoint."""
        return list(self._class_names)

    @property
    def load_error(self) -> str | None:
        """The last load failure message, if any."""
        return self._load_error

    @property
    def weights_path(self) -> Path:
        """Configured weights path (may not exist yet)."""
        return self._settings.resolved_model_path

    # ------------------------------------------------------------------ load
    def _select_weights(self) -> tuple[str, str]:
        """Choose the checkpoint to load.

        Returns:
            ``(weights, source)`` where ``source`` is one of the ``SOURCE_*``
            constants.

        Raises:
            ModelNotAvailableError: If the trained weights are missing and the
                COCO fallback is disabled.

        """
        trained = self._settings.resolved_model_path
        if trained.is_file():
            return str(trained), SOURCE_TRAINED
        if self._settings.allow_coco_fallback:
            LOGGER.warning(
                "weights %s not found; falling back to pretrained %s. "
                "This model detects COCO classes, not fire or smoke.",
                trained,
                self._settings.coco_fallback_weights,
            )
            return self._settings.coco_fallback_weights, SOURCE_COCO_FALLBACK
        raise ModelNotAvailableError(
            _NO_WEIGHTS_MESSAGE,
            {
                "model_path": str(trained),
                "hint": (
                    "Run 'python training/train.py' to produce weights, point MODEL_PATH at an "
                    "existing checkpoint, or set ALLOW_COCO_FALLBACK=true to smoke-test the "
                    "plumbing with a pretrained COCO model."
                ),
            },
        )

    def load(self, force: bool = False) -> None:
        """Load the model if it is not loaded yet.

        Args:
            force: Reload even when a model is already in memory.

        Raises:
            ModelNotAvailableError: If no usable checkpoint can be loaded.

        """
        with self._lock:
            if self._model is not None and not force:
                return
            weights, source = self._select_weights()
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - dependency guard
                self._load_error = "ultralytics is not installed"
                raise ModelNotAvailableError(
                    "ultralytics is not installed; run 'pip install -r backend/requirements.txt'"
                ) from exc

            started = time.perf_counter()
            try:
                model = YOLO(weights)
                if self._settings.device:
                    model.to(self._settings.device)
            except Exception as exc:  # torch/ultralytics raise many error types
                self._load_error = str(exc)
                raise ModelNotAvailableError(
                    f"could not load weights '{weights}': {exc}",
                    {"model_path": weights},
                ) from exc

            self._model = model
            self._model_source = source
            self._load_error = None
            self._class_names = _extract_class_names(model)
            LOGGER.info(
                "loaded %s model from %s in %.2fs (classes: %s)",
                source,
                weights,
                time.perf_counter() - started,
                ", ".join(self._class_names),
            )

    def ensure_loaded(self) -> None:
        """Load the model on first use.

        Raises:
            ModelNotAvailableError: Propagated from :meth:`load`.

        """
        if self._model is None:
            self.load()

    def unload(self) -> None:
        """Drop the model reference so the process can free its memory."""
        with self._lock:
            self._model = None
            self._model_source = None
            LOGGER.info("detector unloaded")

    # ------------------------------------------------------------- inference
    def _predict(self, image: np.ndarray, conf: float, iou: float) -> list[Detection]:
        """Run inference on one BGR image and convert the boxes to schemas."""
        self.ensure_loaded()
        model = self._model
        if model is None:  # pragma: no cover - ensure_loaded raises first
            raise ModelNotAvailableError(_NO_WEIGHTS_MESSAGE)
        try:
            with self._predict_lock:
                results = model.predict(
                    source=image,
                    conf=conf,
                    iou=iou,
                    device=self._settings.device or None,
                    verbose=False,
                )
        except Exception as exc:
            raise ModelNotAvailableError(f"inference failed: {exc}") from exc
        if not results:
            return []
        return _boxes_to_detections(results[0], self._class_names)

    def detect_image(
        self, image: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        """Detect fire and smoke in a still image.

        Args:
            image: HxWx3 BGR array as produced by ``cv2.imread``.
            conf: Confidence threshold override; defaults to ``CONF_THRESHOLD``.
            iou: NMS IoU override; defaults to ``IOU_THRESHOLD``.

        Returns:
            Detections sorted by descending confidence.

        Raises:
            ModelNotAvailableError: If the model cannot be loaded or inference fails.

        """
        detections = self._predict(
            image,
            conf if conf is not None else self._settings.conf_threshold,
            iou if iou is not None else self._settings.iou_threshold,
        )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def detect_frame(
        self, frame: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        """Detect fire and smoke in a single video frame.

        Identical to :meth:`detect_image` but named for the video pipeline, which
        calls it once per sampled frame inside a worker thread.
        """
        return self.detect_image(frame, conf=conf, iou=iou)


def _extract_class_names(model: Any) -> tuple[str, ...]:
    """Read ``model.names`` in index order, falling back to the dataset names."""
    names = getattr(model, "names", None)
    if isinstance(names, dict) and names:
        try:
            return tuple(str(names[key]) for key in sorted(names, key=int))
        except (KeyError, TypeError, ValueError):  # pragma: no cover - odd checkpoints
            return tuple(str(value) for value in names.values())
    if isinstance(names, (list, tuple)) and names:
        return tuple(str(value) for value in names)
    return DEFAULT_CLASS_NAMES


def _boxes_to_detections(result: Any, class_names: Sequence[str]) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into :class:`Detection` models."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
    except AttributeError:  # pragma: no cover - already-numpy stubs in tests
        xyxy = np.asarray(boxes.xyxy)
        confidences = np.asarray(boxes.conf)
        class_ids = np.asarray(boxes.cls).astype(int)

    detections: list[Detection] = []
    for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
        index = int(class_id)
        name = class_names[index] if 0 <= index < len(class_names) else str(index)
        detections.append(
            Detection(
                class_id=index,
                class_name=name,
                confidence=float(min(max(float(confidence), 0.0), 1.0)),
                xyxy=[float(value) for value in box[:4]],
            )
        )
    return detections


_detector: Detector | None = None
_detector_lock: Final = threading.Lock()


def get_detector(settings: Settings | None = None) -> Detector:
    """Return the process-wide :class:`Detector` singleton, creating it if needed."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = Detector(settings or get_settings())
        return _detector


def set_detector(detector: Detector | None) -> None:
    """Replace the singleton. Used by the test-suite to inject a stub."""
    global _detector
    with _detector_lock:
        _detector = detector
