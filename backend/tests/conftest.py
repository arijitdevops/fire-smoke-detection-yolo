"""Shared pytest fixtures.

The suite never loads real YOLO weights: a stub detector with the same surface
as :class:`app.services.detector.Detector` is injected into the singleton, so the
tests exercise routing, validation and the job lifecycle without torch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.config import Settings
from app.main import create_app
from app.schemas import Detection
from app.services import detector as detector_module
from app.services import jobs as jobs_module
from app.services.jobs import JobStore
from fastapi import FastAPI
from fastapi.testclient import TestClient


class StubDetector:
    """Detector double returning a fixed box for every frame."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = (
            detections
            if detections is not None
            else [
                Detection(class_id=1, class_name="fire", confidence=0.91, xyxy=[10, 10, 120, 140]),
                Detection(class_id=0, class_name="smoke", confidence=0.62, xyxy=[40, 5, 200, 90]),
            ]
        )
        self.calls = 0
        self._loaded = True

    # --- surface used by the routers and the video pipeline ---------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_source(self) -> str:
        return "trained"

    @property
    def class_names(self) -> list[str]:
        return ["smoke", "fire"]

    @property
    def load_error(self) -> None:
        return None

    def load(self, force: bool = False) -> None:
        self._loaded = True

    def ensure_loaded(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def detect_image(
        self, image: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        self.calls += 1
        return list(self._detections)

    def detect_frame(
        self, frame: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        return self.detect_image(frame, conf=conf, iou=iou)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing every writable path at a temporary directory."""
    return Settings(
        model_path=tmp_path / "weights" / "best.pt",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        job_db_path=tmp_path / "jobs.sqlite3",
        max_upload_mb=1,
        frame_stride=2,
        job_retention_hours=1,
        allow_coco_fallback=False,
        cors_origins="http://testserver",
        log_level="WARNING",
    )


@pytest.fixture
def stub_detector() -> Iterator[StubDetector]:
    """A stub detector registered as the process-wide singleton."""
    stub = StubDetector()
    detector_module.set_detector(stub)  # type: ignore[arg-type]
    try:
        yield stub
    finally:
        detector_module.set_detector(None)


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """A configured application whose singletons are reset afterwards."""
    application = create_app(settings)
    try:
        yield application
    finally:
        detector_module.set_detector(None)
        jobs_module.set_job_store(None)


@pytest.fixture
def client(app: FastAPI, stub_detector: StubDetector) -> Iterator[TestClient]:
    """Test client with the stub detector installed and lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(app: FastAPI) -> Iterator[TestClient]:
    """Test client with no detector installed, so weights are genuinely missing."""
    detector_module.set_detector(None)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def job_store(settings: Settings) -> Iterator[JobStore]:
    """A standalone job store backed by a temporary SQLite file."""
    store = JobStore(settings.resolved_job_db_path, settings.job_retention_hours)
    try:
        yield store
    finally:
        store.close()
        jobs_module.set_job_store(None)


def encode_image(width: int = 320, height: int = 240, suffix: str = ".jpg") -> bytes:
    """Return the bytes of a small synthetic image."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (32, 64, 200)
    ok, buffer = cv2.imencode(suffix, image)
    assert ok, "failed to encode the fixture image"
    return bytes(buffer.tobytes())
