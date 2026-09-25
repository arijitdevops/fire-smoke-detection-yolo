"""Detection endpoints for still images and videos."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from ..config import Settings, get_settings
from ..errors import MediaProcessingError, UploadValidationError
from ..schemas import Detection, ImageDetectionResponse, JobCreated, JobKind
from ..services.detector import Detector, get_detector
from ..services.jobs import get_job_store, run_video_job
from ..services.video import draw_detections
from ..uploads import SavedUpload, save_upload

LOGGER: Final = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["detect"])

#: JPEG quality used for the annotated preview returned to the browser.
PREVIEW_JPEG_QUALITY: Final[int] = 88


def _validate_threshold(value: float | None, name: str) -> float | None:
    """Return ``value`` when it is a usable probability, else raise."""
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise UploadValidationError(f"{name} must be between 0 and 1", {name: value})
    return value


def _decode_image(path: Path) -> np.ndarray:
    """Read an image from disk as BGR.

    Raises:
        MediaProcessingError: If OpenCV cannot decode the file.

    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise MediaProcessingError(
            "could not decode the uploaded image; it may be corrupt or not a real image"
        )
    return image


def _encode_preview(image: np.ndarray) -> str:
    """Encode an annotated image as a base64 ``data:`` URI.

    Raises:
        MediaProcessingError: If JPEG encoding fails.

    """
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
    if not ok:
        raise MediaProcessingError("could not encode the annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _run_image_detection(
    saved: SavedUpload, detector: Detector, conf: float, iou: float
) -> ImageDetectionResponse:
    """Decode, detect and annotate. Runs in a worker thread."""
    image = _decode_image(saved.path)
    started = time.perf_counter()
    detections: list[Detection] = detector.detect_image(image, conf=conf, iou=iou)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    annotated = draw_detections(image, detections)
    class_counts: dict[str, int] = {}
    for detection in detections:
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1

    height, width = image.shape[:2]
    return ImageDetectionResponse(
        filename=saved.original_filename,
        width=int(width),
        height=int(height),
        detections=detections,
        class_counts=class_counts,
        conf_threshold=conf,
        iou_threshold=iou,
        inference_ms=round(elapsed_ms, 2),
        annotated_image=_encode_preview(annotated),
    )


@router.post(
    "/image",
    response_model=ImageDetectionResponse,
    summary="Detect fire and smoke in a still image",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (jpg, png, bmp or webp)."),
    conf: float | None = Form(default=None, description="Confidence threshold override."),
    iou: float | None = Form(default=None, description="NMS IoU threshold override."),
    settings: Settings = Depends(get_settings),
) -> ImageDetectionResponse:
    """Run synchronous detection and return the boxes plus an annotated preview.

    The annotated image comes back as a base64 ``data:`` URI so a single request
    carries both the JSON detections and the picture the user sees.
    """
    validated_conf = _validate_threshold(conf, "conf")
    validated_iou = _validate_threshold(iou, "iou")
    conf_value = settings.conf_threshold if validated_conf is None else validated_conf
    iou_value = settings.iou_threshold if validated_iou is None else validated_iou

    saved = await save_upload(
        upload=file,
        destination_dir=settings.resolved_upload_dir,
        allowed_extensions=settings.image_extensions,
        max_bytes=settings.max_upload_bytes,
        kind="image",
    )
    detector = get_detector(settings)
    try:
        return await run_in_threadpool(_run_image_detection, saved, detector, conf_value, iou_value)
    finally:
        try:
            saved.path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - platform-specific lock failures
            LOGGER.warning("could not remove %s: %s", saved.path, exc)


@router.post(
    "/video",
    response_model=JobCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a video for annotation",
)
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (mp4, mov, avi, mkv or webm)."),
    settings: Settings = Depends(get_settings),
) -> JobCreated:
    """Store the upload, queue an annotation job and return ``202`` with its id.

    The heavy work runs in FastAPI's background-task threadpool; poll
    ``GET /api/jobs/{job_id}`` or subscribe to ``/api/jobs/{job_id}/stream`` for
    progress. Returns ``503`` straight away when no model weights are available.
    """
    # Fail fast with 503 instead of accepting a job that can never run.
    await run_in_threadpool(get_detector(settings).ensure_loaded)

    saved = await save_upload(
        upload=file,
        destination_dir=settings.resolved_upload_dir,
        allowed_extensions=settings.video_extensions,
        max_bytes=settings.max_upload_bytes,
        kind="video",
    )
    store = get_job_store(settings)
    record = store.create(
        kind=JobKind.VIDEO, filename=saved.original_filename, source_path=saved.path
    )
    background_tasks.add_task(run_video_job, record.id, settings)
    LOGGER.info("queued video job %s (%s)", record.id, saved.original_filename)

    return JobCreated(
        job_id=record.id,
        state=record.state,
        kind=record.kind,
        filename=record.filename,
        created_at=record.created_at,
        status_url=f"/api/jobs/{record.id}",
        stream_url=f"/api/jobs/{record.id}/stream",
    )
