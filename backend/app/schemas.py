"""Pydantic models shared by the routers, the services and the test-suite.

``frontend/src/types.ts`` mirrors these definitions field for field; keep the two
in sync when you change anything here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class JobState(str, Enum):
    """Lifecycle states of an asynchronous video job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """``True`` once the job will not change state again."""
        return self in (JobState.DONE, JobState.FAILED)


class JobKind(str, Enum):
    """The type of media a job processed."""

    VIDEO = "video"


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Progress = Annotated[float, Field(ge=0.0, le=100.0)]


class Detection(BaseModel):
    """A single detected object in one image or video frame."""

    class_id: int = Field(description="Class index: 0 = smoke, 1 = fire.")
    class_name: str = Field(description="Human-readable class name.")
    confidence: Confidence = Field(description="Detector confidence in [0, 1].")
    xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Absolute pixel box as [x1, y1, x2, y2] in the source frame.",
    )

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height


class FrameDetection(BaseModel):
    """Detections for one sampled video frame."""

    frame_index: int = Field(ge=0, description="Zero-based index in the source video.")
    timestamp_seconds: float = Field(ge=0.0, description="Position of the frame in seconds.")
    detections: list[Detection] = Field(default_factory=list)


class ClassSummary(BaseModel):
    """Aggregated statistics for one class across a whole video."""

    class_name: str
    frames: int = Field(ge=0, description="Number of frames containing this class.")
    detections: int = Field(ge=0, description="Total boxes of this class.")
    peak_confidence: Confidence = 0.0
    first_seen_seconds: float | None = Field(
        default=None, description="Timestamp of the first detection of this class."
    )


class DetectionSummary(BaseModel):
    """Everything the UI needs to describe a processed video at a glance."""

    total_frames: int = Field(ge=0, description="Frame count reported by the container.")
    processed_frames: int = Field(ge=0, description="Frames actually run through the model.")
    frames_with_detections: int = Field(ge=0)
    frames_with_fire: int = Field(ge=0)
    frames_with_smoke: int = Field(ge=0)
    total_detections: int = Field(ge=0)
    peak_confidence: Confidence = 0.0
    peak_confidence_class: str | None = None
    first_detection_seconds: float | None = None
    fps: float = Field(ge=0.0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    frame_stride: int = Field(ge=1)
    per_class: list[ClassSummary] = Field(default_factory=list)
    timeline: list[FrameDetection] = Field(
        default_factory=list,
        description=(
            "Sampled frames that contained detections, truncated to keep the payload small."
        ),
    )
    timeline_truncated: bool = Field(
        default=False, description="True when more detected frames exist than the timeline holds."
    )
    codec: str = Field(
        default="h264",
        description="Codec of the annotated MP4: 'h264' (browser playable) or 'mp4v' (fallback).",
    )


class ImageDetectionResponse(BaseModel):
    """Synchronous response for ``POST /api/detect/image``."""

    filename: str = Field(description="Sanitised name of the uploaded file.")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    detections: list[Detection] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(
        default_factory=dict, description="Detection count per class name."
    )
    conf_threshold: Confidence
    iou_threshold: Confidence
    inference_ms: float = Field(ge=0.0)
    annotated_image: str = Field(
        description="Annotated image as a base64 data URI (image/jpeg), ready for an <img> src."
    )


class JobCreated(BaseModel):
    """``202`` response for ``POST /api/detect/video``."""

    job_id: str
    state: JobState = JobState.QUEUED
    kind: JobKind = JobKind.VIDEO
    filename: str
    created_at: datetime
    status_url: str
    stream_url: str


class JobStatus(BaseModel):
    """Full job record returned by ``GET /api/jobs/{job_id}``."""

    job_id: str
    kind: JobKind
    state: JobState
    progress: Progress = 0.0
    message: str | None = None
    error: str | None = None
    filename: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    download_url: str | None = None
    summary: DetectionSummary | None = None


class HealthResponse(BaseModel):
    """``GET /api/health`` payload."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="'ok' when the service can serve detections.")
    version: str
    model_loaded: bool
    model_path: str
    model_source: str | None = Field(
        default=None, description="'trained', 'coco-fallback' or null when nothing is loaded."
    )
    device: str
    class_names: list[str] = Field(default_factory=list)
    detail: str | None = Field(default=None, description="Why the model is unavailable, if it is.")


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response produced by this API."""

    error: str = Field(description="Stable machine-readable code, e.g. 'model_unavailable'.")
    message: str = Field(description="Human-readable explanation.")
    detail: dict[str, Any] = Field(default_factory=dict)


class JobDeleted(BaseModel):
    """``DELETE /api/jobs/{job_id}`` payload."""

    job_id: str
    deleted: bool = True
