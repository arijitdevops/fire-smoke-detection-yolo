"""Typed application errors mapped to HTTP responses.

Every error raised deliberately by this service derives from :class:`ApiError`,
which carries the HTTP status code and a stable machine-readable ``code`` that
the frontend switches on.  ``app.main`` registers a single handler for the base
class, so adding a new error type never requires touching the app factory.
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for errors that map onto an HTTP response.

    Args:
        message: Human-readable message shown to the user.
        detail: Optional structured payload with extra context.

    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class UploadValidationError(ApiError):
    """The uploaded file failed extension, content-type or emptiness checks."""

    status_code = 400
    code = "invalid_upload"


class PayloadTooLargeError(ApiError):
    """The uploaded file exceeded ``MAX_UPLOAD_MB``."""

    status_code = 413
    code = "payload_too_large"


class ModelNotAvailableError(ApiError):
    """No usable detector weights could be loaded.

    Raised when ``MODEL_PATH`` does not exist and the COCO fallback is disabled,
    or when loading the checkpoint fails.  Surfaced as ``503`` so the UI can show
    a "train the model first" banner instead of a crash.
    """

    status_code = 503
    code = "model_unavailable"


class MediaProcessingError(ApiError):
    """The uploaded image or video could not be decoded or written."""

    status_code = 422
    code = "media_error"


class JobNotFoundError(ApiError):
    """No job with the requested identifier exists (or it has expired)."""

    status_code = 404
    code = "job_not_found"


class JobNotFinishedError(ApiError):
    """The job exists but has no downloadable result yet."""

    status_code = 409
    code = "job_not_finished"


class ResultMissingError(ApiError):
    """The job is marked done but its output file has disappeared from disk."""

    status_code = 410
    code = "result_missing"
