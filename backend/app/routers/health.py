"""Health and readiness endpoint."""

from __future__ import annotations

import logging
from typing import Final

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..errors import ModelNotAvailableError
from ..schemas import HealthResponse
from ..services.detector import get_detector

LOGGER: Final = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and model status")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report whether the detector is loaded and which weights are in use.

    This endpoint always returns ``200`` -- a missing model is a normal state for
    a freshly cloned repository, and the UI renders it as a banner. Use the
    ``model_loaded`` field, not the status code, to decide whether detection
    endpoints will work.
    """
    detector = get_detector(settings)
    detail: str | None = None
    if not detector.is_loaded:
        try:
            detector.load()
        except ModelNotAvailableError as exc:
            detail = exc.message
            LOGGER.debug("health check: model unavailable (%s)", exc.message)

    loaded = detector.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=settings.version,
        model_loaded=loaded,
        model_path=str(settings.resolved_model_path),
        model_source=detector.model_source,
        device=settings.device or "auto",
        class_names=detector.class_names if loaded else [],
        detail=detail,
    )
