"""FastAPI application factory for the fire and smoke detection service.

Run it with::

    uvicorn app.main:app --reload --port 8000    # from backend/
    python -m app.main                           # equivalent, reads PORT/HOST
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .errors import ApiError, ModelNotAvailableError
from .routers import detect, health, jobs
from .schemas import ErrorResponse
from .services.detector import get_detector, set_detector
from .services.jobs import get_job_store, set_job_store

LOGGER: Final = logging.getLogger("app")

#: How often expired jobs are swept while the service runs.
CLEANUP_INTERVAL_SECONDS: Final[float] = 3600.0

DESCRIPTION: Final[str] = (
    "Detect fire and smoke in images and videos with a YOLO detector. "
    "Images are processed synchronously; videos are queued as background jobs "
    "that report progress over Server-Sent Events."
)


def configure_logging(level: str) -> None:
    """Configure root logging once, without clobbering uvicorn's handlers."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=False,
    )
    logging.getLogger("app").setLevel(level)


async def _cleanup_loop(settings: Settings) -> None:
    """Periodically delete expired jobs until the task is cancelled."""
    store = get_job_store(settings)
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            removed = await asyncio.to_thread(store.cleanup_expired)
            if removed:
                LOGGER.info("scheduled cleanup removed %d job(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception:  # keep the loop alive across transient I/O errors
            LOGGER.exception("job cleanup pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare directories, warm the model and run background maintenance."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    try:
        settings.ensure_directories()
    except OSError as exc:
        LOGGER.error("could not create working directories: %s", exc)
        raise

    store = get_job_store(settings)
    removed = await asyncio.to_thread(store.cleanup_expired)
    LOGGER.info("startup cleanup removed %d expired job(s)", removed)

    detector = get_detector(settings)
    try:
        await asyncio.to_thread(detector.load)
        LOGGER.info("model warm-up complete (source: %s)", detector.model_source)
    except ModelNotAvailableError as exc:
        # Not fatal: /api/health reports the problem and the UI shows a banner.
        LOGGER.warning("starting without a model: %s", exc.message)

    cleanup_task = asyncio.create_task(_cleanup_loop(settings), name="job-cleanup")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        detector.unload()
        store.close()
        set_detector(None)
        set_job_store(None)
        LOGGER.info("shutdown complete")


def _error_response(status_code: int, code: str, message: str, detail: dict) -> JSONResponse:
    """Build a JSON response using the shared :class:`ErrorResponse` envelope."""
    payload = ErrorResponse(error=code, message=message, detail=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every error shares the :class:`ErrorResponse` shape."""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        """Map a typed application error onto its HTTP status."""
        log = LOGGER.warning if exc.status_code < 500 else LOGGER.error
        log("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return 422 with the offending fields in a stable envelope."""
        return _error_response(
            422,
            "validation_error",
            "the request could not be validated",
            {"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Wrap framework-raised HTTP errors (404, 405, ...) in the envelope."""
        return _error_response(
            exc.status_code, "http_error", str(exc.detail), {"path": request.url.path}
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Log the traceback and return a generic 500 without leaking internals."""
        LOGGER.exception("unhandled error for %s %s", request.method, request.url.path)
        return _error_response(500, "internal_error", "an unexpected server error occurred", {})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional settings override (used by the test-suite).

    Returns:
        A fully wired :class:`FastAPI` instance.

    """
    effective = settings or get_settings()
    configure_logging(effective.log_level)

    app = FastAPI(
        title=effective.app_name,
        version=effective.version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = effective

    origins = effective.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    app.include_router(health.router)
    app.include_router(detect.router)
    app.include_router(jobs.router)
    register_exception_handlers(app)

    if settings is not None:
        # Keep dependency injection consistent with the override.
        app.dependency_overrides[get_settings] = lambda: effective

    LOGGER.info("application configured (CORS origins: %s)", ", ".join(origins) or "none")
    return app


app = create_app()


def main() -> None:
    """Run the service with uvicorn using ``HOST``/``PORT`` from the environment."""
    import os

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level=get_settings().log_level.lower(),
    )


if __name__ == "__main__":
    main()
