"""Application settings, loaded from the environment and an optional ``.env``.

All paths are resolved to absolute paths against the repository root so that the
service behaves identically whether it is started from ``backend/``, from the
repository root, or from inside a container.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER: Final = logging.getLogger(__name__)

#: ``backend/`` -- the directory that holds ``app`` and ``.env``.
BACKEND_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Repository root, used to resolve relative paths such as ``runs/detect/...``.
REPO_ROOT: Final[Path] = BACKEND_ROOT.parent


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated environment value into trimmed, non-empty parts."""
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseSettings):
    """Runtime configuration for the detection API."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # ``model_path`` would otherwise collide with pydantic's ``model_`` namespace.
        protected_namespaces=("settings_",),
    )

    app_name: str = "Fire and Smoke Detection API"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # --- model -------------------------------------------------------------
    model_path: Path = Path("runs/detect/fire_smoke/weights/best.pt")
    allow_coco_fallback: bool = False
    coco_fallback_weights: str = "yolo11n.pt"
    device: str = ""
    conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # --- data and storage --------------------------------------------------
    upload_dir: Path = Path("backend/uploads")
    output_dir: Path = Path("backend/outputs")
    job_db_path: Path = Path("backend/jobs.sqlite3")

    # --- uploads -----------------------------------------------------------
    max_upload_mb: int = Field(default=100, gt=0)
    allowed_image_extensions: str = ".jpg,.jpeg,.png,.bmp,.webp"
    allowed_video_extensions: str = ".mp4,.mov,.avi,.mkv,.webm"

    # --- video processing --------------------------------------------------
    frame_stride: int = Field(default=3, ge=1)
    max_video_frames: int = Field(default=18_000, gt=0)

    # --- jobs and CORS -----------------------------------------------------
    job_retention_hours: int = Field(default=24, gt=0)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Upper-case the level and reject values ``logging`` does not know."""
        level = value.strip().upper()
        if not isinstance(logging.getLevelName(level), int):
            raise ValueError(f"unknown log level: {value!r}")
        return level

    @property
    def max_upload_bytes(self) -> int:
        """Upload size cap in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def image_extensions(self) -> frozenset[str]:
        """Lower-cased allowlist of image suffixes, including the leading dot."""
        return frozenset(ext.lower() for ext in _split_csv(self.allowed_image_extensions))

    @property
    def video_extensions(self) -> frozenset[str]:
        """Lower-cased allowlist of video suffixes, including the leading dot."""
        return frozenset(ext.lower() for ext in _split_csv(self.allowed_video_extensions))

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list; ``*`` is honoured as a single wildcard entry."""
        return _split_csv(self.cors_origins)

    def resolve(self, value: Path) -> Path:
        """Resolve ``value`` against the repository root when it is relative."""
        path = value.expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def resolved_model_path(self) -> Path:
        """Absolute path to the detector weights."""
        return self.resolve(self.model_path)

    @property
    def resolved_upload_dir(self) -> Path:
        """Absolute path to the upload staging directory."""
        return self.resolve(self.upload_dir)

    @property
    def resolved_output_dir(self) -> Path:
        """Absolute path to the annotated-output directory."""
        return self.resolve(self.output_dir)

    @property
    def resolved_job_db_path(self) -> Path:
        """Absolute path to the SQLite job database."""
        return self.resolve(self.job_db_path)

    def ensure_directories(self) -> None:
        """Create the upload, output and job-database directories if needed.

        Raises:
            OSError: If a directory cannot be created; the caller logs and exits.

        """
        for directory in (
            self.resolved_upload_dir,
            self.resolved_output_dir,
            self.resolved_job_db_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.debug("ensured directory %s", directory)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    The cache keeps FastAPI dependency injection cheap and lets tests call
    ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()
