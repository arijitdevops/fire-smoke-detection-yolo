"""In-process job store backed by SQLite.

Video annotation takes far longer than an HTTP request, so uploads create a job
that a background worker thread picks up.  State lives in a small SQLite file so
that progress survives a reload of the module and can be inspected with any
SQLite client while the service runs.

Only one writer exists per process; a re-entrant lock serialises access because
``sqlite3`` connections are shared between the event loop's threadpool workers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from ..config import Settings, get_settings
from ..errors import ApiError, JobNotFoundError
from ..schemas import DetectionSummary, JobKind, JobState, JobStatus

LOGGER: Final = logging.getLogger(__name__)

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    state         TEXT NOT NULL,
    progress      REAL NOT NULL DEFAULT 0,
    message       TEXT,
    error         TEXT,
    filename      TEXT NOT NULL,
    source_path   TEXT,
    result_path   TEXT,
    summary_json  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    expires_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_expires_at ON jobs (expires_at);
"""


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp written by this module."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:  # pragma: no cover - corrupt row
        LOGGER.warning("could not parse timestamp %r", raw)
        return None


@dataclass(slots=True)
class JobRecord:
    """One row of the ``jobs`` table."""

    id: str
    kind: JobKind
    state: JobState
    progress: float
    message: str | None
    error: str | None
    filename: str
    source_path: Path | None
    result_path: Path | None
    summary: DetectionSummary | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None

    def to_status(self, api_prefix: str = "/api") -> JobStatus:
        """Convert the record into the public :class:`JobStatus` schema."""
        download_url = (
            f"{api_prefix}/jobs/{self.id}/download"
            if self.state is JobState.DONE and self.result_path is not None
            else None
        )
        return JobStatus(
            job_id=self.id,
            kind=self.kind,
            state=self.state,
            progress=round(self.progress, 2),
            message=self.message,
            error=self.error,
            filename=self.filename,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
            download_url=download_url,
            summary=self.summary,
        )


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    """Map a SQLite row onto a :class:`JobRecord`."""
    summary: DetectionSummary | None = None
    if row["summary_json"]:
        try:
            summary = DetectionSummary.model_validate(json.loads(row["summary_json"]))
        except (json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - corrupt row
            LOGGER.warning("job %s has an unreadable summary: %s", row["id"], exc)
    return JobRecord(
        id=row["id"],
        kind=JobKind(row["kind"]),
        state=JobState(row["state"]),
        progress=float(row["progress"]),
        message=row["message"],
        error=row["error"],
        filename=row["filename"],
        source_path=Path(row["source_path"]) if row["source_path"] else None,
        result_path=Path(row["result_path"]) if row["result_path"] else None,
        summary=summary,
        created_at=_parse_timestamp(row["created_at"]) or _utcnow(),
        updated_at=_parse_timestamp(row["updated_at"]) or _utcnow(),
        expires_at=_parse_timestamp(row["expires_at"]),
    )


class JobStore:
    """SQLite-backed store for asynchronous job state.

    Args:
        db_path: File to hold the ``jobs`` table. Parent directories are created.
        retention_hours: How long finished jobs (and their files) are kept.

    """

    def __init__(self, db_path: Path, retention_hours: int = 24) -> None:
        self._db_path = db_path
        self._retention = timedelta(hours=max(1, retention_hours))
        self._lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10.0)
        except sqlite3.Error as exc:
            raise RuntimeError(f"could not open job database {db_path}: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        LOGGER.info("job store ready at %s (retention %dh)", db_path, retention_hours)

    # ----------------------------------------------------------------- write
    def create(self, kind: JobKind, filename: str, source_path: Path) -> JobRecord:
        """Insert a new ``queued`` job and return it."""
        now = _utcnow()
        record = JobRecord(
            id=uuid.uuid4().hex,
            kind=kind,
            state=JobState.QUEUED,
            progress=0.0,
            message="queued",
            error=None,
            filename=filename,
            source_path=source_path,
            result_path=None,
            summary=None,
            created_at=now,
            updated_at=now,
            expires_at=now + self._retention,
        )
        with self._lock:
            self._connection.execute(
                "INSERT INTO jobs (id, kind, state, progress, message, error, filename,"
                " source_path, result_path, summary_json, created_at, updated_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?, ?)",
                (
                    record.id,
                    record.kind.value,
                    record.state.value,
                    record.progress,
                    record.message,
                    record.filename,
                    str(source_path),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                ),
            )
            self._connection.commit()
        LOGGER.info("created job %s for %s", record.id, filename)
        return record

    def _update(self, job_id: str, **columns: Any) -> None:
        """Update arbitrary columns of a job, always touching ``updated_at``."""
        if not columns:
            return
        columns["updated_at"] = _utcnow().isoformat()
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*columns.values(), job_id),
            )
            self._connection.commit()
        if cursor.rowcount == 0:
            LOGGER.warning("update for unknown job %s ignored", job_id)

    def mark_running(self, job_id: str, message: str = "processing") -> None:
        """Move a job into the ``running`` state."""
        self._update(job_id, state=JobState.RUNNING.value, progress=0.0, message=message)

    def update_progress(self, job_id: str, progress: float, message: str | None = None) -> None:
        """Record progress in percent, clamped to ``[0, 100]``."""
        clamped = min(100.0, max(0.0, float(progress)))
        if message is None:
            self._update(job_id, progress=clamped)
        else:
            self._update(job_id, progress=clamped, message=message)

    def mark_done(self, job_id: str, result_path: Path, summary: DetectionSummary) -> None:
        """Store the annotated output path and summary, and finish the job."""
        self._update(
            job_id,
            state=JobState.DONE.value,
            progress=100.0,
            message="complete",
            error=None,
            result_path=str(result_path),
            summary_json=summary.model_dump_json(),
        )
        LOGGER.info("job %s finished", job_id)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with a user-facing error message."""
        self._update(job_id, state=JobState.FAILED.value, message="failed", error=error[:2000])
        LOGGER.error("job %s failed: %s", job_id, error)

    # ------------------------------------------------------------------ read
    def get(self, job_id: str) -> JobRecord | None:
        """Return a job by id, or ``None`` when it does not exist."""
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_record(row) if row else None

    def require(self, job_id: str) -> JobRecord:
        """Return a job by id.

        Raises:
            JobNotFoundError: If no such job exists.

        """
        record = self.get(job_id)
        if record is None:
            raise JobNotFoundError("job not found; it may have expired", {"job_id": job_id})
        return record

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        """Return the most recently updated jobs, newest first."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM jobs ORDER BY datetime(updated_at) DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    # --------------------------------------------------------------- cleanup
    def delete(self, job_id: str, remove_files: bool = True) -> bool:
        """Delete a job row and, optionally, its files.

        Returns:
            ``True`` when a row was removed.

        """
        record = self.get(job_id)
        if record is None:
            return False
        if remove_files:
            _remove_files([record.source_path, record.result_path])
        with self._lock:
            self._connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._connection.commit()
        LOGGER.info("deleted job %s", job_id)
        return True

    def cleanup_expired(self) -> int:
        """Delete jobs whose retention window has passed.

        Returns:
            The number of jobs removed.

        """
        now = _utcnow().isoformat()
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            ).fetchall()
        removed = 0
        for row in rows:
            record = _row_to_record(row)
            _remove_files([record.source_path, record.result_path])
            with self._lock:
                self._connection.execute("DELETE FROM jobs WHERE id = ?", (record.id,))
                self._connection.commit()
            removed += 1
        if removed:
            LOGGER.info("cleanup removed %d expired job(s)", removed)
        return removed

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.Error as exc:  # pragma: no cover - close rarely fails
                LOGGER.warning("error closing job database: %s", exc)


def _remove_files(paths: Iterable[Path | None]) -> None:
    """Best-effort deletion of job artefacts."""
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            LOGGER.warning("could not remove %s: %s", path, exc)


_store: JobStore | None = None
_store_lock: Final = threading.Lock()


def get_job_store(settings: Settings | None = None) -> JobStore:
    """Return the process-wide :class:`JobStore`, creating it on first use."""
    global _store
    with _store_lock:
        if _store is None:
            effective = settings or get_settings()
            _store = JobStore(effective.resolved_job_db_path, effective.job_retention_hours)
        return _store


def set_job_store(store: JobStore | None) -> None:
    """Replace the singleton. Used by the test-suite and by shutdown."""
    global _store
    with _store_lock:
        _store = store


def run_video_job(job_id: str, settings: Settings | None = None) -> None:
    """Process a queued video job to completion.

    This function is executed by FastAPI's background-task threadpool: it is
    synchronous, performs the OpenCV work inline, and never touches the event
    loop. All failures are captured onto the job record.

    Args:
        job_id: Identifier returned by :meth:`JobStore.create`.
        settings: Settings override; defaults to the process settings.

    """
    effective = settings or get_settings()
    store = get_job_store(effective)
    record = store.get(job_id)
    if record is None:
        LOGGER.warning("job %s disappeared before it started", job_id)
        return
    if record.source_path is None or not record.source_path.is_file():
        store.mark_failed(job_id, "the uploaded file is no longer available on disk")
        return

    # Imported here so that importing the job store never pulls in torch.
    from .detector import get_detector
    from .video import process_video

    destination = effective.resolved_output_dir / f"{job_id}.mp4"
    store.mark_running(job_id, "loading model")
    try:
        detector = get_detector(effective)
        detector.ensure_loaded()
        store.update_progress(job_id, 0.0, "annotating frames")

        def report(percent: float, processed: int, total: int) -> None:
            """Persist progress; ``total`` is 0 when the container hides it."""
            message = (
                f"annotating frame {processed} of {total}"
                if total
                else f"annotating frame {processed}"
            )
            store.update_progress(job_id, percent, message)

        result = process_video(
            source=record.source_path,
            destination=destination,
            detector=detector,
            frame_stride=effective.frame_stride,
            conf=effective.conf_threshold,
            iou=effective.iou_threshold,
            max_frames=effective.max_video_frames,
            progress=report,
        )
    except ApiError as exc:
        store.mark_failed(job_id, exc.message)
        return
    except Exception as exc:  # last-resort guard: a job must never hang in 'running'
        LOGGER.exception("unexpected failure in job %s", job_id)
        store.mark_failed(job_id, f"unexpected error: {exc}")
        return

    store.mark_done(job_id, result.output_path, result.summary)
    _remove_files([record.source_path])
