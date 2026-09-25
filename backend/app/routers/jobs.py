"""Job status, progress streaming, download and deletion endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from ..config import Settings, get_settings
from ..errors import JobNotFinishedError, ResultMissingError
from ..schemas import JobDeleted, JobState, JobStatus
from ..services.jobs import JobRecord, get_job_store

LOGGER: Final = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: How often the SSE endpoint re-reads job state, in seconds.
POLL_INTERVAL_SECONDS: Final[float] = 0.5

#: Emit a comment line after this many idle polls so proxies keep the socket open.
HEARTBEAT_EVERY: Final[int] = 20

#: Hard cap on a single SSE subscription (1 hour) to avoid leaking generators.
MAX_STREAM_SECONDS: Final[float] = 3600.0

_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Tell nginx not to buffer the stream; harmless elsewhere.
    "X-Accel-Buffering": "no",
}


def _format_event(event: str, payload: str) -> str:
    """Render one Server-Sent Event frame."""
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("", response_model=list[JobStatus], summary="List recent jobs")
async def list_jobs(limit: int = 20, settings: Settings = Depends(get_settings)) -> list[JobStatus]:
    """Return the most recently updated jobs, newest first."""
    store = get_job_store(settings)
    records = await run_in_threadpool(store.list_jobs, min(max(limit, 1), 100))
    return [record.to_status() for record in records]


@router.get("/{job_id}", response_model=JobStatus, summary="Job status and progress")
async def get_job(job_id: str, settings: Settings = Depends(get_settings)) -> JobStatus:
    """Return the current state, progress and (when finished) summary of a job.

    Raises:
        JobNotFoundError: Mapped to ``404`` when the job is unknown or expired.

    """
    store = get_job_store(settings)
    record: JobRecord = await run_in_threadpool(store.require, job_id)
    return record.to_status()


@router.get("/{job_id}/stream", summary="Server-Sent Events progress stream")
async def stream_job(
    job_id: str, request: Request, settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    """Stream progress updates until the job reaches a terminal state.

    Events are named ``progress``, ``done`` and ``failed``; every frame carries a
    JSON-encoded :class:`~app.schemas.JobStatus`. Clients that cannot use
    ``EventSource`` should poll ``GET /api/jobs/{job_id}`` instead.
    """
    store = get_job_store(settings)
    # Raises JobNotFoundError -> 404 before the stream starts.
    await run_in_threadpool(store.require, job_id)

    async def event_source() -> AsyncIterator[str]:
        """Yield SSE frames until the job finishes or the client disconnects."""
        last_payload: str | None = None
        idle_polls = 0
        elapsed = 0.0
        while elapsed < MAX_STREAM_SECONDS:
            if await request.is_disconnected():
                LOGGER.debug("client disconnected from stream for job %s", job_id)
                return
            record = await run_in_threadpool(store.get, job_id)
            if record is None:
                yield _format_event(
                    "failed",
                    '{"error":"job_not_found","message":"job disappeared or expired"}',
                )
                return

            payload = record.to_status().model_dump_json()
            if payload != last_payload:
                last_payload = payload
                idle_polls = 0
                yield _format_event("progress", payload)
            else:
                idle_polls += 1
                if idle_polls % HEARTBEAT_EVERY == 0:
                    yield ": heartbeat\n\n"

            if record.state.is_terminal:
                event = "done" if record.state is JobState.DONE else "failed"
                yield _format_event(event, payload)
                return

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

        LOGGER.info("stream for job %s hit the %.0fs cap", job_id, MAX_STREAM_SECONDS)

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/{job_id}/download", summary="Download the annotated MP4")
async def download_job(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Return the annotated video produced by a finished job.

    Raises:
        JobNotFoundError: ``404`` when the job is unknown.
        JobNotFinishedError: ``409`` when it is still queued or running.
        ResultMissingError: ``410`` when the file has been cleaned up.

    """
    store = get_job_store(settings)
    record: JobRecord = await run_in_threadpool(store.require, job_id)

    if record.state is not JobState.DONE or record.result_path is None:
        raise JobNotFinishedError(
            f"job is '{record.state.value}'; the annotated video is not ready yet",
            {"job_id": job_id, "state": record.state.value, "progress": record.progress},
        )
    if not record.result_path.is_file():
        raise ResultMissingError(
            "the annotated video is no longer on disk; it may have been cleaned up",
            {"job_id": job_id},
        )

    download_name = f"annotated_{record.filename.rsplit('.', 1)[0]}.mp4"
    return FileResponse(
        path=record.result_path,
        media_type="video/mp4",
        filename=download_name,
        headers={"Cache-Control": "no-store"},
    )


@router.delete("/{job_id}", response_model=JobDeleted, summary="Delete a job and its files")
async def delete_job(
    job_id: str, response: Response, settings: Settings = Depends(get_settings)
) -> JobDeleted:
    """Delete a job record together with its upload and annotated output.

    Deleting an unknown job is reported as ``deleted: false`` with a ``404``
    status so that repeated deletes stay safe for the caller.
    """
    store = get_job_store(settings)
    removed = await run_in_threadpool(store.delete, job_id, True)
    if not removed:
        response.status_code = 404
    return JobDeleted(job_id=job_id, deleted=removed)
