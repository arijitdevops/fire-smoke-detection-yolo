"""Job lifecycle: creation, progress, streaming, download and deletion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from app.config import Settings
from app.routers import detect as detect_router
from app.schemas import DetectionSummary, JobKind, JobState
from app.services.jobs import JobStore, get_job_store
from fastapi.testclient import TestClient

VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512


def _summary() -> DetectionSummary:
    """A minimal but valid summary used by the fake worker."""
    return DetectionSummary(
        total_frames=120,
        processed_frames=40,
        frames_with_detections=12,
        frames_with_fire=9,
        frames_with_smoke=5,
        total_detections=21,
        peak_confidence=0.93,
        peak_confidence_class="fire",
        first_detection_seconds=1.25,
        fps=24.0,
        width=640,
        height=360,
        duration_seconds=5.0,
        frame_stride=3,
    )


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Replace the video worker so tests never invoke OpenCV or a model."""

    def _run(job_id: str, job_settings: Settings | None = None) -> None:
        effective = job_settings or settings
        store = get_job_store(effective)
        store.mark_running(job_id)
        store.update_progress(job_id, 50.0, "annotating frames")
        output = effective.resolved_output_dir / f"{job_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(VIDEO_BYTES)
        store.mark_done(job_id, output, _summary())

    monkeypatch.setattr(detect_router, "run_video_job", _run)


def _upload_video(client: TestClient) -> str:
    """Upload a dummy clip and return the created job id."""
    response = client.post(
        "/api/detect/video",
        files={"file": ("driveway.mp4", VIDEO_BYTES, "video/mp4")},
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["state"] == "queued"
    assert payload["status_url"].endswith(payload["job_id"])
    return str(payload["job_id"])


def test_video_job_completes_and_downloads(client: TestClient, fake_worker: None) -> None:
    """A queued job runs, reports done and serves the annotated MP4."""
    job_id = _upload_video(client)

    status = client.get(f"/api/jobs/{job_id}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["state"] == JobState.DONE.value
    assert payload["progress"] == 100.0
    assert payload["download_url"] == f"/api/jobs/{job_id}/download"
    assert payload["summary"]["frames_with_fire"] == 9

    download = client.get(f"/api/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "video/mp4"
    assert "annotated_driveway.mp4" in download.headers["content-disposition"]
    assert download.content == VIDEO_BYTES


def test_job_stream_emits_done_event(client: TestClient, fake_worker: None) -> None:
    """The SSE stream ends with a 'done' event carrying the job status."""
    job_id = _upload_video(client)

    response = client.get(f"/api/jobs/{job_id}/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: progress" in body
    assert "event: done" in body
    assert job_id in body


def test_job_listing_includes_new_job(client: TestClient, fake_worker: None) -> None:
    """Recently updated jobs are listed newest first."""
    job_id = _upload_video(client)

    response = client.get("/api/jobs", params={"limit": 5})

    assert response.status_code == 200
    assert job_id in [item["job_id"] for item in response.json()]


def test_download_before_completion_returns_409(client: TestClient) -> None:
    """Without a worker the job stays queued and the download is refused."""
    job_id = _upload_video(client)

    response = client.get(f"/api/jobs/{job_id}/download")

    assert response.status_code == 409
    assert response.json()["error"] == "job_not_finished"


def test_unknown_job_returns_404(client: TestClient) -> None:
    """An unknown identifier is a 404 with the shared error envelope."""
    response = client.get("/api/jobs/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"] == "job_not_found"


def test_delete_job_removes_record_and_output(
    client: TestClient, fake_worker: None, settings: Settings
) -> None:
    """Deleting a job removes both the row and the annotated file."""
    job_id = _upload_video(client)
    output = settings.resolved_output_dir / f"{job_id}.mp4"
    assert output.is_file()

    deleted = client.delete(f"/api/jobs/{job_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"job_id": job_id, "deleted": True}
    assert not output.exists()

    repeated = client.delete(f"/api/jobs/{job_id}")
    assert repeated.status_code == 404
    assert repeated.json()["deleted"] is False


def test_store_cleanup_removes_expired_jobs(job_store: JobStore, tmp_path: Path) -> None:
    """Jobs past their retention window are swept together with their files."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(VIDEO_BYTES)
    record = job_store.create(JobKind.VIDEO, "clip.mp4", source)

    expired = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    job_store._update(record.id, expires_at=expired)  # noqa: SLF001 - store internals

    assert job_store.cleanup_expired() == 1
    assert job_store.get(record.id) is None
    assert not source.exists()


def test_store_tracks_progress_and_failure(job_store: JobStore, tmp_path: Path) -> None:
    """Progress is clamped and failures are recorded with their message."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(VIDEO_BYTES)
    record = job_store.create(JobKind.VIDEO, "clip.mp4", source)

    job_store.mark_running(record.id)
    job_store.update_progress(record.id, 143.0, "annotating frames")
    running = job_store.require(record.id)
    assert running.state is JobState.RUNNING
    assert running.progress == 100.0

    job_store.mark_failed(record.id, "codec unavailable")
    failed = job_store.require(record.id)
    assert failed.state is JobState.FAILED
    assert failed.error == "codec unavailable"
    assert failed.to_status().download_url is None
