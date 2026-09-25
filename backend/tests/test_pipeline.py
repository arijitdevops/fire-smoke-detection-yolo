"""End-to-end video pipeline: real OpenCV decode, stub detector, real H.264 encode."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.schemas import Detection
from app.services.encoder import CODEC_H264, CODEC_MP4V
from app.services.video import process_video
from fastapi.testclient import TestClient

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")


def _make_clip(path: Path, frames: int = 24, width: int = 161, height: int = 121) -> Path:
    """Write a tiny synthetic clip (odd dimensions on purpose) with OpenCV."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 12.0, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((height, width, 3), index * 8 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _video_codec(path: Path) -> str:
    """Return the codec name ffmpeg reports for the first video stream."""
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stderr.splitlines():
        if "Video:" in line:
            return line.split("Video:", 1)[1].strip().split()[0]
    raise AssertionError(f"no video stream found in {path}:\n{completed.stderr}")


class _TimedDetector:
    """Returns a fire box only from the 12th frame onwards."""

    def __init__(self) -> None:
        self.frame = 0

    def detect_frame(self, frame, conf=None, iou=None):  # noqa: ANN001, D102
        index = self.frame
        self.frame += 1
        if index < 6:  # stride 2 -> frame 12 onwards
            return []
        return [Detection(class_id=1, class_name="fire", confidence=0.8, xyxy=[5, 5, 60, 60])]


def test_process_video_writes_browser_playable_h264(tmp_path: Path) -> None:
    """The annotated output is H.264 and the summary reports first-detection time."""
    source = _make_clip(tmp_path / "in.mp4")
    progress: list[float] = []

    result = process_video(
        source,
        tmp_path / "out.mp4",
        _TimedDetector(),
        frame_stride=2,
        progress=lambda pct, done, total: progress.append(pct),
    )

    assert result.summary.codec == CODEC_H264
    assert _video_codec(result.output_path) == "h264"
    assert result.summary.processed_frames == 12
    assert result.summary.frames_with_fire == 6
    assert result.summary.frames_with_smoke == 0
    assert result.summary.first_detection_seconds == pytest.approx(1.0)
    assert result.summary.per_class[0].first_seen_seconds == pytest.approx(1.0)
    assert progress[-1] == 100.0
    capture = cv2.VideoCapture(str(result.output_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    capture.release()


def test_process_video_mp4v_fallback(tmp_path: Path) -> None:
    """With H.264 disabled the OpenCV writer is used and reported as mp4v."""
    source = _make_clip(tmp_path / "in.mp4", frames=6)
    result = process_video(
        source, tmp_path / "out.mp4", _TimedDetector(), frame_stride=1, prefer_h264=False
    )
    assert result.summary.codec == CODEC_MP4V
    assert result.output_path.stat().st_size > 0


def test_video_upload_end_to_end(client: TestClient, tmp_path: Path) -> None:
    """Upload -> background job -> done -> ranged download of an H.264 MP4."""
    clip = _make_clip(tmp_path / "clip.mp4")
    with clip.open("rb") as handle:
        response = client.post(
            "/api/detect/video", files={"file": ("clip.mp4", handle, "video/mp4")}
        )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    deadline = time.monotonic() + 30
    payload = client.get(f"/api/jobs/{job_id}").json()
    while payload["state"] not in {"done", "failed"} and time.monotonic() < deadline:
        time.sleep(0.2)
        payload = client.get(f"/api/jobs/{job_id}").json()
    assert payload["state"] == "done", payload
    summary = payload["summary"]
    assert summary["codec"] == "h264"
    assert summary["frames_with_fire"] > 0 and summary["frames_with_smoke"] > 0
    assert summary["first_detection_seconds"] == 0.0

    ranged = client.get(f"/api/jobs/{job_id}/download", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 100
    full = client.get(f"/api/jobs/{job_id}/download")
    out = tmp_path / "annotated.mp4"
    out.write_bytes(full.content)
    assert _video_codec(out) == "h264"
