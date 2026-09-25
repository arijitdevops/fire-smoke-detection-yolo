"""Upload validation and still-image detection."""

from __future__ import annotations

from fastapi.testclient import TestClient

from .conftest import encode_image


def test_detect_image_returns_boxes_and_preview(client: TestClient) -> None:
    """A valid image yields detections plus a base64 annotated preview."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("engine_bay.jpg", encode_image(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "engine_bay.jpg"
    assert payload["width"] == 320 and payload["height"] == 240
    assert payload["class_counts"] == {"fire": 1, "smoke": 1}
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")
    assert [item["class_name"] for item in payload["detections"]] == ["fire", "smoke"]


def test_detect_image_sanitises_traversal_filenames(client: TestClient) -> None:
    """A path-traversal filename is reduced to a harmless basename."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("../../../etc/passwd.jpg", encode_image(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "passwd.jpg"


def test_detect_image_rejects_disallowed_extension(client: TestClient) -> None:
    """An executable disguised as an upload is refused with 400."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("payload.exe", b"MZ\x90\x00", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"


def test_detect_image_rejects_wrong_content_type(client: TestClient) -> None:
    """A text/plain part is refused even when the suffix looks fine."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("note.jpg", encode_image(), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"


def test_detect_image_rejects_oversized_upload(client: TestClient) -> None:
    """Anything above MAX_UPLOAD_MB (1 MB in tests) is refused with 413."""
    oversized = b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024)

    response = client.post(
        "/api/detect/image",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_detect_image_rejects_empty_upload(client: TestClient) -> None:
    """A zero-byte part is refused before it reaches the decoder."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"


def test_detect_image_rejects_undecodable_file(client: TestClient) -> None:
    """A valid extension with garbage bytes surfaces a 422 media error."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("broken.png", b"not really a png", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "media_error"


def test_detect_image_rejects_out_of_range_threshold(client: TestClient) -> None:
    """Conf outside [0, 1] is rejected with a clear message."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("frame.jpg", encode_image(), "image/jpeg")},
        data={"conf": "1.5"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"


def test_detect_image_without_weights_returns_503(client_without_model: TestClient) -> None:
    """Without weights the API answers 503 with an actionable message."""
    response = client_without_model.post(
        "/api/detect/image",
        files={"file": ("frame.jpg", encode_image(), "image/jpeg")},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "model_unavailable"
    assert "no weights found" in payload["message"]
    assert "MODEL_PATH" in payload["detail"]["hint"]


def test_detect_video_rejects_non_video(client: TestClient) -> None:
    """The video endpoint enforces its own extension allowlist."""
    response = client.post(
        "/api/detect/video",
        files={"file": ("notes.txt", b"hello", "video/mp4")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"


def test_detect_video_without_weights_returns_503(client_without_model: TestClient) -> None:
    """A video is not queued when the job could never load a model."""
    response = client_without_model.post(
        "/api/detect/video",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"
    assert client_without_model.get("/api/jobs").json() == []
