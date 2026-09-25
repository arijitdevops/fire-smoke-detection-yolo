"""Health endpoint behaviour with and without loadable weights."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_loaded_model(client: TestClient) -> None:
    """With a detector available the service reports 'ok' and its class names."""
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_source"] == "trained"
    assert payload["class_names"] == ["smoke", "fire"]


def test_health_is_degraded_without_weights(client_without_model: TestClient) -> None:
    """A missing checkpoint yields 200 + degraded, never a crash."""
    response = client_without_model.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["model_loaded"] is False
    assert payload["class_names"] == []
    assert "no weights found" in (payload["detail"] or "")


def test_openapi_document_is_served(client: TestClient) -> None:
    """The OpenAPI schema is generated without errors."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/detect/image" in paths
    assert "/api/jobs/{job_id}/download" in paths
