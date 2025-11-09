"""Unit tests for API endpoints."""
import pytest


def test_healthz(client):
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_audio_endpoint_requires_auth(client, sample_pcm_audio):
    """Test that audio endpoint requires authentication."""
    payload = {
        "session_id": "test-session",
        "chunk_index": 0,
        "pcm_base64": sample_pcm_audio,
        "is_final": True
    }

    # No auth header
    response = client.post("/v1/audio", json=payload)
    assert response.status_code == 422  # Missing required header

    # Wrong token
    response = client.post(
        "/v1/audio",
        json=payload,
        headers={"X-Assistant-Token": "wrong-token"}
    )
    assert response.status_code == 401


def test_audio_endpoint_validates_payload(client, auth_headers):
    """Test payload validation."""
    # Empty session_id
    response = client.post(
        "/v1/audio",
        json={
            "session_id": "",
            "chunk_index": 0,
            "pcm_base64": "test",
            "is_final": True
        },
        headers=auth_headers
    )
    assert response.status_code == 422

    # Missing required field
    response = client.post(
        "/v1/audio",
        json={
            "session_id": "test",
            "chunk_index": 0,
            # Missing pcm_base64
            "is_final": True
        },
        headers=auth_headers
    )
    assert response.status_code == 422


def test_non_final_chunk_returns_partial(client, auth_headers, sample_pcm_audio):
    """Test that non-final chunks return quick acknowledgment."""
    payload = {
        "session_id": "test-partial",
        "chunk_index": 0,
        "pcm_base64": sample_pcm_audio,
        "is_final": False
    }

    response = client.post("/v1/audio", json=payload, headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data.get("status") == "partial"
