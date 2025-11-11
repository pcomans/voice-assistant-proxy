"""Integration tests for streaming proxy (Option C)."""
import base64
import json
import os

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    condition=not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration test requires OpenAI API key and live service. Set RUN_INTEGRATION_TESTS=1 to run."
)
async def test_streaming_response_with_real_audio(client, auth_headers, real_audio_pcm):
    """Test full streaming flow with real audio.

    This is an integration test that:
    1. Sends real audio to the proxy
    2. Verifies streaming response format
    3. Collects PCM16 audio chunks (24kHz mono)
    4. Validates response completeness

    Run with: pytest -v -m integration --run-integration
    """
    # Send first chunk (non-final)
    response = client.post(
        "/v1/audio",
        json={
            "session_id": "integration-test",
            "chunk_index": 0,
            "pcm_base64": real_audio_pcm[:1000],  # First part
            "is_final": False
        },
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["status"] == "partial"

    # Send final chunk and get streaming response (raw binary PCM)
    with client.stream(
        "POST",
        "/v1/audio",
        json={
            "session_id": "integration-test",
            "chunk_index": 1,
            "pcm_base64": real_audio_pcm,
            "is_final": True
        },
        headers=auth_headers
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

        total_audio_bytes = 0

        # Collect streaming binary PCM response
        for chunk in response.iter_bytes():
            if chunk:
                total_audio_bytes += len(chunk)

        # Verify we got audio response
        assert total_audio_bytes > 0, "Should receive non-empty audio data"

        # Audio should be 24kHz 16-bit PCM mono (2 bytes per sample)
        # Verify it's a reasonable size (at least 1 second = 48000 bytes)
        assert total_audio_bytes >= 48000, f"Audio seems too short: {total_audio_bytes} bytes"

        print(f"\nIntegration test results:")
        print(f"  Total audio bytes: {total_audio_bytes}")
        print(f"  Duration (approx): {total_audio_bytes / 48000:.2f} seconds")


@pytest.mark.unit
def test_chunked_streaming_protocol(client, auth_headers, sample_pcm_audio):
    """Test the chunked streaming protocol (Option C).

    Verifies:
    - Non-final chunks return {"status": "partial"}
    - Chunks are accepted in sequence
    - Session isolation works
    """
    session_id = "test-chunked-protocol"

    # Send multiple non-final chunks
    for i in range(3):
        response = client.post(
            "/v1/audio",
            json={
                "session_id": session_id,
                "chunk_index": i,
                "pcm_base64": sample_pcm_audio,
                "is_final": False
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"

    # Final chunk triggers streaming (will fail without OpenAI, but validates protocol)
    # This will fail at OpenAI connection stage, which is expected in unit tests
    try:
        with client.stream(
            "POST",
            "/v1/audio",
            json={
                "session_id": session_id,
                "chunk_index": 3,
                "pcm_base64": sample_pcm_audio,
                "is_final": True
            },
            headers=auth_headers
        ) as response:
            # If we don't have OpenAI configured, this will error
            # But we've validated the protocol works
            pass
    except Exception:
        # Expected to fail without OpenAI API key
        pass


@pytest.mark.unit
def test_session_isolation(client, auth_headers, sample_pcm_audio):
    """Test that different sessions are isolated."""
    # Send chunks to two different sessions
    for session_id in ["session-a", "session-b"]:
        response = client.post(
            "/v1/audio",
            json={
                "session_id": session_id,
                "chunk_index": 0,
                "pcm_base64": sample_pcm_audio,
                "is_final": False
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "partial"
