"""Pytest configuration and fixtures."""
import base64
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_assistant_proxy.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"X-Assistant-Token": "498b1b65-26a3-49e8-a55e-46a0b47365e2"}


@pytest.fixture
def sample_pcm_audio():
    """Generate 100ms of silent PCM16 audio at 16kHz for testing."""
    sample_rate = 16000
    duration_ms = 100
    num_samples = int(sample_rate * duration_ms / 1000)

    # Generate silent audio (zeros)
    pcm_data = bytes(num_samples * 2)  # 2 bytes per sample (16-bit)

    return base64.b64encode(pcm_data).decode()


@pytest.fixture
def test_audio_file():
    """Path to test audio file (if exists)."""
    test_file = Path(__file__).parent.parent / "test_data" / "test_audio_16khz.wav"
    if test_file.exists():
        return test_file
    return None


@pytest.fixture
def real_audio_pcm(test_audio_file):
    """Load real test audio as base64 PCM."""
    if test_audio_file is None:
        pytest.skip("test_audio_16khz.wav not found")

    with wave.open(str(test_audio_file), 'rb') as wf:
        pcm_data = wf.readframes(wf.getnframes())

    return base64.b64encode(pcm_data).decode()
