import asyncio
import base64
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import Settings, get_settings


@dataclass
class SessionBuffer:
    pcm_data: bytearray = field(default_factory=bytearray)
    last_chunk: int = -1


class RealtimeProxy:
    """Thin wrapper around OpenAI Realtime API (stub implementation)."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._sessions: Dict[str, SessionBuffer] = {}
        self._lock = asyncio.Lock()

    async def ingest_chunk(self, session_id: str, chunk_index: int, pcm_b64: str, is_final: bool) -> Dict[str, str]:
        async with self._lock:
            buffer = self._sessions.setdefault(session_id, SessionBuffer())
            if chunk_index <= buffer.last_chunk:
                return {"status": "duplicate"}

            buffer.last_chunk = chunk_index
            buffer.pcm_data.extend(base64.b64decode(pcm_b64))

            if not is_final:
                return {"status": "partial"}

            pcm_payload = bytes(buffer.pcm_data)
            del self._sessions[session_id]

        transcript, audio_b64 = await self._forward_to_openai(pcm_payload)
        return {
            "status": "complete",
            "transcript": transcript,
            "audio_base64": audio_b64,
        }

    async def _forward_to_openai(self, pcm_bytes: bytes) -> tuple[str, str]:
        # TODO: Replace with streaming Realtime API integration.
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set")

        # TODO: replace stub with streaming Realtime API implementation.
        transcript = "stub response"
        audio_base64 = base64.b64encode(pcm_bytes).decode()
        return transcript, audio_base64
