import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import opuslib
import websockets

from .config import Settings, get_settings


@dataclass
class SessionBuffer:
    pcm_data: bytearray = field(default_factory=bytearray)
    last_chunk: int = -1


logger = logging.getLogger(__name__)


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
                logger.debug(
                    "Ignoring duplicate chunk session=%s idx=%d (last=%d)",
                    session_id,
                    chunk_index,
                    buffer.last_chunk,
                )
                return {"status": "duplicate"}

            buffer.last_chunk = chunk_index
            buffer.pcm_data.extend(base64.b64decode(pcm_b64))
            logger.info(
                "Stored chunk session=%s idx=%d final=%s total_pcm_bytes=%d",
                session_id,
                chunk_index,
                is_final,
                len(buffer.pcm_data),
            )

            if not is_final:
                return {"status": "partial"}

            pcm_payload = bytes(buffer.pcm_data)
            del self._sessions[session_id]

        logger.info("Forwarding final payload session=%s pcm_bytes=%d", session_id, len(pcm_payload))

        try:
            transcript, audio_b64 = await self._forward_to_openai(pcm_payload)
        except Exception as exc:  # pragma: no cover - defensive logging for production use
            logger.exception("Realtime forward failed for session=%s", session_id)
            raise

        logger.info(
            "Received proxy response session=%s transcript_chars=%d audio_b64_len=%d",
            session_id,
            len(transcript) if transcript is not None else -1,
            len(audio_b64) if audio_b64 is not None else -1,
        )

        return {
            "status": "complete",
            "audio_base64": audio_b64,
        }

    def _resample_audio(self, pcm_bytes: bytes, input_rate: int, output_rate: int) -> bytes:
        """Resample PCM16 audio from input_rate to output_rate."""
        # Convert bytes to int16 numpy array
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Simple linear interpolation resampling
        input_length = len(audio_data)
        output_length = int(input_length * output_rate / input_rate)

        # Create indices for resampling
        input_indices = np.linspace(0, input_length - 1, output_length)
        resampled = np.interp(input_indices, np.arange(input_length), audio_data).astype(np.int16)

        return resampled.tobytes()

    async def _forward_to_openai(self, pcm_bytes: bytes) -> tuple[str, str]:
        """Forward audio to OpenAI Realtime API via WebSocket."""
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set")

        # Device sends 16kHz PCM16, OpenAI expects 24kHz PCM16
        logger.info("Resampling audio from 16kHz to 24kHz")
        resampled_pcm = self._resample_audio(pcm_bytes, 16000, 24000)
        audio_base64 = base64.b64encode(resampled_pcm).decode("utf-8")

        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        transcript_parts = []
        audio_chunks = []

        async with websockets.connect(url, additional_headers=headers) as ws:
            # Configure session for realtime speech-to-speech
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": "You are a helpful voice assistant. Be concise but complete.",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",  # OpenAI only supports pcm16, g711_ulaw, g711_alaw
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": None,  # Disable server VAD, we handle turn detection on device
                    "max_response_output_tokens": 500,  # Allow complete sentences (~2-3 sentences)
                },
            }
            await ws.send(json.dumps(session_update))

            # Send audio input
            input_audio_buffer_append = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64,
            }
            await ws.send(json.dumps(input_audio_buffer_append))

            # Commit the audio buffer and request response
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))

            # Collect response events
            async for message in ws:
                event = json.loads(message)
                event_type = event.get("type")

                logger.debug("Received event: %s", event_type)

                if event_type == "error":
                    error_detail = event.get("error", {})
                    logger.error("OpenAI Realtime API error: %s", error_detail)
                    raise RuntimeError(f"Realtime API error: {error_detail.get('message', 'Unknown error')}")

                # Collect transcript from input transcription
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript:
                        logger.info("Input transcript: %s", transcript)

                # Collect response text
                elif event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        transcript_parts.append(delta)

                elif event_type == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info("Response text: %s", text)

                # Collect response audio chunks
                elif event_type == "response.audio.delta":
                    audio_delta = event.get("delta", "")
                    if audio_delta:
                        audio_chunks.append(audio_delta)

                # Response complete
                elif event_type == "response.done":
                    logger.info("Response completed")
                    break

        # Combine transcript and audio
        transcript = "".join(transcript_parts) if transcript_parts else None
        print(f"[OPUS] Received {len(audio_chunks)} audio chunks from OpenAI", flush=True)
        if audio_chunks:
            # Decode all PCM16 audio chunks and combine
            combined_pcm = b"".join(base64.b64decode(chunk) for chunk in audio_chunks)
            print(f"[OPUS] Combined {len(combined_pcm)} bytes of PCM16 audio", flush=True)

            # Encode PCM16 to Opus (24kHz mono, voice-optimized)
            # OpenAI outputs 24kHz PCM16, encode to Opus with same sample rate
            encoder = opuslib.Encoder(24000, 1, opuslib.APPLICATION_VOIP)

            # Opus frame size: 20ms at 24kHz = 480 samples = 960 bytes (16-bit PCM)
            frame_size = 480
            frame_bytes = frame_size * 2  # 2 bytes per sample (16-bit)

            opus_chunks = []
            for i in range(0, len(combined_pcm), frame_bytes):
                frame = combined_pcm[i:i + frame_bytes]
                # Pad last frame if needed
                if len(frame) < frame_bytes:
                    frame = frame + b'\x00' * (frame_bytes - len(frame))

                try:
                    opus_frame = encoder.encode(frame, frame_size)

                    # Add self-delimited length prefix per RFC 6716 Appendix B
                    # 1 byte for lengths 0-251, 2 bytes for 252-65535
                    frame_len = len(opus_frame)
                    if frame_len < 252:
                        delimited_frame = bytes([frame_len]) + opus_frame
                    else:
                        # Two-byte encoding: first byte is 252, second is (len - 252)
                        delimited_frame = bytes([252, frame_len - 252]) + opus_frame

                    opus_chunks.append(delimited_frame)
                except Exception as e:
                    logger.error("Failed to encode Opus frame: %s", e)
                    break

            # Combine all self-delimited Opus frames and encode to base64
            combined_opus = b"".join(opus_chunks)
            audio_base64_out = base64.b64encode(combined_opus).decode("utf-8")

            print(f"[OPUS] Encoded {len(combined_pcm)} bytes PCM to {len(combined_opus)} bytes Opus in {len(opus_chunks)} self-delimited frames", flush=True)
            if len(opus_chunks) > 0:
                first_len = len(opus_chunks[0])
                prefix_bytes = 1 if opus_chunks[0][0] < 252 else 2
                print(f"[OPUS] First frame: {first_len} bytes total (prefix={prefix_bytes} bytes, payload={first_len-prefix_bytes} bytes)", flush=True)
        else:
            audio_base64_out = ""

        return transcript, audio_base64_out
