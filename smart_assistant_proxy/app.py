import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, constr

from .config import Settings, get_settings
from .realtime import RealtimeProxy


class AudioChunk(BaseModel):
    session_id: constr(min_length=1)
    chunk_index: int
    pcm_base64: constr(min_length=1)
    is_final: bool = False


app = FastAPI(title="Smart Assistant Proxy", version="0.1.0")
proxy_instance = RealtimeProxy()
logger = logging.getLogger(__name__)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio", response_model=None)
async def post_audio(
    chunk: AudioChunk,
    x_assistant_token: str = Header(..., alias="X-Assistant-Token"),
    settings: Settings = Depends(get_settings),
):
    """
    Process audio chunk with streaming support (Option C).

    Non-final chunks: Returns {"status": "partial"} immediately
    Final chunk: Returns StreamingResponse with newline-delimited JSON audio chunks
    """
    if x_assistant_token != settings.assistant_shared_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    logger.info(
        "Received chunk session=%s index=%d final=%s pcm_b64_len=%d",
        chunk.session_id,
        chunk.chunk_index,
        chunk.is_final,
        len(chunk.pcm_base64),
    )

    try:
        result = await proxy_instance.ingest_chunk(
            session_id=chunk.session_id,
            chunk_index=chunk.chunk_index,
            pcm_b64=chunk.pcm_base64,
            is_final=chunk.is_final,
        )
    except Exception as exc:  # pragma: no cover - defensive logging for production use
        logger.exception("Proxy ingest failed for session=%s index=%d", chunk.session_id, chunk.chunk_index)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="proxy_ingest_failed") from exc

    # Check if result is an async generator (final chunk streaming response)
    if hasattr(result, '__anext__'):  # It's an async generator
        logger.info("Returning streaming response session=%s", chunk.session_id)
        return StreamingResponse(
            result,
            media_type="application/x-ndjson",  # Newline-delimited JSON
            headers={
                "X-Session-ID": chunk.session_id,
                "Cache-Control": "no-cache",
            }
        )

    # Non-final chunk - return dict immediately
    logger.info(
        "Returning ack session=%s status=%s",
        chunk.session_id,
        result.get("status"),
    )

    return result
