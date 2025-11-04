import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, status
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


@app.post("/v1/audio")
async def post_audio(
    chunk: AudioChunk,
    x_assistant_token: str = Header(..., alias="X-Assistant-Token"),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
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

    # Enforce 4MB response size limit to match client constraints
    MAX_RESPONSE_SIZE = 4 * 1024 * 1024  # 4MB
    try:
        response_json = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        response_len = len(response_json)
    except (TypeError, ValueError) as exc:
        logger.warning("Failed to serialize proxy response for session=%s: %s", chunk.session_id, exc)
        response_len = -1
        response_json = None

    if response_json and response_len >= MAX_RESPONSE_SIZE:
        logger.error(
            "Response exceeds 4MB limit session=%s body_bytes=%d",
            chunk.session_id,
            response_len,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"response_too_large: {response_len} bytes exceeds 4MB limit",
        )

    logger.info(
        "Returning response session=%s status=%s body_bytes=%d",
        chunk.session_id,
        result.get("status"),
        response_len,
    )

    return result
