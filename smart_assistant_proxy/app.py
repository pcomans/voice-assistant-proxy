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

    result = await proxy_instance.ingest_chunk(
        session_id=chunk.session_id,
        chunk_index=chunk.chunk_index,
        pcm_b64=chunk.pcm_base64,
        is_final=chunk.is_final,
    )
    return result
