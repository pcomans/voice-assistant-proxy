from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    assistant_shared_secret: str
    vad_threshold: float = 0.6  # Higher threshold (0.0-1.0) = less sensitive to noise
    openai_realtime_url: str = "https://api.openai.com/v1/realtime"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
