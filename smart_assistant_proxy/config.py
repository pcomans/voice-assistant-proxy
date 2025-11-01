from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    assistant_shared_secret: str
    openai_realtime_url: str = "https://api.openai.com/v1/realtime"

    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
