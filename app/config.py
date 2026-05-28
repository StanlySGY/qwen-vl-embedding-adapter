from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_alias: str = "qwen3-vl-embedding"
    model_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    host: str = "0.0.0.0"
    port: int = 8080
    max_model_len: int = 8192
    seed: int = 0
    image_fetch_timeout: float = 20.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")


@lru_cache
def get_settings() -> Settings:
    return Settings()
