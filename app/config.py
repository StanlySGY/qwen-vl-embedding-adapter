from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_alias: str = "qwen3-vl-embedding"
    upstream_base_url: AnyHttpUrl = "http://127.0.0.1:8000"
    upstream_model: str = "qwen3-vl-embedding"
    upstream_timeout: float = 120.0
    host: str = "0.0.0.0"
    port: int = 8080

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")

    @property
    def upstream_embeddings_url(self) -> str:
        return f"{str(self.upstream_base_url).rstrip('/')}/v1/embeddings"


@lru_cache
def get_settings() -> Settings:
    return Settings()
