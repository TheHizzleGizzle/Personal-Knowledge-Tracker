from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl, BaseSettings, PostgresDsn, validator


class Settings(BaseSettings):
    project_name: str = "Knowledge Metabolism Tracker API"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    backend_cors_origins: list[AnyHttpUrl] | str = []

    database_url: Optional[PostgresDsn] = None
    sqlite_fallback_path: str = "sqlite:///./database/app.db"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    jwt_algorithm: str = "HS256"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value  # type: ignore[return-value]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()