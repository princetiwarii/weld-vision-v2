from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Any
from pydantic import field_validator


class Settings(BaseSettings):
    APP_NAME: str = "WeldVision API"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"

    GEMINI_API_KEY: str

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-south-1"
    AWS_S3_BUCKET: str

    # PostgreSQL — asyncpg driver
    DATABASE_URL: str  # e.g. postgresql+asyncpg://user:pass@host:5432/weldvision

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Render and other platforms sometimes provide postgres:// or postgresql://
            # but SQLAlchemy's create_async_engine requires postgresql+asyncpg://
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    ALLOWED_ORIGINS: str = "*"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
