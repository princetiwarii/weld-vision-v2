from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


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

    ALLOWED_ORIGINS: str = "*"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
