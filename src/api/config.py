import os
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()


@dataclass
class Settings:
    PG_URL: str = os.getenv("PG_URL", "postgresql+asyncpg://app:app@postgres:5432/ocr")
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://mongo:27017/")
    MONGO_DB: str = os.getenv("MONGO_DB", "ocr_db")
    MONGO_COLL: str = os.getenv("MONGO_COLL", "documents")
    API_KEY: str | None = os.getenv("API_KEY")  # if unset -> no auth required
    CORS_ORIGINS_RAW: str = os.getenv("CORS_ORIGINS", "")
    APP_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("API_PORT", "8080"))

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]


settings = Settings()


# Simple header auth dependency (enabled only if API_KEY is set)
async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
