from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import health, ocr_results, records

app = FastAPI(title="OCR Benchmark API", version="0.1.0")

# CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
app.include_router(health.router)
app.include_router(records.router)
app.include_router(ocr_results.router)
