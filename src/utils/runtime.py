"""Runtime helpers for toggling optional pipeline integrations."""

from __future__ import annotations

import os
from functools import lru_cache

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return default


@lru_cache(maxsize=1)
def pipeline_publish_enabled() -> bool:
    """Return True when DB/Mongo/S3 publishing should run."""

    env_value = os.getenv("PIPELINE_PUBLISH_ENABLED")
    return _parse_bool(env_value, default=True)


@lru_cache(maxsize=1)
def pipeline_use_minio() -> bool:
    """Return True when pipeline steps should hit MinIO for downloads/pushes."""

    env_value = os.getenv("PIPELINE_USE_MINIO")
    return _parse_bool(env_value, default=True)


def reset_runtime_caches() -> None:
    """Reset memoised helpers (mostly useful for tests)."""

    pipeline_publish_enabled.cache_clear()  # type: ignore[attr-defined]
    pipeline_use_minio.cache_clear()  # type: ignore[attr-defined]
