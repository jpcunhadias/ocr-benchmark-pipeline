# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11.13
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# uv binary itself, straight from astral's own image -- no curl/pip needed
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

# System deps
# - libgl* for OpenCV
# - poppler-utils for pdf2image (uses pdftoppm)
# - tesseract-ocr (+ languages) for pytesseract/easyocr runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates tzdata \
      libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
      poppler-utils \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser && mkdir -p /app && chown -R appuser:appuser /app
WORKDIR /app

# ---------- Install dependencies (cached separately from app code) ----------
FROM base AS deps
# Only the dependency manifests -- editing application code shouldn't bust
# this layer's cache, only editing pyproject.toml/uv.lock should. This is an
# app, not a publishable package ([tool.uv] package = false in
# pyproject.toml), so uv doesn't need the actual source tree to resolve and
# install dependencies here.
#
# Includes the dev group (ruff/black/pytest/mypy/pre-commit): `app`'s
# make lint/format/dev-setup targets shell into this same image and expect
# them present, matching the old requirements.txt which bundled them into
# every service too.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# ---------- Runtime image ----------
FROM base AS runtime
COPY --from=deps /opt/venv /opt/venv

# Copy your source (compose will also mount it during dev)
COPY . /app
RUN chown -R appuser:appuser /app

# Drop privileges for runtime
USER appuser

# Quick sanity: tesseract available and venv active
# (not required, but handy during CI logs)
RUN tesseract --version && python -c "import sys; print(sys.executable)"

CMD ["python", "-V"]
