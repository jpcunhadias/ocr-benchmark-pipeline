# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11.13
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

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

# Create virtualenv
RUN python -m venv /opt/venv

# Non-root user
RUN useradd -m -u 1000 appuser && mkdir -p /app && chown -R appuser:appuser /app
WORKDIR /app

# ---------- Build wheels in a separate stage ----------
FROM base AS deps
USER root
# pip cache for faster builds
RUN --mount=type=cache,target=/root/.cache/pip true
COPY requirements.txt /app/requirements.txt
# Build wheels (root is fine)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && \
    pip wheel --wheel-dir=/wheels -r /app/requirements.txt

# ---------- Runtime image ----------
FROM base AS runtime
USER root
COPY --from=deps /wheels /wheels
# Install from wheels into venv, then delete wheels (as root so no perms issue)
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

# Copy your source (compose will also mount it during dev)
COPY . /app
RUN chown -R appuser:appuser /app

# Drop privileges for runtime
USER appuser

# Quick sanity: tesseract available and venv active
# (not required, but handy during CI logs)
RUN tesseract --version && python -c "import sys; print(sys.executable)"

CMD ["python", "-V"]
