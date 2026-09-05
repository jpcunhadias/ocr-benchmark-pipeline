"""On-demand page-image rendering for the localization-accuracy dashboard.

The pipeline treats converted page PNGs as disposable intermediates --
scripts.pipeline.core.process_folder() wipes them before and after every
run. The durable artifact is the source PDF in MinIO (downloading doesn't
delete it), so a specific page is re-rendered from that PDF on demand
instead of trying to retain/persist the PNG.

Two ways to resolve (run_id, document) to a MinIO object:
- Fast path: ocr_page_metrics.source_object, the exact object key recorded
  at pipeline-write time (see scripts.pipeline.core.process_folder).
- Fallback: for runs that predate that column, list objects under the
  prefixes registered on `documents` for this run_id and match by filename
  stem -- correct (the PNG-naming and doc-grouping conventions are exact
  inverses) but does a live MinIO listing instead of one direct lookup.

Raises MinioConnectionError for a genuine backend failure (unreachable
MinIO, auth failure, ...) so callers can distinguish that from an honest
"no matching PDF found" (returned as None) -- conflating the two would
make an infra outage look identical to stale/absent data.
"""

from __future__ import annotations

import io
import os
from pathlib import PurePosixPath

import urllib3
from fastapi.concurrency import run_in_threadpool
from minio import Minio
from minio.error import MinioException
from pdf2image import convert_from_bytes
from sqlalchemy import text

from ..db import AsyncSessionLocal

PDF_RENDER_DPI = 300  # must match scripts/data_prep/convert_pdfs_to_images.py

# S3Error/MinioException are minio's own error hierarchy (auth failures,
# bad requests, server-side errors). A refused/unresolvable connection --
# e.g. MinIO not running -- surfaces one level lower, as a urllib3.HTTPError
# (typically MaxRetryError wrapping a NameResolutionError), confirmed via a
# live "MinIO stopped" test: without also catching this, that failure mode
# fell through as an unhandled 500 instead of the intended 502.
_MINIO_BACKEND_ERRORS = (MinioException, urllib3.exceptions.HTTPError)


class MinioConnectionError(RuntimeError):
    """A real MinIO backend failure, distinct from "no matching object"."""


def _connect_to_minio() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    endpoint = endpoint.split("://", 1)[-1]  # Minio() wants host:port, no scheme
    return Minio(
        endpoint=endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minio"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
        secure=os.getenv("MINIO_ENDPOINT", "").startswith("https://"),
    )


async def _fetch_direct_source_object(run_id: str, document: str) -> str | None:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            text(
                """
                SELECT source_object FROM ocr_page_metrics
                WHERE run_id = :run_id AND document = :document
                  AND source_object IS NOT NULL
                LIMIT 1
                """
            ),
            {"run_id": run_id, "document": document},
        )
        row = res.first()
        return row[0] if row else None


async def _fetch_candidate_source_paths(run_id: str) -> list[str]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            text("SELECT DISTINCT source_path FROM documents WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        return [row[0] for row in res.all() if row[0]]


def _find_source_object(
    minio_client: Minio, bucket: str, source_paths: list[str], document: str
) -> str | None:
    """Find the object under any of the candidate prefixes whose filename
    stem matches `document`. Lets MinIO/urllib3 errors propagate -- a real
    connectivity failure must not look like "not found"."""
    for prefix in source_paths:
        for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
            if obj.object_name is None:
                continue
            if PurePosixPath(obj.object_name).stem == document:
                return obj.object_name
    return None


def _fetch_and_render(
    client: Minio, bucket: str, object_name: str, page: int
) -> bytes | None:
    """Download one object's bytes and rasterize a single page from it."""
    try:
        response = client.get_object(bucket, object_name)
        try:
            pdf_bytes = response.read()
        finally:
            response.close()
            response.release_conn()
    except _MINIO_BACKEND_ERRORS as err:
        raise MinioConnectionError(str(err)) from err

    images = convert_from_bytes(
        pdf_bytes, dpi=PDF_RENDER_DPI, first_page=page, last_page=page
    )
    if not images:
        return None

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def _render_direct_sync(object_name: str, page: int) -> bytes | None:
    bucket = os.getenv("MINIO_BUCKET", "ocr-artifacts")
    client = _connect_to_minio()
    return _fetch_and_render(client, bucket, object_name, page)


def _render_page_sync(
    document: str, page: int, source_paths: list[str]
) -> bytes | None:
    bucket = os.getenv("MINIO_BUCKET", "ocr-artifacts")
    client = _connect_to_minio()

    try:
        object_name = _find_source_object(client, bucket, source_paths, document)
    except _MINIO_BACKEND_ERRORS as err:
        raise MinioConnectionError(str(err)) from err

    if object_name is None:
        return None

    return _fetch_and_render(client, bucket, object_name, page)


async def render_page_image(run_id: str, document: str, page: int) -> bytes | None:
    """Render `page` of `document`'s source PDF as PNG bytes, or None if no
    matching PDF is found. Raises MinioConnectionError on a real backend
    failure. Prefers the object key recorded at write time
    (ocr_page_metrics.source_object); falls back to a live MinIO listing +
    filename-stem match for runs that predate that column."""
    object_name = await _fetch_direct_source_object(run_id, document)
    if object_name is not None:
        return await run_in_threadpool(_render_direct_sync, object_name, page)

    source_paths = await _fetch_candidate_source_paths(run_id)
    if not source_paths:
        return None

    return await run_in_threadpool(_render_page_sync, document, page, source_paths)
