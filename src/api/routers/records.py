from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.ocr_engines.utils import normalize_confidence

from ..db import AsyncSessionLocal
from ..models import AccuracySummary, CalibrationPoint, Extraction, PageMetric, Run

router = APIRouter(prefix="/records", tags=["records"])


@router.get("/runs", response_model=list[Run])
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    async with AsyncSessionLocal() as session:
        q = text(
            """
            SELECT run_id, source_id, period, engine, pipeline_version,
                   config_hash, code_version,
                   started_at, finished_at, status, num_documents, notes
            FROM runs
            ORDER BY started_at DESC
            LIMIT :limit OFFSET :offset
            """
        )
        res = await session.execute(q, {"limit": limit, "offset": offset})
        rows = res.mappings().all()  # dict-like rows
        return [Run(**row) for row in rows]


@router.get("/runs/{run_id}", response_model=Run)
async def get_run(run_id: str):
    async with AsyncSessionLocal() as session:
        q = text(
            """
            SELECT run_id, source_id, period, engine, pipeline_version,
                     config_hash, code_version, started_at, finished_at, status, num_documents, notes
            FROM runs
            WHERE run_id = :run_id
            """
        )
        res = await session.execute(q, {"run_id": run_id})
        row = res.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return Run(**row)


@router.get("/runs/{run_id}/extractions", response_model=list[Extraction])
async def get_extractions(
    run_id: str,
    period: str | None = None,
    document_prefix: str | None = None,
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    async with AsyncSessionLocal() as session:
        query_str = """
            SELECT run_id, timestamp, document, page, raw_text, cleaned_text, char_count, period, doc_id
            FROM extractions
            WHERE run_id = :run_id
        """
        params = {"run_id": run_id, "limit": limit, "offset": offset}

        if period:
            query_str += " AND period = :period"
            params["period"] = period

        if document_prefix:
            query_str += " AND document LIKE :document_prefix"
            params["document_prefix"] = f"{document_prefix}%"

        query_str += " ORDER BY document, page LIMIT :limit OFFSET :offset"

        q = text(query_str)
        res = await session.execute(q, params)
        rows = res.mappings().all()

        return [Extraction(**row) for row in rows]


@router.get("/runs/{run_id}/page-metrics", response_model=list[PageMetric])
async def get_page_metrics(
    run_id: str,
    document: str | None = None,
    only_labeled: bool = Query(
        False,
        description="Only return pages with ground-truth accuracy (cer IS NOT NULL).",
    ),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    async with AsyncSessionLocal() as session:
        query_str = """
            SELECT run_id, timestamp, document, engine, page, elapsed_sec, avg_confidence, char_count, cer, wer
            FROM ocr_page_metrics
            WHERE run_id = :run_id
        """
        params = {"run_id": run_id, "limit": limit, "offset": offset}

        if document:
            query_str += " AND document = :document"
            params["document"] = document

        if only_labeled:
            query_str += " AND cer IS NOT NULL"

        query_str += " ORDER BY document, page LIMIT :limit OFFSET :offset"

        q = text(query_str)
        res = await session.execute(q, params)
        rows = res.mappings().all()

        return [PageMetric(**row) for row in rows]


@router.get("/accuracy-summary", response_model=list[AccuracySummary])
async def get_accuracy_summary(
    period: str | None = Query(
        None, description="Restrict to runs from this period (YYYY-MM)."
    ),
):
    """Mean CER/WER per engine, over pages with a ground-truth label under data/labels/."""
    async with AsyncSessionLocal() as session:
        query_str = """
            SELECT m.engine,
                   COUNT(*) AS total_pages,
                   COUNT(m.cer) AS labeled_pages,
                   AVG(m.cer) AS avg_cer,
                   AVG(m.wer) AS avg_wer
            FROM ocr_page_metrics m
        """
        params = {}

        if period:
            query_str += " JOIN runs r ON r.run_id = m.run_id WHERE r.period = :period"
            params["period"] = period

        query_str += " GROUP BY m.engine ORDER BY avg_cer ASC NULLS LAST"

        q = text(query_str)
        res = await session.execute(q, params)
        rows = res.mappings().all()

        return [AccuracySummary(**row) for row in rows]


@router.get("/calibration", response_model=list[CalibrationPoint])
async def get_calibration_points(
    period: str | None = Query(
        None, description="Restrict to runs from this period (YYYY-MM)."
    ),
    engine: str | None = Query(None, description="Restrict to a single engine."),
    limit: int = Query(2000, ge=1, le=10000),
):
    """Per-page (confidence, cer, wer) points for labeled pages, across all
    runs -- the raw data behind a confidence-vs-accuracy calibration chart."""
    async with AsyncSessionLocal() as session:
        query_str = """
            SELECT m.run_id, m.document, m.page, m.engine, m.avg_confidence, m.cer, m.wer
            FROM ocr_page_metrics m
        """
        wheres = ["m.cer IS NOT NULL"]
        params = {"limit": limit}

        if period:
            query_str += " JOIN runs r ON r.run_id = m.run_id"
            wheres.append("r.period = :period")
            params["period"] = period

        if engine:
            wheres.append("m.engine = :engine")
            params["engine"] = engine

        query_str += " WHERE " + " AND ".join(wheres)
        query_str += " ORDER BY m.avg_confidence LIMIT :limit"

        q = text(query_str)
        res = await session.execute(q, params)
        rows = res.mappings().all()

        points = []
        for row in rows:
            row = dict(row)
            row["confidence_normalized"] = normalize_confidence(
                row["engine"], row["avg_confidence"]
            )
            points.append(CalibrationPoint(**row))
        return points
