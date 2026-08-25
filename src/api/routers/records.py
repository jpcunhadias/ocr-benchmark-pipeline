from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from ..db import AsyncSessionLocal
from ..models import Extraction, Run

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
