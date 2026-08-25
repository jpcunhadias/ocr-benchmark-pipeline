from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..config import settings
from ..mongodb import get_mongo_db

router = APIRouter(prefix="/ocr-results", tags=["ocr-results"])


def _normalize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/")
async def list_ocr_results(
    run_id: str | None = Query(
        None, description="Filter by the run_id stored on the document."
    ),
    doc_id: str | None = Query(
        None, description="Filter by the deterministic doc_id."
    ),
    document_name: str | None = Query(
        None, description="Filter by exact document name."
    ),
    q_name: str | None = Query(
        None, description="Case-insensitive partial search on document name."
    ),
    engine: str | None = Query(None, description="Filter by OCR engine used."),
    month: str | None = Query(None, description="Filter by period (YYYY-MM)."),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    filters = []
    if run_id:
        filters.append({"run_id": run_id})
    if engine:
        filters.append({"engine": engine})
    if month:
        filters.append({"month": month})
    if document_name:
        filters.append({"document_name": document_name})
    if doc_id:
        filters.append(
            {
                "$or": [
                    {"doc_id": doc_id},
                    {"_id": doc_id},
                    {"document_name": doc_id},
                ]
            }
        )
    if q_name:
        filters.append({"document_name": {"$regex": q_name, "$options": "i"}})

    query = {"$and": filters} if filters else {}

    cursor = (
        db[settings.MONGO_COLL]
        .find(query)
        .sort("updated_at", -1)
        .skip(offset)
        .limit(limit)
    )
    results = await cursor.to_list(length=limit)
    return [_normalize_doc(r) for r in results]


@router.get("/{document_id}")
async def get_ocr_result(
    document_id: str,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    candidates = [
        {"_id": document_id},
        {"doc_id": document_id},
        {"document_name": document_id},
    ]

    result = None
    for query in candidates:
        result = await db[settings.MONGO_COLL].find_one(query)
        if result:
            break

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return _normalize_doc(result)


@router.get("/filter/")
async def filter_ocr_results(
    run_id: str,
    doc_id: str,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    query = {"doc_id": doc_id}
    if run_id:
        query["run_id"] = run_id
    result = await db[settings.MONGO_COLL].find_one(query)

    if not result and run_id:
        result = await db[settings.MONGO_COLL].find_one(
            {"_id": doc_id, "run_id": run_id}
        )

    if not result:
        fallback_queries = [
            {"_id": doc_id},
            {"document_name": doc_id},
        ]
        for fq in fallback_queries:
            candidate = await db[settings.MONGO_COLL].find_one(fq)
            if candidate:
                result = candidate
                break

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return _normalize_doc(result)
