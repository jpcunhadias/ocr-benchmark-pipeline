import json
import os
import re
import uuid
from collections.abc import MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient, errors
from pymongo.collection import Collection
from pymongo.database import Database

from src.utils.logger import get_logger

logger = get_logger(__name__)
RUN_ID = os.getenv("RUN_ID") or uuid.uuid4().hex


def get_mongo_client(uri: str | None = None) -> MongoClient:
    uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except errors.PyMongoError as e:
        raise RuntimeError(f"[ERROR] MongoDB connection failed: {e}") from e


def _iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _assert_doc_ok(doc: dict) -> None:
    # minimal, fast checks
    if not isinstance(doc.get("document_name"), str) or not doc["document_name"]:
        raise ValueError("document_name required")
    m = doc.get("month")
    if not (isinstance(m, str) and len(m) == 7 and m[4] == "-"):
        raise ValueError("month must be 'YYYY-MM'")
    if not isinstance(doc.get("pages"), list):
        raise ValueError("pages must be list after normalization")
    if doc.get("num_pages") != len(doc["pages"]):
        raise ValueError("num_pages must equal len(pages)")


# ---------- Pages normalization helpers ----------

_page_num_re = re.compile(r"(\d+)")


def _extract_num(s: str, fallback: int) -> int:
    m = _page_num_re.search(s)
    return int(m.group(1)) if m else fallback


def _pages_from_dict(pages_dict: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[tuple[int, dict[str, Any]]] = []
    for k, v in pages_dict.items():
        n = _extract_num(str(k), fallback=len(items) + 1)
        if isinstance(v, dict):
            v = {**v}
            v["page_no"] = v.get("page_no", v.get("page_number", n))
            v.pop("page_number", None)
            items.append((n, v))
        else:
            items.append((n, {"page_no": n, "text": str(v)}))
    items.sort(key=lambda t: t[0])
    return [v for _, v in items]


def _normalize_pages_in_place(doc: dict[str, Any]) -> None:
    """
    Accepts:
      - pages as dict (legacy): {"page_1": {...}}
      - pages as dict with numeric keys: {"1": {...}}
      - pages as list of dicts: [{"page_number": 1, ...}]
    Produces:
      - pages as list of dicts with 'page_no' set and sorted.
    """
    pages = doc.get("pages", {})
    if isinstance(pages, list):
        arr = []
        for i, p in enumerate(pages, 1):
            if isinstance(p, dict):
                p = {**p}
                p["page_no"] = p.get("page_no", p.get("page_number", i))
                p.pop("page_number", None)
                arr.append(p)
            else:
                arr.append({"page_no": i, "text": str(p)})
        arr.sort(key=lambda x: int(x.get("page_no", 0)))
        doc["pages"] = arr
        return

    if isinstance(pages, dict):
        doc["pages"] = _pages_from_dict(pages)
        return

    doc["pages"] = []


# ---------- Upsert (overwrite) semantics ----------


def upsert_document(coll: Collection, doc: dict) -> bool:
    # sanity: ensure the natural key exists
    if not doc.get("document_name") or not doc.get("month"):
        raise ValueError(
            "[ERROR] upsert_document: document_name and month are required"
        )

    key = {"document_name": doc["document_name"], "month": doc["month"]}
    now = _iso_utc_now()
    doc.setdefault("inserted_at", now)
    doc["updated_at"] = now

    res = coll.replace_one(key, doc, upsert=True)
    return bool(res.upserted_id or res.modified_count)


# ---------- Load & normalize a single JSON ----------


def load_json_document(json_path: Path) -> dict:
    """
    - Ensures 'month' (YYYY-MM)
    - Adds 'year' if missing
    - Normalizes 'pages' to list[dict] with 'page_no'
    - Sets 'num_pages'
    - Derives robust 'document_name'
    - Adds 'inserted_at' if missing (upsert will also add 'updated_at')
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] Invalid JSON in {json_path}: {e}") from e

    # Validate month field
    month = doc.get("month")
    if not (isinstance(month, str) and len(month) == 7 and month[4] == "-"):
        raise ValueError(
            f"[ERROR] 'month' field missing/invalid in {json_path.name}. "
            "Expected 'YYYY-MM'. Ensure pipeline provides this field."
        )

    # Add year if not present
    doc.setdefault("year", int(month.split("-")[0]))

    # Normalize pages
    _normalize_pages_in_place(doc)
    pages_list = doc["pages"]

    # Derive document_name robustly
    doc_name = None
    if pages_list:
        image_path = pages_list[0].get("image_path")
        if isinstance(image_path, str) and image_path.strip():
            stem = Path(image_path).stem
            m = re.match(r"^(.*?)(?:[_-]\d+)?$", stem)  # drop trailing _NNN / -NNN
            doc_name = m.group(1) if m else stem
    if not doc_name:
        doc_name = json_path.stem
    doc["document_name"] = doc_name

    # Invariants
    doc["num_pages"] = len(pages_list)
    doc.setdefault("inserted_at", _iso_utc_now())

    doc["run_id"] = RUN_ID

    return doc


# ---------- Indexes ----------


def _normalize_key(key_obj):
    # key_obj can be SON/dict (has .items()) or list of tuples
    seq = list(key_obj.items()) if hasattr(key_obj, "items") else list(key_obj)
    # sort to avoid order issues in comparison
    return sorted(seq, key=lambda kv: kv[0])


def _same_index(
    existing_ix: MutableMapping[str, Any], desired_ix: dict[str, Any]
) -> bool:
    def norm(ix):
        return {
            "name": ix.get("name"),
            "unique": bool(ix.get("unique", False)),
            "partialFilterExpression": ix.get("partialFilterExpression"),
            "key": _normalize_key(ix["key"]),
        }

    return norm(existing_ix) == norm(desired_ix)


def ensure_indexes(coll: Collection) -> None:
    desired = {
        "name": "uniq_docname_month",
        "key": [("document_name", 1), ("month", 1)],
        "unique": True,
        "partialFilterExpression": {
            "document_name": {"$type": "string"},
            "month": {"$type": "string"},
        },
    }

    # Check existing indexes by name
    existing_by_name = {ix["name"]: ix for ix in coll.list_indexes()}
    if desired["name"] in existing_by_name:
        ix = existing_by_name[desired["name"]]
        if not _same_index(ix, desired):
            logger.info(f"Rebuilding index {desired['name']} to match desired spec")
            coll.drop_index(desired["name"])
            coll.create_index(
                desired["key"],
                unique=desired["unique"],
                name=desired["name"],
                partialFilterExpression=desired["partialFilterExpression"],
            )
    else:
        coll.create_index(
            desired["key"],
            unique=desired["unique"],
            name=desired["name"],
            partialFilterExpression=desired["partialFilterExpression"],
        )

    # Secondary indexes (idempotent; keep simple)
    coll.create_index(
        [("month", 1), ("source", 1), ("department", 1)], name="q_month_source_dept"
    )
    coll.create_index([("status", 1), ("inserted_at", -1)], name="q_status_recent")


# ---------- Bulk insert (upsert) ----------


def insert_all_from_folder(
    folder_path: Path,
    db_name: str = "ocr_db",
    collection: str = "documents",
    mongo_uri: str | None = None,
) -> None:
    client = get_mongo_client(mongo_uri)
    db: Database = client[db_name]
    coll: Collection = db[collection]

    ensure_indexes(coll)

    upserts = 0
    skips = 0

    for json_file in folder_path.rglob("*.json"):
        try:
            doc = load_json_document(json_file)
            _assert_doc_ok(doc)
            if upsert_document(coll, doc):
                upserts += 1
            else:
                skips += 1
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")

    logger.info(f"Upserts (inserted or updated): {upserts}, No-op: {skips}")


if __name__ == "__main__":
    base_path = Path("results/tesseract")
    insert_all_from_folder(base_path)
