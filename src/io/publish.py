from __future__ import annotations

import io
import os
from datetime import datetime

import boto3
import pandas as pd
from botocore.client import Config
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, PyMongoError
from sqlalchemy import create_engine, text

from src.utils.logger import get_logger
from src.utils.runtime import pipeline_publish_enabled, pipeline_use_minio

logger = get_logger(__name__)
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://app:app@postgres:5432/ocr")


def pg_engine():
    return create_engine(PG_URL, future=True)


def reset_run(run_id: str, tables: list[str]):
    """Delete previous rows for run_id from tables (idempotent loads)."""
    if not tables:
        return
    if not pipeline_publish_enabled():
        logger.debug(
            "[offline] Skipping reset_run for %s (publishing disabled)", run_id
        )
        return
    eng = pg_engine()
    with eng.begin() as con:
        for t in tables:
            con.execute(text(f"DELETE FROM {t} WHERE run_id=:r"), {"r": run_id})


def append_df(table: str, df: pd.DataFrame, dtype: dict | None = None):
    if df is None or df.empty:
        return 0

    if not pipeline_publish_enabled():
        logger.debug(
            "[offline] Skipping append_df for table %s (publishing disabled)", table
        )
        return len(df)

    df.to_sql(table, pg_engine(), if_exists="append", index=False, dtype=dtype)
    return len(df)


def register_run(run_row: dict):
    if not pipeline_publish_enabled():
        logger.debug("[offline] Skipping register_run for %s", run_row.get("run_id"))
        return
    eng = pg_engine()
    sql = """
      INSERT INTO runs(run_id, source_id, period, engine, pipeline_version, config_hash, code_version, started_at, status, num_documents, notes)
      VALUES(:run_id,:source_id,:period,:engine,:pipeline_version,:config_hash,:code_version, now(), :status, :num_documents, :notes)
      ON CONFLICT (run_id) DO UPDATE
        SET source_id=EXCLUDED.source_id,
            period=EXCLUDED.period,
            engine=EXCLUDED.engine,
            pipeline_version=EXCLUDED.pipeline_version,
            config_hash=EXCLUDED.config_hash,
            code_version=EXCLUDED.code_version,
            status=EXCLUDED.status,
            num_documents=EXCLUDED.num_documents,
            notes=EXCLUDED.notes;
    """
    with eng.begin() as con:
        con.execute(text(sql), run_row)


def finish_run(run_id: str, status: str = "OK", notes: str | None = None):
    if not pipeline_publish_enabled():
        logger.debug("[offline] Skipping finish_run for %s (status=%s)", run_id, status)
        return
    with pg_engine().begin() as con:
        con.execute(
            text(
                "UPDATE runs SET finished_at=now(), status=:s, notes=COALESCE(:n, notes) WHERE run_id=:r"
            ),
            {"s": status, "n": notes, "r": run_id},
        )


def register_document(doc_row: dict):
    if not pipeline_publish_enabled():
        logger.debug(
            "[offline] Skipping register_document for %s",
            doc_row.get("document_id"),
        )
        return
    eng = pg_engine()
    sql = """
      INSERT INTO documents(document_id, document_name, month, source, department, source_path, engine, pipeline_version, run_id, inserted_at)
      VALUES(:document_id, :document_name, :month, :source, :department, :source_path, :engine, :pipeline_version, :run_id, now())
      ON CONFLICT (document_id) DO UPDATE
        SET document_name = EXCLUDED.document_name,
            month = EXCLUDED.month,
            source = EXCLUDED.source,
            department = EXCLUDED.department,
            source_path = EXCLUDED.source_path,
            engine = EXCLUDED.engine,
            pipeline_version = EXCLUDED.pipeline_version,
            run_id = EXCLUDED.run_id,
            inserted_at = EXCLUDED.inserted_at;
    """
    with eng.begin() as con:
        con.execute(text(sql), doc_row)


# ---------- Mongo ----------
_mongo_client: MongoClient | None = None


def _get_mongo_client() -> MongoClient:
    """
    Returns a cached MongoClient instance, performing a health check on first connection.
    Raises ConnectionFailure on error.
    """
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    logger.info("Connecting to MongoDB at %s...", mongo_uri)

    try:
        client: MongoClient = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command("ismaster")
        logger.info("✓ MongoDB connection successful.")
        _mongo_client = client
        return _mongo_client
    except ConnectionFailure as e:
        logger.error(
            "Could not connect to MongoDB. Please check the MONGO_URI environment variable "
            "and ensure the database is running and accessible.",
            exc_info=True,
        )
        raise e


def get_mongo_collection():
    """Connect to MongoDB and return the collection object."""
    client = _get_mongo_client()
    db = client[os.getenv("MONGO_DB", "ocr_db")]
    return db[os.getenv("MONGO_COLL", "documents")]


def publish_document_to_mongo(doc_id: str, doc_content: dict) -> bool:
    """
    Upserts a single document into MongoDB with robust error handling.

    Returns:
        bool: True on success, False on failure.
    """
    if not doc_id:
        logger.error("Cannot publish document to MongoDB with an empty doc_id.")
        return False

    if not pipeline_publish_enabled():
        logger.debug("[offline] Skipping Mongo publish for %s", doc_id)
        return True

    try:
        coll = get_mongo_collection()
        timestamp = datetime.utcnow().isoformat()
        payload = doc_content.copy()
        payload.setdefault("doc_id", doc_id)
        payload.setdefault("document_name", doc_id)
        payload["updated_at"] = timestamp

        op = UpdateOne(
            {"_id": doc_id},
            {
                "$set": payload,
                "$setOnInsert": {"created_at": timestamp},
            },
            upsert=True,
        )
        result = coll.bulk_write([op])

        if result.upserted_count > 0:
            logger.info(f"Inserted document '{doc_id}' into MongoDB.")
        elif result.modified_count > 0:
            logger.info(f"Updated document '{doc_id}' in MongoDB.")
        return True

    except PyMongoError as e:
        logger.error(
            f"Failed to publish document '{doc_id}' to MongoDB. "
            f"Reason: {e}. Skipping this document.",
            exc_info=True,
        )
        # Optionally, log the failing document content (careful with size/secrets)
        # logger.debug("Failing payload for doc_id %s: %s", doc_id, payload)
        return False


# ---------- Parquet to MinIO ----------
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def push_parquet(df: pd.DataFrame, bucket: str, key: str):
    if df is None or df.empty:
        return

    if not pipeline_use_minio():
        logger.debug("[offline] Skipping MinIO push for %s", key)
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
