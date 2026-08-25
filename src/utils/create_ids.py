from __future__ import annotations

import hashlib


def make_run_id(source_id: str, period: str, engine: str, pipeline_version: str = "1"):
    base = f"{source_id}|{period}|{engine}|{pipeline_version}".lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def make_doc_id(document_name: str, period: str) -> str:
    """Creates a deterministic document ID."""
    base = f"{document_name}|{period}".lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()
