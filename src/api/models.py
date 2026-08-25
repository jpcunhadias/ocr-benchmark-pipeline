from datetime import datetime

from pydantic import BaseModel, constr


class Run(BaseModel):
    run_id: str
    source_id: str | None = None
    period: constr(pattern=r"^\d{4}-\d{2}$")  # 'YYYY-MM'
    engine: str | None = None
    pipeline_version: str | None = None
    config_hash: str | None = None
    code_version: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    status: str | None = None
    num_documents: int | None = None
    notes: str | None = None


class Extraction(BaseModel):
    run_id: str
    timestamp: datetime | None = None
    document: str
    page: int | None = None
    raw_text: str | None = None
    cleaned_text: str | None = None
    char_count: int | None = None
    period: str | None = None
    doc_id: str | None = None
