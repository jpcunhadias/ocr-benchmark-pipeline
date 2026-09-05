from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

Period = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}$")]  # 'YYYY-MM'


class Run(BaseModel):
    run_id: str
    source_id: str | None = None
    period: Period
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


class PageMetric(BaseModel):
    run_id: str
    timestamp: datetime | None = None
    document: str
    engine: str | None = None
    page: int | None = None
    elapsed_sec: float | None = None
    avg_confidence: float | None = None
    char_count: int | None = None
    cer: float | None = None
    wer: float | None = None
    fields_total: int | None = None
    fields_correct: int | None = None
    field_accuracy: float | None = None
    avg_iou: float | None = None
    localization_fields_total: int | None = None
    localization_fields_correct: int | None = None


class AccuracySummary(BaseModel):
    engine: str | None = None
    total_pages: int
    labeled_pages: int
    avg_cer: float | None = None
    avg_wer: float | None = None
    avg_field_accuracy: float | None = None
    avg_iou: float | None = None


class FieldAccuracyBreakdown(BaseModel):
    engine: str | None = None
    field_name: str
    total: int
    correct: int
    accuracy: float | None = None


class LocalizationAccuracyBreakdown(BaseModel):
    engine: str | None = None
    field_name: str
    total: int
    located: int
    correct: int
    avg_iou: float | None = None


class ThroughputSummary(BaseModel):
    engine: str | None = None
    total_pages: int
    avg_sec_per_page: float | None = None
    median_sec_per_page: float | None = None
    p95_sec_per_page: float | None = None
    pages_per_sec: float | None = None


class CalibrationPoint(BaseModel):
    run_id: str
    document: str
    page: int | None = None
    engine: str | None = None
    avg_confidence: float | None = None
    confidence_normalized: float | None = None
    cer: float | None = None
    wer: float | None = None
