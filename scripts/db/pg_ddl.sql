-- =========================================
-- Drop existing objects
-- =========================================
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS ocr_document_stats CASCADE;
DROP TABLE IF EXISTS ocr_page_metrics CASCADE;
DROP TABLE IF EXISTS extractions CASCADE;
DROP TABLE IF EXISTS runs CASCADE;

-- =========================================
-- OCR Benchmark – Local Dev DDL
-- Safe to re-run (IF NOT EXISTS everywhere)
-- =========================================

-- Pipeline runs
CREATE TABLE IF NOT EXISTS runs (
  run_id            TEXT PRIMARY KEY,
  source_id         TEXT,
  period            CHAR(7),
  engine            TEXT,
  pipeline_version  TEXT,
  config_hash       TEXT,
  code_version      TEXT,
  started_at        TIMESTAMPTZ DEFAULT NOW(),
  finished_at       TIMESTAMPTZ,
  status            TEXT,
  num_documents     INT,
  notes             TEXT
);

-- Documents processed by a run
CREATE TABLE IF NOT EXISTS documents (
  document_id     TEXT PRIMARY KEY,
  document_name   TEXT NOT NULL,
  month           CHAR(7) NOT NULL,
  source          TEXT,
  department      TEXT,
  source_path     TEXT,
  engine          TEXT,
  pipeline_version TEXT,
  run_id          TEXT,
  inserted_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_month_source ON documents(month, source);

-- Per-document OCR runtime stats
CREATE TABLE IF NOT EXISTS ocr_document_stats (
  run_id          TEXT,
  document        TEXT NOT NULL,
  engine          TEXT,
  num_pages       INT,
  elapsed_sec     NUMERIC,
  cpu_percent     NUMERIC,
  memory_percent  NUMERIC,
  PRIMARY KEY (run_id, document)
);
CREATE INDEX IF NOT EXISTS idx_ocr_document_stats_engine ON ocr_document_stats(engine);

-- Per-page OCR metrics
CREATE TABLE IF NOT EXISTS ocr_page_metrics (
  run_id          TEXT,
  timestamp       TIMESTAMPTZ,
  document        TEXT NOT NULL,
  engine          TEXT,
  page            INT,
  elapsed_sec     NUMERIC,
  avg_confidence  NUMERIC,
  char_count      INT,
  cer             NUMERIC,  -- character error rate vs. data/labels ground truth; NULL when unlabeled
  wer             NUMERIC,  -- word error rate vs. data/labels ground truth; NULL when unlabeled
  PRIMARY KEY (run_id, document, page)
);
CREATE INDEX IF NOT EXISTS idx_ocr_page_metrics_engine ON ocr_page_metrics(engine);

-- Cleaned OCR text extracted per page
CREATE TABLE IF NOT EXISTS extractions (
  run_id        TEXT,
  timestamp     TIMESTAMPTZ,
  document      TEXT NOT NULL,
  page          INT,
  raw_text      TEXT,
  cleaned_text  TEXT,
  char_count    INT,
  period        CHAR(7),
  doc_id        TEXT,
  PRIMARY KEY (run_id, document, page)
);
CREATE INDEX IF NOT EXISTS idx_extractions_filter ON extractions(run_id, document, page);
