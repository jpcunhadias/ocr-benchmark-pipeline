# Makefile for OCR Benchmark Project

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# --- Configurable variables ---
PY        ?= python
ENGINE    ?= tesseract
DATA_DIR  ?= data/processed
SUBFOLDER ?= ""
MONTH     ?= 2025-04
RUN_ID    ?= dev-run-$(shell date +%s)
OFFLINE   ?= 0

export PYTHONPATH := $(shell pwd)

.PHONY: help clean lint format dev-setup run download convert ocr extract \
        docker-up docker-down docker-logs app-sh pg-ddl pg-load insert

help:
	@echo "Usage: make <command> [VAR=value]"
	@echo ""
	@echo "Commands:"
	@echo "  --- Docker & Setup ---"
	@echo "  make build / rebuild          Build the app image (Dockerfile)"
	@echo "  make docker-up                Start all services"
	@echo "  make docker-down              Stop all services"
	@echo "  make app-sh                   Shell into app container"
	@echo "  make dev-setup                Format & lint code inside Docker"
	@echo ""
	@echo "  --- OCR Pipeline ---"
	@echo "  make run [ENGINE=...] [OFFLINE=1] Run COMPLETE OCR pipeline: download→convert→OCR→extract→publish"
	@echo "  make download                 Download PDFs from MinIO interactively"
	@echo "  make convert SUBFOLDER=...    Convert PDFs to images"
	@echo "  make ocr [ENGINE=...] [MONTH=...]  Run OCR benchmark on processed images"
	@echo "  make extract [ENGINE=...] [MONTH=...] Extract cleaned text into tabular format"
	@echo ""
	@echo "  --- Database ---"
	@echo "  make pg-ddl                   Apply Postgres DDL (scripts/db/pg_ddl.sql)"
	@echo "  make pg-load RUN_ID=...       Load all CSVs into Postgres for a given run"
	@echo "  make insert                   Upsert OCR JSONs from results/ into Mongo"
	@echo ""
	@echo "Configurable variables:"
	@echo "  ENGINE    : OCR engine to use (default: tesseract)"
	@echo "  MONTH     : Period in YYYY-MM format (default: 2025-04)"
	@echo "  SUBFOLDER : Subfolder under data/pdf to convert"
	@echo "  RUN_ID    : ID for the pipeline run (default: dev-run-<timestamp>)"
	@echo "  OFFLINE   : Set to 1 to run without MinIO/database (default: 0)"


# ---------- Dev hygiene ----------
clean:
	find results/ -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
	find data/pdf/ -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
	find data/processed/ -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

lint:
	docker-compose exec app ruff check . --fix --exclude .venv,notebooks

format:
	docker-compose exec app black . --exclude notebooks

dev-setup: format lint

# ---------- Build & Compose ----------
build:
	docker compose build app

rebuild:
	DOCKER_BUILDKIT=1 docker compose build --no-cache app

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

app-sh:
	docker compose exec app bash

# ---------- OCR Pipeline ----------
run:
	docker-compose exec app python scripts/pipeline/pipeline_loop.py --engine $(ENGINE) $(if $(filter 1,$(OFFLINE)),--offline,)

download:
	docker-compose exec app python scripts/data_prep/download_pdfs_from_minio.py

convert:
	docker-compose exec app python scripts/data_prep/convert_pdfs_to_images.py \
		--input_dir $(if $(SUBFOLDER),data/pdf/$(SUBFOLDER),data/pdf) \
		--output_dir data/processed \
		--dpi 300

ocr:
	docker-compose exec app python scripts/ocr/run_benchmark.py \
		--config configs/engines/$(ENGINE).yaml \
		--data_dir $(DATA_DIR) \
		--output_dir results/$(ENGINE) \
		$(if $(MONTH),--month $(MONTH),)

extract:
	docker-compose exec app python scripts/ocr/extract_from_json.py \
		--engine $(ENGINE) \
		--month $(MONTH) \
		--results_dir results

# ---------- Database ----------
pg-ddl:
	cat scripts/db/pg_ddl.sql | docker compose exec -T postgres psql -U app -d ocr

pg-load:
	RUN_ID=$(RUN_ID) docker compose exec -e RUN_ID=$(RUN_ID) app python scripts/db/load_all_csvs.py

insert:
	docker-compose exec app python src/data/mongo_exporter.py
