"""Run the offline OCR pipeline (convert -> OCR -> extract) from the dashboard.

Reuses run_local_pipeline.py's functions directly -- the same code path
already exercised by `python run_local_pipeline.py --step all ...` -- so
this page is a UI in front of it, not a second implementation. Deliberately
offline only (no MinIO/Postgres): triggering the real production pipeline
from here would need the streamlit container to hold database/object-store
credentials it doesn't have.
"""

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_local_pipeline as local_pipeline  # noqa: E402

st.title("Run Pipeline")
st.caption(
    "Runs the offline pipeline (no MinIO/Postgres) against a PDF already "
    "under data/pdf/, or one you upload below. Same code path as "
    "`python run_local_pipeline.py --step all`."
)

ENGINE_OPTIONS = sorted(p.stem for p in Path("configs/engines").glob("*.yaml"))

if not ENGINE_OPTIONS:
    st.error("No engine configs found under configs/engines/.")
    st.stop()

default_engine_index = (
    ENGINE_OPTIONS.index("tesseract") if "tesseract" in ENGINE_OPTIONS else 0
)
engine = st.selectbox("Engine", ENGINE_OPTIONS, index=default_engine_index)

st.divider()
st.markdown("### Choose a document")

tab_existing, tab_upload = st.tabs(["Use an existing PDF folder", "Upload a new PDF"])

selected_input_dir: Path | None = None

with tab_existing:
    pdf_root = Path("data/pdf")
    existing_dirs = sorted(
        d for d in pdf_root.rglob("*") if d.is_dir() and list(d.glob("*.pdf"))
    )
    if not existing_dirs:
        st.info(
            "No PDFs found under data/pdf/ yet. Use the upload tab, or copy "
            "PDFs into data/pdf/<period>/<document>/ and reload this page."
        )
    else:
        labels = [str(d.relative_to(pdf_root)) for d in existing_dirs]
        picked = st.selectbox("Folder", labels)
        selected_input_dir = existing_dirs[labels.index(picked)]

with tab_upload:
    period = st.text_input("Period (YYYY-MM)", value="2025-04")
    doc_name = st.text_input("Document folder name", value="uploaded-doc")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded and period and doc_name:
        upload_dir = Path("data/pdf") / period / doc_name
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        st.success(f"Saved to {dest}")
        selected_input_dir = upload_dir

st.divider()

run_clicked = st.button(
    "Run pipeline", type="primary", disabled=selected_input_dir is None
)

if run_clicked and selected_input_dir is not None:
    local_pipeline.setup_directories()
    processed_dir = Path("data/processed") / selected_input_dir.name
    results_dir = Path("results") / engine / selected_input_dir.name
    config_path = f"configs/engines/{engine}.yaml"

    log = io.StringIO()
    with st.spinner("Running pipeline..."), contextlib.redirect_stdout(log):
        local_pipeline.convert_pdfs(str(selected_input_dir), str(processed_dir))
        results = local_pipeline.run_ocr(
            config_path, str(processed_dir), str(results_dir)
        )
        local_pipeline.extract_fields(str(results_dir), engine)

    st.success(f"Pipeline completed -- {len(results)} page(s) processed.")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        display_cols = [
            c
            for c in [
                "document",
                "page",
                "elapsed_sec",
                "avg_confidence",
                "n_chars",
                "has_ground_truth",
                "cer",
                "wer",
            ]
            if c in df_results.columns
        ]
        st.dataframe(df_results[display_cols], use_container_width=True)

    extracted_file = results_dir / "extracted_text.txt"
    if extracted_file.exists():
        st.markdown("### Extracted text preview")
        st.text(extracted_file.read_text()[:5000])

    with st.expander("Log"):
        st.code(log.getvalue())

    st.caption(f"Results saved under {results_dir}")
