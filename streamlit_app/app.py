import streamlit as st

from utils import df_or_empty, list_extractions, sidebar_run_and_period

st.set_page_config(page_title="OCR Benchmark Dashboard", layout="wide")

st.title("🔍 OCR Benchmark Dashboard")
st.markdown("### Run history and extracted text browser")

st.info(
    """
    **How to use this dashboard:**

    1. 🔍 **Select a run**: Use the sidebar to pick which pipeline run to inspect
    2. 📊 **Browse extractions**: See the cleaned OCR text extracted per page for that run
    3. 🚀 **Run the pipeline**: See the "Run Pipeline" page (still a placeholder)
    """
)

# Sidebar for run selection
run_id, period = sidebar_run_and_period()
st.session_state["run_id"] = run_id
st.session_state["period"] = period

st.markdown("### Selected Run")
col1, col2 = st.columns(2)
with col1:
    st.metric("Run ID", run_id or "None selected")
with col2:
    st.metric("Period", period or "—")

st.divider()
st.markdown("### Recent Runs")

try:
    from utils import runs_df

    df = runs_df()
    if not df.empty:
        total_runs = len(df)
        if "status" in df.columns:
            ok_runs = len(df[df["status"] == "OK"])
            failed_runs = len(df[df["status"] == "FAILED"])

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Runs", total_runs)
            metric_col2.metric(
                "Completed",
                ok_runs,
                delta=None if ok_runs == 0 else f"{100 * ok_runs / total_runs:.1f}%",
            )
            metric_col3.metric(
                "Failed",
                failed_runs,
                delta=(
                    None if failed_runs == 0 else f"{100 * failed_runs / total_runs:.1f}%"
                ),
                delta_color="inverse",
            )

        display_cols = [
            col
            for col in [
                "run_id",
                "source_id",
                "period",
                "engine",
                "status",
                "started_at",
            ]
            if col in df.columns
        ]
        if "started_at" in df.columns:
            df = df.sort_values("started_at", ascending=False)
        st.dataframe(df[display_cols].head(20), use_container_width=True)
    else:
        st.info("No runs found yet. Run the OCR pipeline to get started!")
except Exception as e:
    st.warning(f"Could not load run data: {e}")

st.divider()
st.markdown("### Extracted Text (selected run)")

if run_id:
    rows = list_extractions(run_id=run_id, period=period)
    df_ext = df_or_empty(rows)
    if df_ext.empty:
        st.info("No extractions found for this run yet.")
    else:
        st.dataframe(
            df_ext[["document", "page", "char_count", "cleaned_text"]],
            use_container_width=True,
        )
else:
    st.info("Select a run in the sidebar to browse its extracted text.")

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
    ### OCR Benchmark Pipeline

    This project benchmarks OCR engines (Tesseract, PaddleOCR, EasyOCR) against
    scanned documents:

    1. **OCR extraction**: reads PDFs, converts pages to images, and runs the
       selected engine
    2. **Text cleanup**: normalizes OCR noise into cleaned text per page
    3. **Metrics**: tracks confidence, elapsed time, and character counts per
       engine/run
    4. **Dashboard**: this app, for browsing runs and extracted text
    """
    )
