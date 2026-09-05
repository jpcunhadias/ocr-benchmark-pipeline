import pandas as pd
import streamlit as st

from utils import (
    accuracy_summary,
    calibration_points,
    df_or_empty,
    list_extractions,
    list_page_metrics,
    sidebar_run_and_period,
    throughput_summary,
)

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
                    None
                    if failed_runs == 0
                    else f"{100 * failed_runs / total_runs:.1f}%"
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
st.markdown("### Accuracy by Engine")
st.caption(
    "Mean CER/WER against ground-truth labels under data/labels/, across all runs "
    "for the selected period. Only pages with a matching label file count."
)

acc_rows = accuracy_summary(period=period)
df_acc = df_or_empty(acc_rows)
if df_acc.empty:
    st.info(
        "No accuracy data yet. Add ground-truth labels under "
        "data/labels/<document>/<document>_<page>.txt and re-run the pipeline."
    )
else:
    display_acc = df_acc.copy()
    for col in ("avg_cer", "avg_wer"):
        if col in display_acc.columns:
            display_acc[col] = display_acc[col].astype(float).round(4)
    st.dataframe(display_acc, use_container_width=True)

st.divider()
st.markdown("### Confidence Calibration")
st.caption(
    "Does each engine's self-reported confidence actually track its accuracy? "
    "Confidence is normalized to 0-1 (Tesseract reports 0-100 natively, EasyOCR "
    "reports 0-1) so engines are comparable. A well-calibrated engine should show "
    "higher confidence on lower-error pages."
)

cal_rows = calibration_points(period=period)
df_cal = df_or_empty(cal_rows)
chart_df = (
    df_cal.dropna(subset=["confidence_normalized", "cer"])
    if not df_cal.empty
    else df_cal
)
if chart_df.empty:
    st.info(
        "No labeled pages with both confidence and accuracy data yet. Add "
        "ground-truth labels under data/labels/ and re-run the pipeline."
    )
else:
    st.scatter_chart(chart_df, x="confidence_normalized", y="cer", color="engine")

    corr_rows = []
    for eng, group in chart_df.groupby("engine"):
        has_variance = group["confidence_normalized"].nunique() > 1
        corr = (
            group["confidence_normalized"].corr(group["cer"])
            if len(group) >= 2 and has_variance
            else None
        )
        corr_rows.append(
            {"engine": eng, "pages": len(group), "confidence_vs_cer_corr": corr}
        )
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)
    st.caption(
        "confidence_vs_cer_corr near -1 means higher confidence reliably predicts "
        "lower error for that engine; near 0 means its confidence score isn't a "
        "useful accuracy signal."
    )

st.divider()
st.markdown("### Throughput")
st.caption(
    "Pages processed per second by engine, across all runs for the selected "
    "period. Median/p95 show per-page latency spread, not just the average."
)

tp_rows = throughput_summary(period=period)
df_tp = df_or_empty(tp_rows)
if df_tp.empty:
    st.info("No timing data yet. Run the OCR pipeline to populate this.")
else:
    display_tp = df_tp.copy()
    for col in (
        "avg_sec_per_page",
        "median_sec_per_page",
        "p95_sec_per_page",
        "pages_per_sec",
    ):
        if col in display_tp.columns:
            display_tp[col] = display_tp[col].astype(float).round(3)
    st.bar_chart(display_tp.set_index("engine")["pages_per_sec"])
    st.dataframe(display_tp, use_container_width=True)

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

st.divider()
st.markdown("### Page Accuracy (selected run)")

if run_id:
    metric_rows = list_page_metrics(run_id, only_labeled=True)
    df_metrics = df_or_empty(metric_rows)
    if df_metrics.empty:
        st.info("No labeled pages for this run.")
    else:
        st.dataframe(
            df_metrics[["document", "page", "cer", "wer", "avg_confidence"]],
            use_container_width=True,
        )
else:
    st.info("Select a run in the sidebar to see its per-page accuracy.")

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
       engine/run, plus CER/WER accuracy against ground-truth labels where
       available
    4. **Dashboard**: this app, for browsing runs and extracted text
    """
    )
