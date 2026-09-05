import os
import socket
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st


def _normalize_api_url(url: str) -> str:
    """Fallback to localhost when Docker service hostname is unreachable."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return url

    try:
        socket.gethostbyname(host)
        return url
    except socket.gaierror:
        pass

    # Common local fallback when streamlit runs outside docker-compose
    if host == "api":
        netloc = "localhost"
        if parsed.port:
            netloc = f"localhost:{parsed.port}"
        fallback = parsed._replace(netloc=netloc)
        return urlunparse(fallback)

    return url


_env_api_url = os.getenv("OCR_API_URL", "http://localhost:8080")
API_URL = _normalize_api_url(_env_api_url)
if _env_api_url != API_URL:
    print(
        f"[streamlit] OCR_API_URL fallback: unable to resolve '{_env_api_url}', using '{API_URL}' instead."
    )
API_KEY = os.getenv("OCR_API_KEY", "dev-secret")


def api_get(path: str, params: dict | None = None) -> list | dict:
    url = f"{API_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"X-API-Key": API_KEY}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on GET {path}: {e}")
        return []


@st.cache_data(ttl=60)
def list_runs(limit=200):
    return api_get("/records/runs", {"limit": limit}) or []


def runs_df():
    rows = list_runs()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def list_extractions(
    run_id: str,
    period: str | None = None,
    document_prefix: str | None = None,
    limit: int = 500,
    offset: int = 0,
):
    params = {
        "period": period,
        "document_prefix": document_prefix,
        "limit": limit,
        "offset": offset,
    }
    return api_get(f"/records/runs/{run_id}/extractions", params)


@st.cache_data(ttl=60)
def list_page_metrics(
    run_id: str,
    document: str | None = None,
    only_labeled: bool = False,
    limit: int = 1000,
    offset: int = 0,
):
    params = {
        "document": document,
        "only_labeled": only_labeled,
        "limit": limit,
        "offset": offset,
    }
    return api_get(f"/records/runs/{run_id}/page-metrics", params)


@st.cache_data(ttl=60)
def accuracy_summary(period: str | None = None):
    params = {"period": period} if period else None
    return api_get("/records/accuracy-summary", params)


@st.cache_data(ttl=60)
def calibration_points(period: str | None = None, engine: str | None = None):
    params = {"period": period, "engine": engine}
    return api_get("/records/calibration", params)


@st.cache_data(ttl=60)
def throughput_summary(period: str | None = None):
    params = {"period": period} if period else None
    return api_get("/records/throughput-summary", params)


@st.cache_data(ttl=60)
def field_accuracy_breakdown(period: str | None = None):
    params = {"period": period} if period else None
    return api_get("/records/field-accuracy", params)


@st.cache_data(ttl=60)
def localization_accuracy_breakdown(period: str | None = None):
    params = {"period": period} if period else None
    return api_get("/records/localization-accuracy", params)


@st.cache_data(ttl=60)
def localization_results(run_id: str, document: str, page: int):
    params = {"document": document, "page": page}
    return api_get(f"/records/runs/{run_id}/localization-results", params)


@st.cache_data(ttl=None)
def page_image(run_id: str, document: str, page: int) -> bytes | None:
    """Fetch the rendered page image. Unlike the other cached fetchers
    (ttl=60, tuned for metrics that could still change), (run_id, document,
    page) permanently determines this output once a run is finished, so no
    TTL is needed. A dedicated fetcher, not api_get() -- that helper
    unconditionally calls .json() and routes failures through st.error,
    neither of which fits a binary response where "not found" is routine."""
    url = f"{API_URL.rstrip('/')}/records/runs/{run_id}/page-image"
    headers = {"X-API-Key": API_KEY}
    try:
        r = requests.get(
            url,
            headers=headers,
            params={"document": document, "page": page},
            timeout=30,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        st.error(f"API error fetching page image: {e}")
        return None


def df_or_empty(rows) -> pd.DataFrame:
    return pd.DataFrame(rows) if isinstance(rows, list) and rows else pd.DataFrame()


def sidebar_run_and_period():
    st.sidebar.header("Filters")
    df_runs = runs_df()

    run_opts = df_runs["run_id"].tolist() if not df_runs.empty else []

    try:
        index = run_opts.index(st.session_state.get("run_id"))
    except (ValueError, KeyError):
        index = 0 if run_opts else None

    run_id = st.sidebar.selectbox("Run ID", run_opts, index=index)

    # try to get period for selected run (if present on runs table)
    period = None
    if not df_runs.empty and run_id:
        row = df_runs.loc[df_runs["run_id"] == run_id]
        if not row.empty and "period" in row.columns:
            period = row["period"].iloc[0] or None

    # allow override
    period = st.sidebar.text_input("Period (YYYY-MM)", value=period or "")

    return run_id, (period or None)
