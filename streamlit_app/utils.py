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
