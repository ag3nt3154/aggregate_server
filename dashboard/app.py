# dashboard/app.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.charts import error_rate_chart, request_count_chart, token_chart
from dashboard.data import load_data
from dashboard.regression import fit_response_model


def main() -> None:
    st.set_page_config(page_title="Aggregate Server Dashboard", layout="wide")
    st_autorefresh(interval=15_000, key="autorefresh")

    st.sidebar.title("Settings")
    db_dir = st.sidebar.text_input("DB directory", value="./data/logs")

    df = load_data(db_dir, days=30)
    overview_tab, backend_tab = st.tabs(["Overview", "Per Backend"])

    with overview_tab:
        _render_overview(df)

    with backend_tab:
        _render_backend(df)


def _render_overview(df: pd.DataFrame) -> None:
    st.header("Overview — last 30 days")
    if df.empty:
        st.info("No data yet. Start the server and make some non-streaming requests.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total requests", f"{len(df):,}")
    col2.metric("Total input tokens", f"{df['input_tokens'].sum():,.0f}")
    col3.metric("Total output tokens", f"{df['output_tokens'].sum():,.0f}")

    st.subheader("Hourly request count (3h rolling mean)")
    st.altair_chart(request_count_chart(df), use_container_width=True)

    st.subheader("Hourly token usage")
    st.altair_chart(token_chart(df), use_container_width=True)

    st.subheader("Response time model")
    result = fit_response_model(df)
    if result is None:
        st.info(
            "Not enough data for regression (need ≥ 10 successful non-streaming rows "
            "with token counts)."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency (ms)", f"{result.latency_ms:.1f}")
        c2.metric("Prompt speed (ms/token)", f"{result.pp_speed_ms_per_token:.4f}")
        c3.metric("Gen speed (ms/token)", f"{result.tg_speed_ms_per_token:.4f}")
        c4.metric("R²", f"{result.r_squared:.3f}")
        st.caption(f"Fitted on {result.n_samples:,} samples.")


def _render_backend(df: pd.DataFrame) -> None:
    st.header("Per backend — last 30 days")
    if df.empty or df["backend_id"].isna().all():
        st.info("No backend data available yet.")
        return

    backends = sorted(df["backend_id"].dropna().unique())
    selected = st.selectbox("Backend", backends)
    bdf = df[df["backend_id"] == selected]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total requests", f"{len(bdf):,}")
    col2.metric("Total input tokens", f"{bdf['input_tokens'].sum():,.0f}")
    col3.metric("Total output tokens", f"{bdf['output_tokens'].sum():,.0f}")

    st.subheader("Hourly request count (3h rolling mean)")
    st.altair_chart(request_count_chart(bdf), use_container_width=True)

    st.subheader("Hourly error rate (3h rolling mean)")
    st.altair_chart(error_rate_chart(bdf), use_container_width=True)
