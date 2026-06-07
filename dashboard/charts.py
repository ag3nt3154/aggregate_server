# dashboard/charts.py
from __future__ import annotations

import altair as alt
import pandas as pd

_ROLLING_WINDOW = 3


def _hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["count"] = 1
    hourly = (
        df.set_index("datetime")
        .resample("1h")["count"]
        .sum()
        .reset_index()
    )
    hourly["rolling"] = hourly["count"].rolling(_ROLLING_WINDOW, min_periods=1).mean()
    return hourly


def request_count_chart(df: pd.DataFrame) -> alt.Chart:
    hourly = _hourly_counts(df)
    base = alt.Chart(hourly).encode(x=alt.X("datetime:T", title="Hour"))
    bars = base.mark_bar(opacity=0.4, color="#4c78a8").encode(
        y=alt.Y("count:Q", title="Requests")
    )
    line = base.mark_line(color="red", strokeWidth=2).encode(
        y=alt.Y("rolling:Q", title="3h rolling mean")
    )
    return alt.layer(bars, line).properties(height=250)  # type: ignore[return-value]


def token_chart(df: pd.DataFrame) -> alt.Chart:
    frames = []
    for col, label in [("input_tokens", "input"), ("output_tokens", "output")]:
        hourly = (
            df.set_index("datetime")
            .resample("1h")[col]
            .sum()
            .reset_index()
            .rename(columns={col: "tokens"})
        )
        hourly["type"] = label
        frames.append(hourly)
    combined = pd.concat(frames, ignore_index=True)
    return (  # type: ignore[no-any-return]
        alt.Chart(combined)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("datetime:T", title="Hour"),
            y=alt.Y("tokens:Q", title="Tokens"),
            color=alt.Color("type:N", legend=alt.Legend(title="Token type")),
        )
        .properties(height=250)
    )


def error_rate_chart(df: pd.DataFrame) -> alt.Chart:
    df = df.copy()
    df["is_error"] = (df["status_code"] >= 400).astype(int)
    grp = df.set_index("datetime").resample("1h")["is_error"]
    hourly = pd.DataFrame({"total": grp.count(), "errors": grp.sum()}).reset_index()
    hourly["error_rate"] = hourly["errors"] / hourly["total"].clip(lower=1)
    hourly["rolling"] = hourly["error_rate"].rolling(_ROLLING_WINDOW, min_periods=1).mean()
    base = alt.Chart(hourly).encode(x=alt.X("datetime:T", title="Hour"))
    bars = base.mark_bar(opacity=0.4, color="#e45756").encode(
        y=alt.Y("error_rate:Q", title="Error rate", axis=alt.Axis(format=".0%"))
    )
    line = base.mark_line(color="darkred", strokeWidth=2).encode(
        y=alt.Y("rolling:Q")
    )
    return alt.layer(bars, line).properties(height=250)  # type: ignore[return-value]
