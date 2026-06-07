# tests/test_dashboard_regression.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.regression import RegressionResult, fit_response_model


def _make_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_returns_none_when_insufficient_samples() -> None:
    df = _make_df([
        {"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50}
        for _ in range(9)
    ])
    result = fit_response_model(df)
    assert result is None


def test_exact_fit_known_coefficients() -> None:
    # backend_time = 20 + 0.01 * input + 0.05 * output  (exact linear)
    rng = np.random.default_rng(42)
    n = 50
    inp = rng.integers(100, 1000, size=n).astype(float)
    out = rng.integers(10, 200, size=n).astype(float)
    bt = 20.0 + 0.01 * inp + 0.05 * out
    df = _make_df([
        {"backend_time_ms": bt[i], "input_tokens": inp[i], "output_tokens": out[i]}
        for i in range(n)
    ])
    result = fit_response_model(df)
    assert result is not None
    assert abs(result.latency_ms - 20.0) < 0.01
    assert abs(result.pp_speed_ms_per_token - 0.01) < 0.001
    assert abs(result.tg_speed_ms_per_token - 0.05) < 0.001
    assert result.r_squared > 0.999
    assert result.n_samples == 50


def test_filters_null_rows() -> None:
    rows: list[dict[str, Any]] = [
        {"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50},
    ] * 10
    rows += [
        {"backend_time_ms": None, "input_tokens": 100, "output_tokens": 50},
        {"backend_time_ms": 500.0, "input_tokens": None, "output_tokens": 50},
    ]
    df = _make_df(rows)
    result = fit_response_model(df)
    assert result is not None
    assert result.n_samples == 10


def test_result_is_dataclass() -> None:
    rows: list[dict[str, Any]] = [
        {"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50}
    ] * 20
    result = fit_response_model(_make_df(rows))
    assert isinstance(result, RegressionResult)
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.r_squared, float)
