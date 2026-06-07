# dashboard/regression.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_SAMPLES = 10


@dataclass
class RegressionResult:
    latency_ms: float
    pp_speed_ms_per_token: float
    tg_speed_ms_per_token: float
    r_squared: float
    n_samples: int


def fit_response_model(df: pd.DataFrame) -> RegressionResult | None:
    """Fit: backend_time_ms = latency + pp_speed*input_tokens + tg_speed*output_tokens.

    Returns None when fewer than _MIN_SAMPLES usable rows exist.
    """
    mask = (
        df["backend_time_ms"].notna()
        & df["input_tokens"].notna()
        & df["output_tokens"].notna()
    )
    sub = df[mask]
    if len(sub) < _MIN_SAMPLES:
        return None

    X = np.column_stack([
        np.ones(len(sub)),
        sub["input_tokens"].to_numpy(dtype=float),
        sub["output_tokens"].to_numpy(dtype=float),
    ])
    y = sub["backend_time_ms"].to_numpy(dtype=float)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    latency, pp_speed, tg_speed = coeffs

    ss_res = float(np.sum((y - X @ coeffs) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return RegressionResult(
        latency_ms=float(latency),
        pp_speed_ms_per_token=float(pp_speed),
        tg_speed_ms_per_token=float(tg_speed),
        r_squared=r_squared,
        n_samples=len(sub),
    )
