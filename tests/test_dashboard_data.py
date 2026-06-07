# tests/test_dashboard_data.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dashboard.data import load_data

_SCHEMA = """
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    inbound_model TEXT NOT NULL,
    canonical_model TEXT NOT NULL,
    backend_id TEXT,
    status_code INTEGER NOT NULL,
    queue_time_ms REAL NOT NULL,
    backend_time_ms REAL,
    total_time_ms REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error_message TEXT
)
"""

_INSERT = """
INSERT INTO requests (
    request_id, timestamp, inbound_model, canonical_model,
    backend_id, status_code, queue_time_ms, backend_time_ms,
    total_time_ms, input_tokens, output_tokens, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _make_db(path: Path, rows: list[tuple[Any, ...]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.executemany(_INSERT, rows)
    conn.commit()
    conn.close()


def test_load_data_empty_dir_returns_empty_df(tmp_path: Path) -> None:
    df = load_data(tmp_path / "nonexistent", days=30)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "timestamp" in df.columns


def test_load_data_reads_single_db(tmp_path: Path) -> None:
    db_dir = tmp_path / "logs"
    db_dir.mkdir()
    ts = 1_749_254_400.0  # 2026-06-07 UTC
    _make_db(
        db_dir / "2026-06-07.db",
        [("req-1", ts, "gpt-4", "llama3", "b1", 200, 10.0, 500.0, 510.0, 100, 50, None)],
    )
    df = load_data(db_dir, days=30)
    assert len(df) == 1
    assert df.iloc[0]["request_id"] == "req-1"
    assert "datetime" in df.columns


def test_load_data_skips_missing_days(tmp_path: Path) -> None:
    db_dir = tmp_path / "logs"
    db_dir.mkdir()
    ts = 1_749_254_400.0
    _make_db(
        db_dir / "2026-06-07.db",
        [("req-1", ts, "m", "m", "b1", 200, 1.0, 1.0, 2.0, 1, 1, None)],
    )
    # 2026-06-05.db does not exist — should not error
    df = load_data(db_dir, days=30)
    assert len(df) == 1


def test_load_data_concatenates_multiple_dbs(tmp_path: Path) -> None:
    db_dir = tmp_path / "logs"
    db_dir.mkdir()
    ts1 = 1_749_254_400.0  # 2026-06-07
    ts2 = 1_749_168_000.0  # 2026-06-06
    _make_db(
        db_dir / "2026-06-07.db",
        [("r1", ts1, "m", "m", "b1", 200, 1.0, 1.0, 2.0, 1, 1, None)],
    )
    _make_db(
        db_dir / "2026-06-06.db",
        [("r2", ts2, "m", "m", "b1", 200, 1.0, 1.0, 2.0, 1, 1, None)],
    )
    df = load_data(db_dir, days=30)
    assert len(df) == 2
    assert set(df["request_id"]) == {"r1", "r2"}
