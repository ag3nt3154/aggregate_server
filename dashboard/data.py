# dashboard/data.py
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

_COLUMNS = [
    "id", "request_id", "timestamp", "inbound_model", "canonical_model",
    "backend_id", "status_code", "queue_time_ms", "backend_time_ms",
    "total_time_ms", "input_tokens", "output_tokens", "error_message",
]


def load_data(db_dir: str | Path, days: int = 30) -> pd.DataFrame:
    """Load last `days` of request logs from daily SQLite files.

    Returns an empty DataFrame with correct columns when no data exists.
    Silently skips missing days and unreadable files.
    """
    db_dir = Path(db_dir)
    today = datetime.now(tz=UTC).date()
    frames: list[pd.DataFrame] = []

    for offset in range(days):
        d = today - timedelta(days=offset)
        db_path = db_dir / f"{d.isoformat()}.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM requests", conn)
            conn.close()
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=_COLUMNS + ["datetime"])

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="s", utc=True)
    return combined
