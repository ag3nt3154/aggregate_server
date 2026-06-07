# SQLite Request Logging + Streamlit Dashboard — Design Spec

**Date:** 2026-06-07  
**Status:** Approved

---

## Overview

Add per-request SQLite logging to `aggregate_server` and a standalone Streamlit dashboard
that visualises request volume, token usage, response time breakdown (via linear regression),
and per-backend error rates. The dashboard depends only on the DB directory path — it imports
nothing from `aggregate_server`.

**Scope:** non-streaming requests only. Streaming requests are not logged.

---

## 1. Data Model

### Daily DB files

Location: `./data/logs/YYYY-MM-DD.db` — one SQLite file per calendar day.  
The `./data/logs/` directory is created automatically on first write.

### Schema (per daily file)

```sql
CREATE TABLE IF NOT EXISTS requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       TEXT    NOT NULL,
    timestamp        REAL    NOT NULL,
    inbound_model    TEXT    NOT NULL,
    canonical_model  TEXT    NOT NULL,
    backend_id       TEXT,
    status_code      INTEGER NOT NULL,
    queue_time_ms    REAL    NOT NULL,
    backend_time_ms  REAL,
    total_time_ms    REAL    NOT NULL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    error_message    TEXT
);
```

### Column semantics

| Column | Source | Notes |
|--------|--------|-------|
| `request_id` | `uuid.uuid4()` in router | Unique per HTTP request |
| `timestamp` | `time.time()` at enqueue | Unix epoch (wall clock) |
| `inbound_model` | `body["model"]` | Raw client-supplied model name |
| `canonical_model` | `resolve_model()` result | After alias resolution |
| `backend_id` | `entry.config.id` | NULL if request never dispatched |
| `status_code` | HTTP status returned to client | 200, 404, 502, 503, 504, etc. |
| `queue_time_ms` | `(dispatch_at - enqueue_at) × 1000` | Time waiting in asyncio.Queue |
| `backend_time_ms` | `(complete_at - dispatch_at) × 1000` | NULL if never dispatched |
| `total_time_ms` | `queue_time_ms + backend_time_ms` | NULL backend_time → use queue_time only |
| `input_tokens` | `response.json()["usage"]["prompt_tokens"]` | NULL on error or missing usage |
| `output_tokens` | `response.json()["usage"]["completion_tokens"]` | NULL on error or missing usage |
| `error_message` | Exception message / ForwardError text | NULL on success |

### Regression model

`backend_time_ms = latency + pp_speed × input_tokens + tg_speed × output_tokens`

Fitted via `np.linalg.lstsq` on rows where `backend_time_ms IS NOT NULL` and both token
columns are non-null. Design matrix: `[1, input_tokens, output_tokens]`. Outputs:
`latency (ms)`, `prompt_processing_speed (ms/token)`, `token_generation_speed (ms/token)`.

---

## 2. Logging Architecture (server-side)

### New file: `aggregate_server/log_writer.py`

**`LogRecord`** — dataclass mirroring the DB schema (all fields, typed to match).

**`LogWriter`** — owns write-side lifecycle:

- `__init__(db_dir: str | Path = "./data/logs", queue_maxsize: int = 1000)`
- `async def run() -> None` — background coroutine; drains queue in batches of ≤ 50,
  writes to today's DB (rotates automatically at midnight UTC), creates file + schema on
  first write of each day. Uses `aiosqlite`.
- `def enqueue(record: LogRecord) -> None` — `queue.put_nowait()`; on `asyncio.QueueFull`
  logs a warning and drops the record silently (never blocks the hot path).

### Changes to existing files

**`aggregate_server/dispatcher.py`**

- `PendingRequest` gains `enqueue_at: float` field (monotonic timestamp).
- `Dispatcher.__init__` gains `log_writer: LogWriter | None = None` (optional — defaults
  to None so existing tests need no changes).
- `_handle_request`: stamps `dispatch_at = time.monotonic()` at entry; after
  `forward_request` succeeds stamps `complete_at`; extracts tokens from
  `result.response.json().get("usage", {})`; calls `self._log_writer.enqueue(record)` if
  writer is set. Also logs errors (ForwardError / exhausted backends) with appropriate
  status codes and NULL token fields.

**`aggregate_server/router.py`**

- Lifespan instantiates `LogWriter`, passes it to `Dispatcher`, starts `log_writer.run()`
  as an asyncio task alongside dispatch tasks.
- `chat_completions` stamps `enqueue_at = time.monotonic()` before `dispatcher.enqueue()`,
  generates `request_id = str(uuid.uuid4())`, threads both through `PendingRequest`.
- Error paths (504 timeout, QueueFullError) also emit a `LogRecord` via the writer with
  the appropriate status code and NULL backend fields.

### Separation guarantee

`log_writer.py` has no imports from the dashboard package. The dashboard package has no
imports from `aggregate_server`. The only coupling is the DB file path and schema.

---

## 3. Dashboard Architecture

### File structure

```
dashboard.py                    # entry point: `streamlit run dashboard.py`
dashboard/
    __init__.py
    app.py                      # main layout, tab routing, autorefresh wiring
    data.py                     # load_data() — reads daily SQLite files, returns DataFrame
    charts.py                   # Altair/Plotly chart helpers
    regression.py               # fit_response_model() → RegressionResult dataclass
```

**`dashboard.py`** (root entry point):
```python
from dashboard.app import main
main()
```

### `dashboard/data.py`

```python
def load_data(db_dir: str | Path, days: int = 30) -> pd.DataFrame:
```

- Iterates last `days` calendar dates (today inclusive).
- For each date: opens `db_dir/YYYY-MM-DD.db` with `sqlite3.connect` if it exists; queries
  `SELECT * FROM requests`; skips missing files silently.
- Concatenates all results into a single `pd.DataFrame` with a parsed `datetime` column
  derived from `timestamp`.
- Returns empty DataFrame (correct columns, zero rows) when no data exists.

No imports from `aggregate_server`. Depends only on `sqlite3` (stdlib), `pandas`, `pathlib`.

### `dashboard/regression.py`

```python
@dataclass
class RegressionResult:
    latency_ms: float
    pp_speed_ms_per_token: float
    tg_speed_ms_per_token: float
    r_squared: float
    n_samples: int

def fit_response_model(df: pd.DataFrame) -> RegressionResult | None:
```

Filters to rows with non-null `backend_time_ms`, `input_tokens`, `output_tokens`.
Returns `None` when fewer than 10 samples (not enough for meaningful regression).
Uses `np.linalg.lstsq` — no sklearn dependency.

### `dashboard/app.py`

```python
st_autorefresh(interval=15_000, key="autorefresh")
db_dir = st.sidebar.text_input("DB directory", value="./data/logs")
df = load_data(db_dir, days=30)
overview_tab, backend_tab = st.tabs(["Overview", "Per Backend"])
```

**Overview tab:**
| Widget | Content |
|--------|---------|
| KPI row | Total requests · Total input tokens · Total output tokens |
| Line chart | Hourly request count + 3-hour rolling mean |
| Line chart | Hourly tokens in / tokens out |
| Regression panel | `latency`, `pp_speed`, `tg_speed`, R², sample count |

**Per-backend tab:**
| Widget | Content |
|--------|---------|
| Dropdown | Select backend ID (from distinct `backend_id` values) |
| KPI row | Requests · Input tokens · Output tokens (filtered) |
| Line chart | Hourly request count + 3-hour rolling mean |
| Line chart | Hourly error rate (status ≥ 400) + 3-hour rolling mean |

### New dependencies

Added to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
# core dependencies (server-side)
# add to [project.dependencies]:
#   "aiosqlite>=0.20.0"

[project.optional-dependencies]
dashboard = [
    "streamlit>=1.35.0",
    "streamlit-autorefresh>=1.0.1",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "altair>=5.0.0",
]
```

`aiosqlite` is a core server dependency (not optional) — add to `[project.dependencies]`.

---

## 4. File Change Summary

| File | Action |
|------|--------|
| `aggregate_server/log_writer.py` | **New** — LogRecord, LogWriter |
| `aggregate_server/dispatcher.py` | **Modify** — add enqueue_at, log_writer wiring |
| `aggregate_server/router.py` | **Modify** — lifespan wires LogWriter, stamps enqueue_at + request_id |
| `pyproject.toml` | **Modify** — add aiosqlite (core), dashboard optional group |
| `dashboard.py` | **New** — entry point |
| `dashboard/__init__.py` | **New** — empty |
| `dashboard/app.py` | **New** — Streamlit layout |
| `dashboard/data.py` | **New** — load_data() |
| `dashboard/charts.py` | **New** — chart helpers |
| `dashboard/regression.py` | **New** — fit_response_model() |
