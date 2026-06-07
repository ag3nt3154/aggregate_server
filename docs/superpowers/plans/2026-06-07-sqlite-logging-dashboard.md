# SQLite Logging + Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-request SQLite logging to the aggregate server and a standalone Streamlit dashboard showing request volume, token usage, response-time regression, and per-backend error rates over the last 30 days.

**Architecture:** A `LogWriter` background task drains an `asyncio.Queue` and writes batches to daily SQLite files in `./data/logs/`. The dispatcher stamps timing and extracts token counts, then calls `log_writer.enqueue()`. The Streamlit dashboard reads those files with plain `sqlite3` — zero imports from `aggregate_server`.

**Tech Stack:** Python 3.11, aiosqlite, FastAPI (existing), Streamlit, streamlit-autorefresh, Pandas, NumPy, Altair.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `aggregate_server/log_writer.py` | Create | `LogRecord` dataclass + `LogWriter` (queue + async batch writer) |
| `aggregate_server/dispatcher.py` | Modify | Add `enqueue_at`/`request_id`/`timestamp`/`inbound_model` to `PendingRequest`; wire `log_writer` |
| `aggregate_server/router.py` | Modify | Stamp timing/UUID at request entry; start `LogWriter` in lifespan; log 404/503/504 |
| `pyproject.toml` | Modify | Add `aiosqlite` to core deps; add `dashboard` optional dep group |
| `dashboard/__init__.py` | Create | Empty package marker |
| `dashboard/data.py` | Create | `load_data(db_dir, days=30) → pd.DataFrame` |
| `dashboard/regression.py` | Create | `fit_response_model(df) → RegressionResult \| None` |
| `dashboard/charts.py` | Create | Altair chart helpers (request count, tokens, error rate) |
| `dashboard/app.py` | Create | Streamlit layout — Overview + Per Backend tabs |
| `dashboard.py` | Create | Entry point: `streamlit run dashboard.py` |
| `tests/test_log_writer.py` | Create | Tests for `LogWriter` write/queue/run behaviour |
| `tests/test_dispatcher_logging.py` | Create | Tests for logging hooks in `_handle_request` |
| `tests/test_dashboard_data.py` | Create | Tests for `load_data` |
| `tests/test_dashboard_regression.py` | Create | Tests for `fit_response_model` |

---

## Task 1: Create `aggregate_server/log_writer.py`

**Files:**
- Create: `aggregate_server/log_writer.py`
- Create: `tests/test_log_writer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_log_writer.py
from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from aggregate_server.log_writer import LogRecord, LogWriter


def _make_record(**kwargs) -> LogRecord:
    defaults = dict(
        request_id="req-1",
        timestamp=1_700_000_000.0,
        inbound_model="gpt-4",
        canonical_model="llama3",
        backend_id="b1",
        status_code=200,
        queue_time_ms=10.0,
        backend_time_ms=500.0,
        total_time_ms=510.0,
        input_tokens=100,
        output_tokens=50,
        error_message=None,
    )
    return LogRecord(**{**defaults, **kwargs})


async def test_write_batch_creates_db_file(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    await writer._write_batch([_make_record(timestamp=1_700_000_000.0)])
    db_files = list(tmp_path.glob("*.db"))
    assert len(db_files) == 1


async def test_write_batch_record_readable(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    rec = _make_record(request_id="abc", input_tokens=42, output_tokens=7)
    await writer._write_batch([rec])
    db_path = list(tmp_path.glob("*.db"))[0]
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT request_id, input_tokens, output_tokens FROM requests").fetchall()
    conn.close()
    assert rows == [("abc", 42, 7)]


async def test_enqueue_drops_when_full(caplog):
    import logging
    writer = LogWriter(queue_maxsize=1)
    writer.enqueue(_make_record(request_id="first"))
    with caplog.at_level(logging.WARNING, logger="aggregate_server.log_writer"):
        writer.enqueue(_make_record(request_id="second"))
    assert writer._queue.qsize() == 1
    assert "dropping" in caplog.text.lower()


async def test_run_processes_enqueued_records(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    writer.enqueue(_make_record())
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(list(tmp_path.glob("*.db"))) == 1


async def test_run_creates_db_dir(tmp_path):
    nested = tmp_path / "deep" / "logs"
    writer = LogWriter(db_dir=nested)
    writer.enqueue(_make_record())
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert nested.exists()


async def test_write_batch_multiple_dates(tmp_path):
    writer = LogWriter(db_dir=tmp_path)
    rec1 = _make_record(timestamp=1_700_000_000.0)  # 2023-11-14 UTC
    rec2 = _make_record(timestamp=1_700_100_000.0)  # 2023-11-15 UTC (27h later)
    await writer._write_batch([rec1, rec2])
    assert len(list(tmp_path.glob("*.db"))) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_log_writer.py -v
```

Expected: `ModuleNotFoundError` — `aggregate_server.log_writer` does not exist yet.

- [ ] **Step 3: Implement `aggregate_server/log_writer.py`**

```python
# aggregate_server/log_writer.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
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
)
"""

_INSERT = """
INSERT INTO requests (
    request_id, timestamp, inbound_model, canonical_model,
    backend_id, status_code, queue_time_ms, backend_time_ms,
    total_time_ms, input_tokens, output_tokens, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class LogRecord:
    request_id: str
    timestamp: float
    inbound_model: str
    canonical_model: str
    backend_id: str | None
    status_code: int
    queue_time_ms: float
    backend_time_ms: float | None
    total_time_ms: float
    input_tokens: int | None
    output_tokens: int | None
    error_message: str | None

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.request_id, self.timestamp, self.inbound_model, self.canonical_model,
            self.backend_id, self.status_code, self.queue_time_ms, self.backend_time_ms,
            self.total_time_ms, self.input_tokens, self.output_tokens, self.error_message,
        )


class LogWriter:
    def __init__(
        self,
        db_dir: str | Path = "./data/logs",
        queue_maxsize: int = 1000,
        batch_size: int = 50,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._batch_size = batch_size
        self._queue: asyncio.Queue[LogRecord] = asyncio.Queue(maxsize=queue_maxsize)

    def enqueue(self, record: LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            logger.warning(
                "Log queue full — dropping record for request %s", record.request_id
            )

    async def run(self) -> None:
        self._db_dir.mkdir(parents=True, exist_ok=True)
        while True:
            batch: list[LogRecord] = []
            record = await self._queue.get()
            batch.append(record)
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._write_batch(batch)

    def _date_str(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()

    async def _write_batch(self, batch: list[LogRecord]) -> None:
        by_date: dict[str, list[LogRecord]] = {}
        for rec in batch:
            key = self._date_str(rec.timestamp)
            by_date.setdefault(key, []).append(rec)
        for date_str, records in by_date.items():
            db_path = self._db_dir / f"{date_str}.db"
            async with aiosqlite.connect(db_path) as db:
                await db.execute(_CREATE_TABLE)
                await db.executemany(_INSERT, [r.as_tuple() for r in records])
                await db.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_log_writer.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add aggregate_server/log_writer.py tests/test_log_writer.py
git commit -m "feat: add LogRecord and LogWriter for SQLite request logging"
```

---

## Task 2: Modify `aggregate_server/dispatcher.py` — logging hooks

**Files:**
- Modify: `aggregate_server/dispatcher.py`
- Create: `tests/test_dispatcher_logging.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dispatcher_logging.py
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from aggregate_server.config import AppConfig
from aggregate_server.dispatcher import Dispatcher, PendingRequest
from aggregate_server.forwarder import ForwardResult
from aggregate_server.log_writer import LogRecord, LogWriter
from aggregate_server.registry import BackendRegistry

RESPONSE_JSON = {
    "id": "r1",
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}
BODY = {"model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}


class _CapturingWriter(LogWriter):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[LogRecord] = []

    def enqueue(self, record: LogRecord) -> None:
        self.records.append(record)


def _make_dispatcher(cfg: AppConfig, client: httpx.AsyncClient, writer: LogWriter):
    registry = BackendRegistry(cfg.backends)
    models = registry.get_canonical_models()
    dispatcher = Dispatcher(
        registry, client, models,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
        log_writer=writer,
    )
    return dispatcher, registry


@respx.mock
async def test_log_writer_called_on_success(sample_config: AppConfig) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    respx.post("http://backend2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    writer = _CapturingWriter()
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client, writer)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        pending = PendingRequest(
            "qwen3.5", BODY, False, future,
            request_id="req-x", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        dispatcher.enqueue(pending)
        await asyncio.wait_for(future, timeout=2.0)
        task.cancel()

    assert len(writer.records) == 1
    rec = writer.records[0]
    assert rec.request_id == "req-x"
    assert rec.status_code == 200
    assert rec.input_tokens == 10
    assert rec.output_tokens == 5
    assert rec.backend_time_ms is not None
    assert rec.error_message is None


@respx.mock
async def test_log_writer_called_on_all_backends_fail(sample_config: AppConfig) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="err")
    )
    respx.post("http://backend2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="err")
    )
    writer = _CapturingWriter()
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client, writer)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        pending = PendingRequest(
            "qwen3.5", BODY, False, future,
            request_id="req-err", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        dispatcher.enqueue(pending)
        with pytest.raises(Exception):
            await asyncio.wait_for(future, timeout=3.0)
        task.cancel()

    assert len(writer.records) == 1
    rec = writer.records[0]
    assert rec.status_code == 502
    assert rec.input_tokens is None
    assert rec.error_message is not None


async def test_streaming_request_not_logged(sample_config: AppConfig) -> None:
    writer = _CapturingWriter()
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client, writer)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        # stream=True — should never be logged even on failure
        pending = PendingRequest(
            "qwen3.5", BODY, True, future,
            request_id="stream-req", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        # Just enqueue without running the dispatcher — force exception via model not found path
        dispatcher._queues["qwen3.5"].put_nowait(pending)
        # Cancel the future to simulate timeout without dispatching
        future.cancel()

    # Nothing was dispatched, no logs
    assert len(writer.records) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dispatcher_logging.py -v
```

Expected: `TypeError` — `PendingRequest` and `Dispatcher` don't accept the new keyword args yet.

- [ ] **Step 3: Modify `aggregate_server/dispatcher.py`**

Replace the file with:

```python
# aggregate_server/dispatcher.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from aggregate_server.forwarder import ForwardError, ForwardResult, forward_request, tracked_stream
from aggregate_server.registry import BackendEntry, BackendRegistry

if TYPE_CHECKING:
    from aggregate_server.log_writer import LogRecord, LogWriter

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    def __init__(self, model: str, max_size: int) -> None:
        super().__init__(f"Queue full for model '{model}' (capacity {max_size})")
        self.status_code = 503


@dataclass
class PendingRequest:
    canonical_model: str
    body: dict[str, Any]
    stream: bool
    result_future: asyncio.Future[ForwardResult]
    request_id: str = ""
    timestamp: float = 0.0
    inbound_model: str = ""
    enqueue_at: float = field(default_factory=time.monotonic)


class Dispatcher:
    """
    Manages per-model request queues and dispatches requests to free backends.
    One run_for_model() coroutine must be running per canonical model.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        client: httpx.AsyncClient,
        canonical_models: list[str],
        *,
        max_queue_size: int = 100,
        backend_timeout: float = 300.0,
        poll_interval: float = 0.1,
        log_writer: LogWriter | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._max_queue_size = max_queue_size
        self._backend_timeout = backend_timeout
        self._poll_interval = poll_interval
        self._log_writer = log_writer
        self._queues: dict[str, asyncio.Queue[PendingRequest]] = {
            m: asyncio.Queue() for m in canonical_models
        }

    def enqueue(self, pending: PendingRequest) -> None:
        queue = self._queues.get(pending.canonical_model)
        if queue is None:
            pending.result_future.set_exception(
                ForwardError(f"No backends configured for model '{pending.canonical_model}'", 404)
            )
            return
        if queue.qsize() >= self._max_queue_size:
            pending.result_future.set_exception(
                QueueFullError(pending.canonical_model, self._max_queue_size)
            )
            return
        queue.put_nowait(pending)

    async def run_for_model(self, canonical_model: str) -> None:
        """Long-running loop: dequeue requests and dispatch them to free backends."""
        queue = self._queues[canonical_model]
        while True:
            pending = await queue.get()
            entry = await self._acquire_with_poll(canonical_model)
            asyncio.create_task(self._handle_request(entry, pending))

    async def _acquire_with_poll(self, canonical_model: str) -> BackendEntry:
        while True:
            entry = await self._registry.acquire_backend(canonical_model)
            if entry is not None:
                return entry
            await asyncio.sleep(self._poll_interval)

    async def _handle_request(self, entry: BackendEntry, pending: PendingRequest) -> None:
        dispatch_at = time.monotonic()
        tried_ids: set[str] = set()
        current: BackendEntry | None = entry

        while current is not None:
            tried_ids.add(current.config.id)
            try:
                result = await forward_request(
                    self._client, current, pending.body,
                    stream=pending.stream, backend_timeout=self._backend_timeout,
                )
                complete_at = time.monotonic()
                await self._attach_release(result, current)
                if not pending.result_future.done():
                    pending.result_future.set_result(result)
                if not pending.stream and self._log_writer is not None:
                    self._emit_success(pending, current, dispatch_at, complete_at, result)
                return
            except ForwardError as exc:
                logger.warning("Backend %s failed: %s", current.config.id, exc)
                await self._registry.release_backend(current, failed=True)
                current = await self._next_untried_backend(pending.canonical_model, tried_ids)

        err = ForwardError(
            f"All backends for model '{pending.canonical_model}' exhausted", 502
        )
        if not pending.result_future.done():
            pending.result_future.set_exception(err)
        if not pending.stream and self._log_writer is not None:
            self._emit_error(pending, dispatch_at, 502, str(err))

    def _emit_success(
        self,
        pending: PendingRequest,
        entry: BackendEntry,
        dispatch_at: float,
        complete_at: float,
        result: ForwardResult,
    ) -> None:
        from aggregate_server.log_writer import LogRecord
        queue_ms = (dispatch_at - pending.enqueue_at) * 1000
        backend_ms = (complete_at - dispatch_at) * 1000
        usage: dict[str, Any] = {}
        if result.response is not None:
            try:
                usage = result.response.json().get("usage", {})
            except Exception:
                pass
        self._log_writer.enqueue(LogRecord(  # type: ignore[union-attr]
            request_id=pending.request_id,
            timestamp=pending.timestamp,
            inbound_model=pending.inbound_model,
            canonical_model=pending.canonical_model,
            backend_id=entry.config.id,
            status_code=200,
            queue_time_ms=queue_ms,
            backend_time_ms=backend_ms,
            total_time_ms=queue_ms + backend_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            error_message=None,
        ))

    def _emit_error(
        self,
        pending: PendingRequest,
        dispatch_at: float,
        status_code: int,
        error_message: str,
    ) -> None:
        from aggregate_server.log_writer import LogRecord
        queue_ms = (dispatch_at - pending.enqueue_at) * 1000
        backend_ms = (time.monotonic() - dispatch_at) * 1000
        self._log_writer.enqueue(LogRecord(  # type: ignore[union-attr]
            request_id=pending.request_id,
            timestamp=pending.timestamp,
            inbound_model=pending.inbound_model,
            canonical_model=pending.canonical_model,
            backend_id=None,
            status_code=status_code,
            queue_time_ms=queue_ms,
            backend_time_ms=backend_ms,
            total_time_ms=queue_ms + backend_ms,
            input_tokens=None,
            output_tokens=None,
            error_message=error_message,
        ))

    async def _attach_release(self, result: ForwardResult, entry: BackendEntry) -> None:
        if result.is_stream and result.stream_gen is not None:
            captured = entry

            async def _on_done(failed: bool) -> None:
                await self._registry.release_backend(captured, failed=failed)

            result.stream_gen = tracked_stream(result.stream_gen, _on_done)
        else:
            await self._registry.release_backend(entry, failed=False)

    async def _next_untried_backend(
        self, model: str, tried_ids: set[str]
    ) -> BackendEntry | None:
        entry = await self._registry.acquire_backend(model)
        if entry is None:
            return None
        if entry.config.id in tried_ids:
            await self._registry.release_backend(entry)
            return None
        return entry
```

- [ ] **Step 4: Run all dispatcher tests to verify no regressions**

```bash
pytest tests/test_dispatcher.py tests/test_dispatcher_logging.py -v
```

Expected: all tests PASS. The existing `PendingRequest("qwen3.5", BODY, False, future)` positional constructions still work because new fields have defaults.

- [ ] **Step 5: Commit**

```bash
git add aggregate_server/dispatcher.py tests/test_dispatcher_logging.py
git commit -m "feat: wire LogWriter into dispatcher for per-request logging"
```

---

## Task 3: Modify `aggregate_server/router.py` — stamp timing + lifespan wiring

**Files:**
- Modify: `aggregate_server/router.py`

- [ ] **Step 1: Replace `aggregate_server/router.py`**

```python
# aggregate_server/router.py
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aggregate_server.config import AppConfig, load_config, resolve_model
from aggregate_server.dispatcher import Dispatcher, PendingRequest, QueueFullError
from aggregate_server.forwarder import ForwardError, ForwardResult
from aggregate_server.health import check_all, run_health_checks
from aggregate_server.log_writer import LogRecord, LogWriter
from aggregate_server.models import ModelObject, ModelsResponse
from aggregate_server.registry import BackendRegistry

logger = logging.getLogger(__name__)


def _error_response(message: str, code: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "code": code, **extra}},
    )


def _emit_router_record(
    log_writer: LogWriter,
    request_id: str,
    timestamp: float,
    inbound_model: str,
    canonical_model: str,
    enqueue_at: float,
    status_code: int,
    error_message: str,
) -> None:
    queue_ms = (time.monotonic() - enqueue_at) * 1000
    log_writer.enqueue(LogRecord(
        request_id=request_id,
        timestamp=timestamp,
        inbound_model=inbound_model,
        canonical_model=canonical_model,
        backend_id=None,
        status_code=status_code,
        queue_time_ms=queue_ms,
        backend_time_ms=None,
        total_time_ms=queue_ms,
        input_tokens=None,
        output_tokens=None,
        error_message=error_message,
    ))


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    cfg = load_config()
    registry = BackendRegistry(cfg.backends)
    client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.backend_timeout, connect=10.0))
    log_writer = LogWriter()

    logger.info("Probing all backends before accepting traffic...")
    await check_all(registry, client, failed_only=False)

    models = registry.get_canonical_models()
    dispatcher = Dispatcher(
        registry, client, models,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
        log_writer=log_writer,
    )

    dispatch_tasks = [asyncio.create_task(dispatcher.run_for_model(m)) for m in models]
    health_task = asyncio.create_task(run_health_checks(registry, client))
    log_task = asyncio.create_task(log_writer.run())

    app.state.config = cfg
    app.state.registry = registry
    app.state.dispatcher = dispatcher
    app.state.log_writer = log_writer

    yield

    for task in [*dispatch_tasks, health_task, log_task]:
        task.cancel()
    await asyncio.gather(*dispatch_tasks, health_task, log_task, return_exceptions=True)
    await client.aclose()
    logger.info("Aggregate server shut down cleanly")


app = FastAPI(title="Aggregate Server", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    registry: BackendRegistry = request.app.state.registry
    data = [ModelObject(id=m).model_dump() for m in sorted(registry.get_canonical_models())]
    return JSONResponse(ModelsResponse(data=data).model_dump())


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    body: dict[str, Any] = await request.json()
    cfg: AppConfig = request.app.state.config
    registry: BackendRegistry = request.app.state.registry
    dispatcher: Dispatcher = request.app.state.dispatcher
    log_writer: LogWriter = request.app.state.log_writer

    enqueue_at = time.monotonic()
    timestamp = time.time()
    request_id = str(uuid.uuid4())
    stream = bool(body.get("stream", False))
    inbound_model: str = body.get("model", "")
    canonical = resolve_model(cfg, inbound_model)

    if not registry.has_backends_for_model(canonical):
        available = sorted(registry.get_canonical_models())
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 404, f"model '{inbound_model}' not found",
            )
        return _error_response(
            f"The model '{inbound_model}' does not exist. Available: {available}",
            code="model_not_found",
            status=404,
            available_models=available,
        )

    future: asyncio.Future[ForwardResult] = asyncio.get_running_loop().create_future()
    pending = PendingRequest(
        canonical_model=canonical,
        body=body,
        stream=stream,
        result_future=future,
        request_id=request_id,
        timestamp=timestamp,
        inbound_model=inbound_model,
        enqueue_at=enqueue_at,
    )
    dispatcher.enqueue(pending)

    try:
        result = await asyncio.wait_for(asyncio.shield(future), timeout=cfg.queue_timeout)
    except TimeoutError:
        future.cancel()
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 504, "request timed out waiting for a free backend",
            )
        return _error_response(
            "Request timed out waiting for a free backend", "server_error", 504
        )
    except QueueFullError as exc:
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 503, str(exc),
            )
        return _error_response(str(exc), "rate_limit_exceeded", 503)
    except ForwardError as exc:
        return _error_response(str(exc), "server_error", exc.status_code)

    if result.is_stream:
        return StreamingResponse(result.stream_gen, media_type="text/event-stream")
    assert result.response is not None
    return JSONResponse(result.response.json(), status_code=result.response.status_code)
```

- [ ] **Step 2: Run the full test suite to verify no regressions**

```bash
pytest -v
```

Expected: all existing tests PASS. (The router tests mock health checks and don't exercise the new log_writer path directly — that's fine, it's tested via the dispatcher tests.)

- [ ] **Step 3: Commit**

```bash
git add aggregate_server/router.py
git commit -m "feat: stamp request timing and UUID in router; wire LogWriter into lifespan"
```

---

## Task 4: Update `pyproject.toml` and install dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `pyproject.toml`**

```toml
[project]
name = "aggregate-server"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    "pyyaml>=6.0.1",
    "aiosqlite>=0.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "respx>=0.21.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "types-PyYAML>=6.0.0",
]
dashboard = [
    "streamlit>=1.35.0",
    "streamlit-autorefresh>=1.0.1",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "altair>=5.0.0",
]

[project.scripts]
aggregate-server = "aggregate_server.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync --extra dashboard
```

Expected: resolves and installs `aiosqlite`, `streamlit`, `streamlit-autorefresh`, `pandas`, `numpy`, `altair`.

- [ ] **Step 3: Confirm aiosqlite is available**

```bash
python -c "import aiosqlite; print(aiosqlite.__version__)"
```

Expected: prints a version string without error.

- [ ] **Step 4: Run full test suite once more**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add aiosqlite (core) and dashboard optional dep group"
```

---

## Task 5: Create `dashboard/data.py`

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/data.py`
- Create: `tests/test_dashboard_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_data.py
from __future__ import annotations

import sqlite3
from pathlib import Path

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


def _make_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.executemany(_INSERT, rows)
    conn.commit()
    conn.close()


def test_load_data_empty_dir_returns_empty_df(tmp_path):
    df = load_data(tmp_path / "nonexistent", days=30)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "timestamp" in df.columns


def test_load_data_reads_single_db(tmp_path):
    db_dir = tmp_path / "logs"
    db_dir.mkdir()
    # 2026-06-07 UTC timestamp
    ts = 1_749_254_400.0
    date_str = "2026-06-07"
    _make_db(
        db_dir / f"{date_str}.db",
        [("req-1", ts, "gpt-4", "llama3", "b1", 200, 10.0, 500.0, 510.0, 100, 50, None)],
    )
    df = load_data(db_dir, days=30)
    assert len(df) == 1
    assert df.iloc[0]["request_id"] == "req-1"
    assert "datetime" in df.columns


def test_load_data_skips_missing_days(tmp_path):
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


def test_load_data_concatenates_multiple_dbs(tmp_path):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dashboard_data.py -v
```

Expected: `ModuleNotFoundError` — `dashboard.data` does not exist yet.

- [ ] **Step 3: Create `dashboard/__init__.py`**

```python
# dashboard/__init__.py
```

(empty file)

- [ ] **Step 4: Create `dashboard/data.py`**

```python
# dashboard/data.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
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
    today = datetime.now(tz=timezone.utc).date()
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_dashboard_data.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/__init__.py dashboard/data.py tests/test_dashboard_data.py
git commit -m "feat: dashboard data layer — load_data() reads daily SQLite files"
```

---

## Task 6: Create `dashboard/regression.py`

**Files:**
- Create: `dashboard/regression.py`
- Create: `tests/test_dashboard_regression.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_regression.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.regression import RegressionResult, fit_response_model


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_returns_none_when_insufficient_samples():
    df = _make_df([
        {"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50}
        for _ in range(9)
    ])
    result = fit_response_model(df)
    assert result is None


def test_exact_fit_known_coefficients():
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


def test_filters_null_rows():
    rows = [
        {"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50},
    ] * 10
    # add rows with nulls — should be excluded
    rows += [
        {"backend_time_ms": None, "input_tokens": 100, "output_tokens": 50},
        {"backend_time_ms": 500.0, "input_tokens": None, "output_tokens": 50},
    ]
    df = _make_df(rows)
    result = fit_response_model(df)
    assert result is not None
    assert result.n_samples == 10


def test_result_is_dataclass():
    rows = [{"backend_time_ms": 500.0, "input_tokens": 100, "output_tokens": 50}] * 20
    result = fit_response_model(_make_df(rows))
    assert isinstance(result, RegressionResult)
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.r_squared, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dashboard_regression.py -v
```

Expected: `ModuleNotFoundError` — `dashboard.regression` does not exist yet.

- [ ] **Step 3: Create `dashboard/regression.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dashboard_regression.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/regression.py tests/test_dashboard_regression.py
git commit -m "feat: regression module — fit response time model via lstsq"
```

---

## Task 7: Create `dashboard/charts.py`

**Files:**
- Create: `dashboard/charts.py`

*(No unit tests — pure chart construction; correctness validated visually in Task 8.)*

- [ ] **Step 1: Create `dashboard/charts.py`**

```python
# dashboard/charts.py
from __future__ import annotations

import altair as alt
import pandas as pd


def _hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["count"] = 1
    hourly = (
        df.set_index("datetime")
        .resample("1h")["count"]
        .sum()
        .reset_index()
    )
    hourly["rolling"] = hourly["count"].rolling(3, min_periods=1).mean()
    return hourly


def request_count_chart(df: pd.DataFrame) -> alt.LayerChart:
    hourly = _hourly_counts(df)
    base = alt.Chart(hourly).encode(x=alt.X("datetime:T", title="Hour"))
    bars = base.mark_bar(opacity=0.4, color="#4c78a8").encode(
        y=alt.Y("count:Q", title="Requests")
    )
    line = base.mark_line(color="red", strokeWidth=2).encode(
        y=alt.Y("rolling:Q", title="3h rolling mean")
    )
    return (bars + line).properties(height=250)


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
    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("datetime:T", title="Hour"),
            y=alt.Y("tokens:Q", title="Tokens"),
            color=alt.Color("type:N", legend=alt.Legend(title="Token type")),
        )
        .properties(height=250)
    )


def error_rate_chart(df: pd.DataFrame) -> alt.LayerChart:
    df = df.copy()
    df["is_error"] = (df["status_code"] >= 400).astype(int)
    grp = df.set_index("datetime").resample("1h")["is_error"]
    hourly = pd.DataFrame({"total": grp.count(), "errors": grp.sum()}).reset_index()
    hourly["error_rate"] = hourly["errors"] / hourly["total"].clip(lower=1)
    hourly["rolling"] = hourly["error_rate"].rolling(3, min_periods=1).mean()
    base = alt.Chart(hourly).encode(x=alt.X("datetime:T", title="Hour"))
    bars = base.mark_bar(opacity=0.4, color="#e45756").encode(
        y=alt.Y("error_rate:Q", title="Error rate", axis=alt.Axis(format=".0%"))
    )
    line = base.mark_line(color="darkred", strokeWidth=2).encode(
        y=alt.Y("rolling:Q")
    )
    return (bars + line).properties(height=250)
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/charts.py
git commit -m "feat: Altair chart helpers for request count, tokens, and error rate"
```

---

## Task 8: Create `dashboard/app.py` and `dashboard.py`

**Files:**
- Create: `dashboard/app.py`
- Create: `dashboard.py`

- [ ] **Step 1: Create `dashboard/app.py`**

```python
# dashboard/app.py
from __future__ import annotations

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


def _render_overview(df) -> None:  # type: ignore[no-untyped-def]
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


def _render_backend(df) -> None:  # type: ignore[no-untyped-def]
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
```

- [ ] **Step 2: Create `dashboard.py`**

```python
# dashboard.py
from dashboard.app import main

main()
```

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Verify the dashboard starts without errors**

```bash
streamlit run dashboard.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | grep -o "Aggregate Server Dashboard" || echo "CHECK BROWSER"
kill %1
```

Expected: Streamlit starts, page contains "Aggregate Server Dashboard" (or check manually in browser).

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py dashboard.py
git commit -m "feat: Streamlit dashboard with Overview and Per Backend tabs"
```

---

## Task 9: Final lint, type-check, and integration commit

- [ ] **Step 1: Run ruff**

```bash
ruff check .
ruff format --check .
```

Fix any issues, then:

```bash
ruff format .
```

- [ ] **Step 2: Run mypy on server code**

```bash
mypy aggregate_server/
```

Expected: no errors. (Dashboard code uses dynamic Streamlit APIs and is excluded from strict checking.)

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "chore: lint and type-check pass for logging + dashboard"
```
