# Multi-Model Aliases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one model alias to map to multiple canonical backend models so requests are dispatched round-robin across the combined pool of matching backends.

**Architecture:** Widen `model_aliases` to `dict[str, str | list[str]]` (normalised to `dict[str, list[str]]` at load time); `resolve_model` returns `list[str]`; the dispatcher queues requests by a sorted-join key over canonical model lists; the registry acquires from any matching backend across the whole list.

**Tech Stack:** Python, Pydantic v2, asyncio, FastAPI, httpx, pytest, ruff, mypy

---

## File Map

| File | Change |
|------|--------|
| `aggregate_server/config.py` | Normalise `model_aliases` values to `list[str]`; `resolve_model` returns `list[str]`; add `get_callable_models` and `get_model_groups` helpers |
| `aggregate_server/registry.py` | `acquire_backend` takes `list[str]`; rename `has_backends_for_model` → `has_backends_for_models(list[str])` |
| `aggregate_server/dispatcher.py` | `PendingRequest.canonical_models: list[str]`; `Dispatcher.__init__` takes `canonical_model_groups: list[list[str]]`; queue key = `",".join(sorted(canonical_models))`; expose `queue_keys` property |
| `aggregate_server/router.py` | Use `resolve_model` returning `list[str]`; use `has_backends_for_models`; use `get_model_groups` / `get_callable_models`; `/v1/models` lists aliases |
| `tests/conftest.py` | No structural change needed — `AppConfig` still accepts `str` values; normalisation is internal |
| `tests/test_config.py` | Update assertions for normalised `model_aliases`; update `resolve_model` return type checks; add tests for list alias form and new helpers |
| `tests/test_registry.py` | Update all `acquire_backend("model")` → `acquire_backend(["model"])`; add multi-model acquire test; add `has_backends_for_models` tests |
| `tests/test_dispatcher.py` | Update `PendingRequest` positional arg to list; update `_make_dispatcher` to use `get_model_groups` |
| `tests/test_dispatcher_logging.py` | Same `PendingRequest` and `_make_dispatcher` updates |

---

## Task 1: Config layer — normalise aliases, update `resolve_model`, add helpers

**Files:**
- Modify: `aggregate_server/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py` (replace the three `test_resolve_model_*` tests and the alias assertion in `test_load_config_valid`, then add new tests):

```python
from aggregate_server.config import (
    AppConfig, BackendConfig, get_callable_models, get_model_groups,
    load_config, resolve_model,
)


# --- updated existing tests ---

def test_load_config_valid(tmp_path: Path) -> None:
    path = _write_config(tmp_path, """
        backends:
          - id: b1
            url: http://host:8080
            api_key: key1
            model: gpt4
        model_aliases:
          gpt-4: gpt4
        queue_timeout: 30
    """)
    cfg = load_config(path)
    assert len(cfg.backends) == 1
    assert cfg.backends[0].id == "b1"
    assert cfg.model_aliases == {"gpt-4": ["gpt4"]}   # <-- was "gpt4"
    assert cfg.queue_timeout == 30.0


def test_resolve_model_with_alias(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "qwen-chat") == ["qwen3.5"]


def test_resolve_model_no_alias(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "qwen3.5") == ["qwen3.5"]


def test_resolve_model_unknown_passthrough(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "unknown-model") == ["unknown-model"]


# --- new tests ---

def test_list_alias_normalised() -> None:
    cfg = AppConfig(
        backends=[
            BackendConfig(id="b1", url="http://h:8080", api_key="k", model="qwen3.5-9b"),
            BackendConfig(id="b2", url="http://h:8080", api_key="k", model="qwen3.5-9b-q8"),
        ],
        model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
    )
    assert cfg.model_aliases == {"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]}


def test_string_alias_normalised_to_list() -> None:
    cfg = AppConfig(
        backends=[BackendConfig(id="b1", url="http://h:8080", api_key="k", model="qwen3.5")],
        model_aliases={"qwen-chat": "qwen3.5"},
    )
    assert cfg.model_aliases == {"qwen-chat": ["qwen3.5"]}


def test_resolve_model_list_alias() -> None:
    cfg = AppConfig(
        backends=[
            BackendConfig(id="b1", url="http://h:8080", api_key="k", model="qwen3.5-9b"),
            BackendConfig(id="b2", url="http://h:8080", api_key="k", model="qwen3.5-9b-q8"),
        ],
        model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
    )
    assert resolve_model(cfg, "qwen3.5") == ["qwen3.5-9b", "qwen3.5-9b-q8"]


def test_get_callable_models_aliases_only() -> None:
    """Aliased canonicals are hidden; only alias keys and un-aliased canonicals appear."""
    cfg = AppConfig(
        backends=[
            BackendConfig(id="b1", url="http://h:8080", api_key="k", model="qwen3.5-9b"),
            BackendConfig(id="b2", url="http://h:8080", api_key="k", model="qwen3.5-9b-q8"),
            BackendConfig(id="b3", url="http://h:8080", api_key="k", model="llama3"),
        ],
        model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
    )
    result = get_callable_models(cfg, ["qwen3.5-9b", "qwen3.5-9b-q8", "llama3"])
    assert result == ["qwen3.5", "llama3"]


def test_get_callable_models_unaliased_canonical_included() -> None:
    """Canonical model with no alias still appears."""
    cfg = AppConfig(
        backends=[BackendConfig(id="b1", url="http://h:8080", api_key="k", model="llama3")],
    )
    result = get_callable_models(cfg, ["llama3"])
    assert result == ["llama3"]


def test_get_model_groups_mixed() -> None:
    cfg = AppConfig(
        backends=[
            BackendConfig(id="b1", url="http://h:8080", api_key="k", model="qwen3.5-9b"),
            BackendConfig(id="b2", url="http://h:8080", api_key="k", model="qwen3.5-9b-q8"),
            BackendConfig(id="b3", url="http://h:8080", api_key="k", model="llama3"),
        ],
        model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
    )
    groups = get_model_groups(cfg, ["qwen3.5-9b", "qwen3.5-9b-q8", "llama3"])
    assert ["qwen3.5-9b", "qwen3.5-9b-q8"] in groups
    assert ["llama3"] in groups
    assert len(groups) == 2
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_config.py -v
```

Expected: multiple failures — `ImportError` on `get_callable_models`/`get_model_groups`, assertion errors on `resolve_model` returning `str` instead of `list[str]`, and `model_aliases` value being `"gpt4"` not `["gpt4"]`.

- [ ] **Step 3: Implement config changes**

Replace the contents of `aggregate_server/config.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import AnyHttpUrl, BaseModel, model_validator


class BackendConfig(BaseModel):
    id: str
    url: AnyHttpUrl
    api_key: str
    model: str


class AppConfig(BaseModel):
    backends: list[BackendConfig]
    model_aliases: dict[str, str | list[str]] = {}
    queue_timeout: float = 60.0
    backend_timeout: float = 300.0
    max_queue_size: int = 100

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> AppConfig:
        ids = [b.id for b in self.backends]
        if len(ids) != len(set(ids)):
            dupes = {x for x in ids if ids.count(x) > 1}
            raise ValueError(f"Duplicate backend IDs: {dupes}")
        return self

    @model_validator(mode="after")
    def _normalise_aliases(self) -> AppConfig:
        self.model_aliases = {
            k: ([v] if isinstance(v, str) else list(v))
            for k, v in self.model_aliases.items()
        }
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.yaml. Raises FileNotFoundError or ValidationError on failure."""
    resolved = Path(path if path is not None else os.getenv("CONFIG_PATH", "config.yaml"))
    raw = yaml.safe_load(resolved.read_text())
    return AppConfig.model_validate(raw)


def resolve_model(config: AppConfig, inbound_model: str) -> list[str]:
    """Resolve an inbound model name through aliases to its canonical form(s)."""
    return config.model_aliases.get(inbound_model, [inbound_model])


def get_callable_models(config: AppConfig, canonical_models: list[str]) -> list[str]:
    """Models a client may name: alias keys + un-aliased canonicals."""
    aliased = {c for targets in config.model_aliases.values() for c in targets}
    unaliased = sorted(m for m in canonical_models if m not in aliased)
    return sorted(config.model_aliases.keys()) + unaliased


def get_model_groups(config: AppConfig, canonical_models: list[str]) -> list[list[str]]:
    """All dispatch groups: each alias's target list + single-element groups for un-aliased."""
    aliased = {c for targets in config.model_aliases.values() for c in targets}
    unaliased = [[m] for m in canonical_models if m not in aliased]
    return list(config.model_aliases.values()) + unaliased
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add aggregate_server/config.py tests/test_config.py
git commit -m "feat: normalise model_aliases to list[str]; resolve_model returns list[str]"
```

---

## Task 2: Registry — multi-model `acquire_backend` and `has_backends_for_models`

**Files:**
- Modify: `aggregate_server/registry.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

Replace the full contents of `tests/test_registry.py`:

```python
from __future__ import annotations

import asyncio

from aggregate_server.config import AppConfig, BackendConfig
from aggregate_server.registry import BackendRegistry, BackendState


def _registry(cfg: AppConfig) -> BackendRegistry:
    return BackendRegistry(cfg.backends)


async def test_acquire_returns_free_backend(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend(["qwen3.5"])
    assert entry is not None
    assert entry.state == BackendState.BUSY


async def test_acquire_returns_none_when_none_free(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    await reg.acquire_backend(["llama3"])
    result = await reg.acquire_backend(["llama3"])
    assert result is None


async def test_round_robin_prefers_oldest(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    first = await reg.acquire_backend(["qwen3.5"])
    assert first is not None
    await reg.release_backend(first)

    second = await reg.acquire_backend(["qwen3.5"])
    assert second is not None
    assert second.config.id != first.config.id


async def test_release_marks_free(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend(["qwen3.5"])
    assert entry is not None
    await reg.release_backend(entry)
    assert entry.state == BackendState.FREE


async def test_release_failed_marks_failed(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend(["qwen3.5"])
    assert entry is not None
    await reg.release_backend(entry, failed=True)
    assert entry.state == BackendState.FAILED


async def test_failed_backend_not_acquired(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    e1 = await reg.acquire_backend(["qwen3.5"])
    assert e1 is not None
    await reg.release_backend(e1, failed=True)
    e2 = await reg.acquire_backend(["qwen3.5"])
    assert e2 is not None
    assert e2.config.id != e1.config.id


async def test_restore_backend(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend(["qwen3.5"])
    assert entry is not None
    await reg.release_backend(entry, failed=True)
    await reg.restore_backend(entry)
    assert entry.state == BackendState.FREE


async def test_concurrent_acquire_no_double_claim(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    results = await asyncio.gather(
        reg.acquire_backend(["qwen3.5"]),
        reg.acquire_backend(["qwen3.5"]),
    )
    busy = [r for r in results if r is not None]
    ids = [r.config.id for r in busy]
    assert len(ids) == len(set(ids)), "Same backend claimed twice concurrently"


async def test_acquire_across_multiple_models() -> None:
    """acquire_backend picks free backends across a list of canonical models."""
    configs = [
        BackendConfig(id="a1", url="http://a1:8080", api_key="k", model="qwen3.5-9b"),
        BackendConfig(id="a2", url="http://a2:8080", api_key="k", model="qwen3.5-9b-q8"),
    ]
    reg = BackendRegistry(configs)
    e1 = await reg.acquire_backend(["qwen3.5-9b", "qwen3.5-9b-q8"])
    e2 = await reg.acquire_backend(["qwen3.5-9b", "qwen3.5-9b-q8"])
    assert e1 is not None
    assert e2 is not None
    assert e1.config.id != e2.config.id
    e3 = await reg.acquire_backend(["qwen3.5-9b", "qwen3.5-9b-q8"])
    assert e3 is None


async def test_has_backends_for_models(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    assert reg.has_backends_for_models(["qwen3.5"]) is True
    assert reg.has_backends_for_models(["qwen3.5", "llama3"]) is True
    assert reg.has_backends_for_models(["no-such-model"]) is False
    assert reg.has_backends_for_models(["no-such-model", "qwen3.5"]) is True
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_registry.py -v
```

Expected: failures — `acquire_backend` receives `list[str]` but currently expects `str`; `has_backends_for_models` not defined.

- [ ] **Step 3: Implement registry changes**

Replace the full contents of `aggregate_server/registry.py`:

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto

from aggregate_server.config import BackendConfig


class BackendState(Enum):
    FREE = auto()
    BUSY = auto()
    FAILED = auto()


@dataclass
class BackendEntry:
    config: BackendConfig
    state: BackendState = BackendState.FREE
    last_used_at: float = field(default_factory=float)


class BackendRegistry:
    """
    Async-safe registry tracking the state of all backend endpoints.

    Round-robin selection picks the FREE backend with the smallest last_used_at
    (monotonic time) across all supplied canonical model names.
    """

    def __init__(self, configs: list[BackendConfig]) -> None:
        self._entries: list[BackendEntry] = [BackendEntry(config=c) for c in configs]
        self._lock = asyncio.Lock()

    async def acquire_backend(self, canonical_models: list[str]) -> BackendEntry | None:
        """Atomically claim the longest-idle FREE backend for any of the given models."""
        async with self._lock:
            candidates = [
                e for e in self._entries
                if e.state == BackendState.FREE and e.config.model in canonical_models
            ]
            if not candidates:
                return None
            entry = min(candidates, key=lambda e: e.last_used_at)
            entry.state = BackendState.BUSY
            entry.last_used_at = time.monotonic()
            return entry

    async def release_backend(self, entry: BackendEntry, *, failed: bool = False) -> None:
        """Return a backend to FREE (or FAILED), recording the release time."""
        async with self._lock:
            entry.state = BackendState.FAILED if failed else BackendState.FREE
            entry.last_used_at = time.monotonic()

    async def restore_backend(self, entry: BackendEntry) -> None:
        """Restore a FAILED backend to FREE (called by health checker on success)."""
        async with self._lock:
            if entry.state == BackendState.FAILED:
                entry.state = BackendState.FREE

    async def list_all(self) -> list[BackendEntry]:
        """Return a snapshot of all entries (any state). Used by health checker."""
        async with self._lock:
            return list(self._entries)

    def get_canonical_models(self) -> list[str]:
        """Return the set of canonical model names across all configured backends."""
        return list({e.config.model for e in self._entries})

    def has_backends_for_models(self, canonical_models: list[str]) -> bool:
        """Check (without locking) whether any backend is configured for any of these models."""
        return any(e.config.model in canonical_models for e in self._entries)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_registry.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add aggregate_server/registry.py tests/test_registry.py
git commit -m "feat: acquire_backend accepts list[str]; add has_backends_for_models"
```

---

## Task 3: Dispatcher — multi-model queues and `PendingRequest.canonical_models`

**Files:**
- Modify: `aggregate_server/dispatcher.py`
- Modify: `tests/test_dispatcher.py`
- Modify: `tests/test_dispatcher_logging.py`

- [ ] **Step 1: Write failing tests**

Replace the full contents of `tests/test_dispatcher.py`:

```python
from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from aggregate_server.config import AppConfig, BackendConfig, get_model_groups
from aggregate_server.dispatcher import Dispatcher, PendingRequest, QueueFullError
from aggregate_server.forwarder import ForwardError, ForwardResult
from aggregate_server.registry import BackendRegistry

RESPONSE_JSON = {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
BODY = {"model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}


def _make_dispatcher(
    cfg: AppConfig, client: httpx.AsyncClient
) -> tuple[Dispatcher, BackendRegistry]:
    registry = BackendRegistry(cfg.backends)
    groups = get_model_groups(cfg, registry.get_canonical_models())
    dispatcher = Dispatcher(
        registry, client, groups,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
    )
    return dispatcher, registry


@respx.mock
async def test_successful_dispatch(sample_config: AppConfig) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    respx.post("http://backend2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )

    async with httpx.AsyncClient() as client:
        dispatcher, registry = _make_dispatcher(sample_config, client)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        pending = PendingRequest(["qwen3.5"], BODY, False, future)

        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        dispatcher.enqueue(pending)
        result = await asyncio.wait_for(future, timeout=2.0)
        task.cancel()

    assert not result.is_stream
    assert result.response is not None


async def test_queue_full_sets_exception(sample_config: AppConfig) -> None:
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client)
        loop = asyncio.get_running_loop()
        for _ in range(sample_config.max_queue_size):
            future: asyncio.Future[ForwardResult] = loop.create_future()
            dispatcher.enqueue(PendingRequest(["qwen3.5"], BODY, False, future))

        overflow_future: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest(["qwen3.5"], BODY, False, overflow_future))

        assert overflow_future.done()
        with pytest.raises(QueueFullError):
            overflow_future.result()


async def test_unknown_model_sets_error(sample_config: AppConfig) -> None:
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest(["no-such-model"], BODY, False, future))

        assert future.done()
        with pytest.raises(ForwardError) as exc_info:
            future.result()
        assert exc_info.value.status_code == 404


@respx.mock
async def test_escalates_to_next_backend_on_failure(sample_config: AppConfig) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="err")
    )
    respx.post("http://backend2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )

    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest(["qwen3.5"], BODY, False, future))

        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        result = await asyncio.wait_for(future, timeout=3.0)
        task.cancel()

    assert result.response is not None


@respx.mock
async def test_dispatch_across_multi_model_alias() -> None:
    """Requests for a multi-model alias group route to backends of any matching canonical."""
    configs = [
        BackendConfig(id="a1", url="http://a1:8080", api_key="k", model="qwen3.5-9b"),
        BackendConfig(id="a2", url="http://a2:8080", api_key="k", model="qwen3.5-9b-q8"),
    ]
    cfg = AppConfig(
        backends=configs,
        model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
        max_queue_size=10,
        backend_timeout=5.0,
        queue_timeout=5.0,
    )
    respx.post("http://a1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    respx.post("http://a2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    queue_key = "qwen3.5-9b,qwen3.5-9b-q8"

    async with httpx.AsyncClient() as client:
        registry = BackendRegistry(configs)
        groups = get_model_groups(cfg, registry.get_canonical_models())
        dispatcher = Dispatcher(registry, client, groups, max_queue_size=10, backend_timeout=5.0)

        loop = asyncio.get_running_loop()
        f1: asyncio.Future[ForwardResult] = loop.create_future()
        f2: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest(["qwen3.5-9b", "qwen3.5-9b-q8"], BODY, False, f1))
        dispatcher.enqueue(PendingRequest(["qwen3.5-9b", "qwen3.5-9b-q8"], BODY, False, f2))

        task = asyncio.create_task(dispatcher.run_for_model(queue_key))
        r1, r2 = await asyncio.wait_for(asyncio.gather(f1, f2), timeout=3.0)
        task.cancel()

    assert r1.response is not None
    assert r2.response is not None
```

Replace the full contents of `tests/test_dispatcher_logging.py`:

```python
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from aggregate_server.config import AppConfig, get_model_groups
from aggregate_server.dispatcher import Dispatcher, PendingRequest
from aggregate_server.forwarder import ForwardError, ForwardResult
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


def _make_dispatcher(
    cfg: AppConfig, client: httpx.AsyncClient, writer: LogWriter
) -> tuple[Dispatcher, BackendRegistry]:
    registry = BackendRegistry(cfg.backends)
    groups = get_model_groups(cfg, registry.get_canonical_models())
    dispatcher = Dispatcher(
        registry, client, groups,
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
            ["qwen3.5"], BODY, False, future,
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
            ["qwen3.5"], BODY, False, future,
            request_id="req-err", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        dispatcher.enqueue(pending)
        with pytest.raises(ForwardError):
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
        future.cancel()
        pending = PendingRequest(
            ["qwen3.5"], BODY, True, future,
            request_id="stream-req", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        _ = pending

    assert len(writer.records) == 0
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_dispatcher.py tests/test_dispatcher_logging.py -v
```

Expected: failures — `PendingRequest` first positional arg is currently `str` not `list[str]`; `Dispatcher.__init__` currently takes `canonical_models: list[str]`; `get_model_groups` import fails.

- [ ] **Step 3: Implement dispatcher changes**

Replace the full contents of `aggregate_server/dispatcher.py`:

```python
# aggregate_server/dispatcher.py
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from aggregate_server.forwarder import ForwardError, ForwardResult, forward_request, tracked_stream
from aggregate_server.registry import BackendEntry, BackendRegistry

if TYPE_CHECKING:
    from aggregate_server.log_writer import LogWriter

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    def __init__(self, queue_key: str, max_size: int) -> None:
        super().__init__(f"Queue full for model '{queue_key}' (capacity {max_size})")
        self.status_code = 503


@dataclass
class PendingRequest:
    canonical_models: list[str]
    body: dict[str, Any]
    stream: bool
    result_future: asyncio.Future[ForwardResult]
    request_id: str = ""
    timestamp: float = 0.0
    inbound_model: str = ""
    enqueue_at: float = field(default_factory=time.monotonic)


class Dispatcher:
    """
    Manages per-model-group request queues and dispatches requests to free backends.
    One run_for_model() coroutine must be running per queue key.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        client: httpx.AsyncClient,
        canonical_model_groups: list[list[str]],
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
            self._key(g): asyncio.Queue() for g in canonical_model_groups
        }

    @staticmethod
    def _key(canonical_models: list[str]) -> str:
        return ",".join(sorted(canonical_models))

    @property
    def queue_keys(self) -> list[str]:
        return list(self._queues.keys())

    def enqueue(self, pending: PendingRequest) -> None:
        key = self._key(pending.canonical_models)
        queue = self._queues.get(key)
        if queue is None:
            pending.result_future.set_exception(
                ForwardError(
                    f"No backends configured for model '{key}'", 404
                )
            )
            return
        if queue.qsize() >= self._max_queue_size:
            pending.result_future.set_exception(
                QueueFullError(key, self._max_queue_size)
            )
            return
        queue.put_nowait(pending)

    async def run_for_model(self, queue_key: str) -> None:
        """Long-running loop: dequeue requests and dispatch them to free backends."""
        queue = self._queues[queue_key]
        while True:
            pending = await queue.get()
            entry = await self._acquire_with_poll(pending.canonical_models)
            asyncio.create_task(self._handle_request(entry, pending))

    async def _acquire_with_poll(self, canonical_models: list[str]) -> BackendEntry:
        while True:
            entry = await self._registry.acquire_backend(canonical_models)
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
                current = await self._next_untried_backend(pending.canonical_models, tried_ids)

        label = ",".join(pending.canonical_models)
        err = ForwardError(f"All backends for model group '{label}' exhausted", 502)
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
            with contextlib.suppress(Exception):
                usage = result.response.json().get("usage", {})
        self._log_writer.enqueue(LogRecord(  # type: ignore[union-attr]
            request_id=pending.request_id,
            timestamp=pending.timestamp,
            inbound_model=pending.inbound_model,
            canonical_model=entry.config.model,
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
            canonical_model=pending.canonical_models[0],
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
        self, canonical_models: list[str], tried_ids: set[str]
    ) -> BackendEntry | None:
        entry = await self._registry.acquire_backend(canonical_models)
        if entry is None:
            return None
        if entry.config.id in tried_ids:
            await self._registry.release_backend(entry)
            return None
        return entry
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_dispatcher.py tests/test_dispatcher_logging.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add aggregate_server/dispatcher.py tests/test_dispatcher.py tests/test_dispatcher_logging.py
git commit -m "feat: dispatcher queues by model group; PendingRequest.canonical_models"
```

---

## Task 4: Router — wire everything together, fix `/v1/models`

**Files:**
- Modify: `aggregate_server/router.py`
- Modify: `tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Replace the full contents of `tests/test_router.py`:

```python
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from aggregate_server.router import app

RESPONSE_JSON = {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
CHAT_BODY = {"model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def config_path(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
backends:
  - id: b1
    url: http://backend1:8080
    api_key: key1
    model: qwen3.5
model_aliases:
  qwen-chat: qwen3.5
queue_timeout: 5
backend_timeout: 10
max_queue_size: 10
""")
    return str(cfg)


@respx.mock
def test_non_streaming_roundtrip(config_path: str) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=CHAT_BODY)

    assert resp.status_code == 200
    assert "choices" in resp.json()


@respx.mock
def test_model_not_found_returns_404(config_path: str) -> None:
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                               json={"model": "no-such-model", "messages": []})

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "model_not_found"
    assert "available_models" in body["error"]


@respx.mock
def test_alias_resolved_before_routing() -> None:
    route = respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            model_aliases={"qwen-chat": "qwen3.5"},
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                               json={"model": "qwen-chat", "messages": []})

    assert resp.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen3.5"


def test_list_models_returns_aliases_not_canonicals() -> None:
    """
    /v1/models should return alias keys + un-aliased canonicals.
    Canonical models that are alias targets must not appear directly.
    """
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[
                BackendConfig(id="b1", url="http://b1:8080", api_key="k", model="qwen3.5-9b"),
                BackendConfig(id="b2", url="http://b2:8080", api_key="k", model="qwen3.5-9b-q8"),
                BackendConfig(id="b3", url="http://b3:8080", api_key="k", model="llama3"),
            ],
            model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "qwen3.5" in ids        # alias
    assert "llama3" in ids          # un-aliased canonical
    assert "qwen3.5-9b" not in ids  # aliased canonical — must be hidden
    assert "qwen3.5-9b-q8" not in ids


@respx.mock
def test_multi_model_alias_routes_to_any_backend() -> None:
    """A request using a multi-model alias reaches one of the alias's target backends."""
    route_9b = respx.post("http://b1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    route_q8 = respx.post("http://b2:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[
                BackendConfig(id="b1", url="http://b1:8080", api_key="k", model="qwen3.5-9b"),
                BackendConfig(id="b2", url="http://b2:8080", api_key="k", model="qwen3.5-9b-q8"),
            ],
            model_aliases={"qwen3.5": ["qwen3.5-9b", "qwen3.5-9b-q8"]},
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                               json={"model": "qwen3.5", "messages": []})

    assert resp.status_code == 200
    assert route_9b.called or route_q8.called
```

- [ ] **Step 2: Run tests to verify failures**

```
pytest tests/test_router.py -v
```

Expected: failures — `resolve_model` now returns `list`, `has_backends_for_model` no longer exists, `list_models` still returns canonicals not aliases, `_make_dispatcher` signature mismatch.

- [ ] **Step 3: Implement router changes**

Replace the full contents of `aggregate_server/router.py`:

```python
# aggregate_server/router.py
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aggregate_server.config import (
    AppConfig, get_callable_models, get_model_groups, load_config, resolve_model,
)
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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    registry = BackendRegistry(cfg.backends)
    client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.backend_timeout, connect=10.0))
    log_writer = LogWriter()

    logger.info("Probing all backends before accepting traffic...")
    await check_all(registry, client, failed_only=False)

    canonical_models = registry.get_canonical_models()
    groups = get_model_groups(cfg, canonical_models)
    dispatcher = Dispatcher(
        registry, client, groups,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
        log_writer=log_writer,
    )

    dispatch_tasks = [
        asyncio.create_task(dispatcher.run_for_model(k)) for k in dispatcher.queue_keys
    ]
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
    cfg: AppConfig = request.app.state.config
    registry: BackendRegistry = request.app.state.registry
    callable_names = get_callable_models(cfg, registry.get_canonical_models())
    models = [ModelObject(id=m) for m in callable_names]
    return JSONResponse(ModelsResponse(data=models).model_dump())


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
    canonical_models = resolve_model(cfg, inbound_model)

    if not registry.has_backends_for_models(canonical_models):
        available = get_callable_models(cfg, registry.get_canonical_models())
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical_models[0],
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
        canonical_models=canonical_models,
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
                log_writer, request_id, timestamp, inbound_model, canonical_models[0],
                enqueue_at, 504, "request timed out waiting for a free backend",
            )
        return _error_response(
            "Request timed out waiting for a free backend", "server_error", 504
        )
    except QueueFullError as exc:
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical_models[0],
                enqueue_at, 503, str(exc),
            )
        return _error_response(str(exc), "rate_limit_exceeded", 503)
    except ForwardError as exc:
        return _error_response(str(exc), "server_error", exc.status_code)

    if result.is_stream:
        if result.stream_gen is None:
            raise RuntimeError("Stream ForwardResult has no generator")
        return StreamingResponse(result.stream_gen, media_type="text/event-stream")
    if result.response is None:
        raise RuntimeError("Non-stream ForwardResult has no response body")
    return JSONResponse(result.response.json(), status_code=result.response.status_code)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_router.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add aggregate_server/router.py tests/test_router.py
git commit -m "feat: router uses multi-model resolve; /v1/models lists aliases"
```

---

## Task 5: Full suite, lint, and type check

**Files:** all

- [ ] **Step 1: Run the full test suite**

```
pytest -v
```

Expected: all tests pass. Fix any failures before proceeding.

- [ ] **Step 2: Run ruff**

```
ruff check aggregate_server/ tests/ dashboard/
```

Expected: no errors. Fix any issues, then re-run to confirm.

- [ ] **Step 3: Run mypy**

```
mypy aggregate_server/ dashboard/
```

Expected: no errors. Fix any issues, then re-run to confirm.

- [ ] **Step 4: Update config.yaml with list-form alias example**

Update the `model_aliases` section of `config.yaml` to show both forms:

```yaml
backends:
  - id: backend_1
    url: http://192.168.1.10:8080
    api_key: sk-placeholder-1
    model: qwen3.5-9b
  - id: backend_2
    url: http://192.168.1.11:8080
    api_key: sk-placeholder-2
    model: qwen3.5-9b-q8
  - id: backend_3
    url: http://192.168.1.12:8080
    api_key: sk-placeholder-3
    model: llama3

model_aliases:
  qwen3.5: [qwen3.5-9b, qwen3.5-9b-q8]   # multi-model alias
  llama3-instruct: llama3                   # single-model alias (string form still works)

queue_timeout: 60       # seconds a request may wait in queue before 504
backend_timeout: 300    # seconds to wait for a backend response before 502
max_queue_size: 100     # per-model queue depth; 503 returned when full
```

- [ ] **Step 5: Commit**

```
git add .
git commit -m "chore: update config.yaml example for multi-model aliases; clean lint/mypy"
```
