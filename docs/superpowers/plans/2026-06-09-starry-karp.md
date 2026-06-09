# Starry Karp Integration Test Script — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/starry_karp.py` — a self-contained integration test that spins up 4 fake OpenAI-compatible backends and the aggregate server in-process, then verifies load balancing and per-model queue isolation across two test phases.

**Architecture:** All components run as `uvicorn.Server` asyncio tasks inside a single `asyncio.run(main())`. Fake backends expose `/v1/chat/completions` (with fixed latency), `/stats`, and `/reset`. The aggregate server is started twice — once per phase — with a generated temp `config.yaml`. Phase 1: all 4 backends serve `test-model`. Phase 2: backends 1+2 serve `model-a`, 3+4 serve `model-b`. Requests are sent in 4 waves of 5 with 0.5 s between waves.

**Tech Stack:** FastAPI, uvicorn, httpx, pyyaml, pytest (all already in `pyproject.toml`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/starry_karp.py` | Create | Complete standalone integration test script |
| `tests/test_starry_karp_verifier.py` | Create | Unit tests for pure verifier logic |

---

## Task 1: Scaffold `scripts/starry_karp.py` with FakeBackend factory

**Files:**
- Create: `scripts/starry_karp.py`

- [ ] **Step 1: Create the scripts directory and stub file**

```bash
mkdir scripts
```

Then create `scripts/starry_karp.py`:

```python
#!/usr/bin/env python3
"""
Starry Karp — end-to-end integration test for aggregate_server.

Runs 4 fake OpenAI-compatible backends + the aggregate server in-process,
then verifies load balancing (Phase 1) and per-model queue isolation (Phase 2).

Usage:
    python scripts/starry_karp.py
Exit codes:
    0 — all assertions passed
    1 — one or more assertions failed
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────

FAKE_BACKENDS: list[dict] = [
    {"id": "backend_1", "port": 9001, "latency": 0.5},
    {"id": "backend_2", "port": 9002, "latency": 1.0},
    {"id": "backend_3", "port": 9003, "latency": 2.0},
    {"id": "backend_4", "port": 9004, "latency": 3.0},
]
AGG_PORT = 8765

# ── FakeBackend ───────────────────────────────────────────────────────────────


@dataclass
class RequestRecord:
    model: str
    received_at: float
    responded_at: float


def make_fake_backend(backend_id: str, latency: float) -> FastAPI:
    """Return a FastAPI app that simulates an OpenAI-compatible backend."""
    app: FastAPI = FastAPI()
    requests_log: list[RequestRecord] = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        received = time.monotonic()
        await asyncio.sleep(latency)
        responded = time.monotonic()
        requests_log.append(
            RequestRecord(
                model=body.get("model", ""),
                received_at=received,
                responded_at=responded,
            )
        )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        )

    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(
            {
                "backend_id": backend_id,
                "hit_count": len(requests_log),
                "requests": [
                    {
                        "model": r.model,
                        "received_at": r.received_at,
                        "responded_at": r.responded_at,
                    }
                    for r in requests_log
                ],
            }
        )

    @app.post("/reset")
    async def reset() -> JSONResponse:
        requests_log.clear()
        return JSONResponse({"status": "ok"})

    return app
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from scripts.starry_karp import make_fake_backend, FAKE_BACKENDS; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/starry_karp.py
git commit -m "feat(starry-karp): scaffold script with FakeBackend factory"
```

---

## Task 2: Server lifecycle helpers and config writers

**Files:**
- Modify: `scripts/starry_karp.py` — append after the FakeBackend section

- [ ] **Step 1: Append lifecycle helpers and config writers to `scripts/starry_karp.py`**

```python
# ── Server lifecycle ──────────────────────────────────────────────────────────


async def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Poll until a TCP port accepts connections or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"Port {port} not ready after {timeout:.0f}s")


async def start_server(
    app: FastAPI, port: int
) -> tuple[uvicorn.Server, asyncio.Task]:
    """Start a uvicorn server on the given port and wait until it accepts connections."""
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", loop="none"
    )
    server = uvicorn.Server(config)
    task: asyncio.Task = asyncio.create_task(server.serve())
    await wait_for_port("127.0.0.1", port)
    return server, task


async def stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    """Signal the server to exit and wait for the task to finish."""
    server.should_exit = True
    await task


# ── Config writers ────────────────────────────────────────────────────────────


def _backend_entry(backend_id: str, port: int, model: str) -> dict:
    return {
        "id": backend_id,
        "url": f"http://127.0.0.1:{port}",
        "api_key": "test-key",
        "model": model,
    }


def write_phase1_config(path: str) -> None:
    """All 4 backends serve 'test-model'."""
    cfg = {
        "backends": [
            _backend_entry(b["id"], b["port"], "test-model")
            for b in FAKE_BACKENDS
        ],
        "queue_timeout": 30,
        "backend_timeout": 60,
        "max_queue_size": 50,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def write_phase2_config(path: str) -> None:
    """Backends 1+2 serve 'model-a', backends 3+4 serve 'model-b'."""
    model_map = {
        "backend_1": "model-a",
        "backend_2": "model-a",
        "backend_3": "model-b",
        "backend_4": "model-b",
    }
    cfg = {
        "backends": [
            _backend_entry(b["id"], b["port"], model_map[b["id"]])
            for b in FAKE_BACKENDS
        ],
        "queue_timeout": 30,
        "backend_timeout": 60,
        "max_queue_size": 50,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


async def wait_for_agg_server(
    client: httpx.AsyncClient, timeout: float = 20.0
) -> None:
    """Poll /v1/models until the aggregate server returns 200."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = await client.get(
                f"http://127.0.0.1:{AGG_PORT}/v1/models", timeout=2.0
            )
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError("Aggregate server did not become ready in time")
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import scripts.starry_karp; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/starry_karp.py
git commit -m "feat(starry-karp): add server lifecycle helpers and config writers"
```

---

## Task 3: TestRunner

**Files:**
- Modify: `scripts/starry_karp.py` — append after config writers

- [ ] **Step 1: Append TestRunner to `scripts/starry_karp.py`**

```python
# ── TestRunner ────────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    status_code: int
    body: dict


async def _send_one(client: httpx.AsyncClient, model: str) -> RequestResult:
    try:
        resp = await client.post(
            f"http://127.0.0.1:{AGG_PORT}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "ping"}]},
            timeout=60.0,
        )
        return RequestResult(status_code=resp.status_code, body=resp.json())
    except Exception as exc:
        return RequestResult(status_code=0, body={"error": str(exc)})


async def send_wave(
    client: httpx.AsyncClient, model: str, count: int
) -> list[RequestResult]:
    """Fire `count` requests concurrently, all for the same model."""
    return list(await asyncio.gather(*[_send_one(client, model) for _ in range(count)]))


async def run_phase1(client: httpx.AsyncClient) -> list[RequestResult]:
    """4 waves of 5 requests, all targeting 'test-model'."""
    results: list[RequestResult] = []
    for _ in range(4):
        results.extend(await send_wave(client, "test-model", 5))
        await asyncio.sleep(0.5)
    return results


async def run_phase2(client: httpx.AsyncClient) -> list[RequestResult]:
    """4 waves of 5 requests, alternating between 'model-a' and 'model-b'."""
    results: list[RequestResult] = []
    for i in range(4):
        model = "model-a" if i % 2 == 0 else "model-b"
        results.extend(await send_wave(client, model, 5))
        await asyncio.sleep(0.5)
    return results
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import scripts.starry_karp; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/starry_karp.py
git commit -m "feat(starry-karp): add TestRunner (send_wave, run_phase1, run_phase2)"
```

---

## Task 4: Verifier (TDD — write tests first)

**Files:**
- Create: `tests/test_starry_karp_verifier.py`
- Modify: `scripts/starry_karp.py` — append Verifier section

- [ ] **Step 1: Write the failing tests**

Create `tests/test_starry_karp_verifier.py`:

```python
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from scripts.starry_karp import AssertionResult, RequestResult, verify_phase1, verify_phase2

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_results(n: int, status: int = 200) -> list[RequestResult]:
    return [RequestResult(status_code=status, body={}) for _ in range(n)]


def make_stats(
    backend_id: str,
    hit_count: int,
    models: list[str],
) -> dict:
    return {
        "backend_id": backend_id,
        "hit_count": hit_count,
        "requests": [{"model": m} for m in models],
    }


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def test_phase1_all_pass() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 10, ["test-model"] * 10),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 3, ["test-model"] * 3),
        "backend_4": make_stats("backend_4", 2, ["test-model"] * 2),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    failures = [c for c in checks if not c.passed]
    assert failures == [], [c.message for c in failures]


def test_phase1_fails_when_not_200() -> None:
    results = make_results(19, 200) + [RequestResult(status_code=404, body={})]
    stats = {
        "backend_1": make_stats("backend_1", 10, ["test-model"] * 10),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 3, ["test-model"] * 3),
        "backend_4": make_stats("backend_4", 2, ["test-model"] * 2),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "200" in c.message for c in checks)


def test_phase1_fails_when_backend_not_hit() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 15, ["test-model"] * 15),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 0, []),
        "backend_4": make_stats("backend_4", 0, []),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "all" in c.message.lower() for c in checks)


def test_phase1_fails_when_b1_not_faster_than_b4() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 2, ["test-model"] * 2),
        "backend_2": make_stats("backend_2", 6, ["test-model"] * 6),
        "backend_3": make_stats("backend_3", 6, ["test-model"] * 6),
        "backend_4": make_stats("backend_4", 6, ["test-model"] * 6),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "backend_1" in c.message for c in checks)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def test_phase2_all_pass() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 6, ["model-a"] * 6),
        "backend_2": make_stats("backend_2", 4, ["model-a"] * 4),
        "backend_3": make_stats("backend_3", 7, ["model-b"] * 7),
        "backend_4": make_stats("backend_4", 3, ["model-b"] * 3),
    }
    checks = verify_phase2(results, stats)
    failures = [c for c in checks if not c.passed]
    assert failures == [], [c.message for c in failures]


def test_phase2_fails_when_wrong_model_on_backend() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 5, ["model-a"] * 5),
        "backend_2": make_stats("backend_2", 5, ["model-a"] * 5),
        "backend_3": make_stats("backend_3", 5, ["model-a"] * 3 + ["model-b"] * 2),
        "backend_4": make_stats("backend_4", 5, ["model-b"] * 5),
    }
    checks = verify_phase2(results, stats)
    assert any(not c.passed for c in checks)


def test_phase2_fails_when_hit_counts_wrong() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 8, ["model-a"] * 8),
        "backend_2": make_stats("backend_2", 8, ["model-a"] * 8),
        "backend_3": make_stats("backend_3", 2, ["model-b"] * 2),
        "backend_4": make_stats("backend_4", 2, ["model-b"] * 2),
    }
    checks = verify_phase2(results, stats)
    assert any(not c.passed for c in checks)
```

- [ ] **Step 2: Run tests — expect ImportError (verify_phase1/verify_phase2 not yet defined)**

```bash
pytest tests/test_starry_karp_verifier.py -v
```

Expected: `ImportError` or `FAILED` because `verify_phase1` and `verify_phase2` don't exist yet.

- [ ] **Step 3: Implement Verifier in `scripts/starry_karp.py`**

Append to the script:

```python
# ── Verifier ──────────────────────────────────────────────────────────────────


@dataclass
class AssertionResult:
    passed: bool
    message: str


def verify_phase1(
    results: list[RequestResult],
    stats: dict[str, dict],
    elapsed: float,
) -> list[AssertionResult]:
    """Run Phase 1 assertions: all 200s, all backends hit, faster > slower."""
    checks: list[AssertionResult] = []
    failed = [r.status_code for r in results if r.status_code != 200]
    checks.append(AssertionResult(
        not failed,
        f"All 20 requests returned 200 (non-200 codes: {failed})",
    ))
    all_hit = all(stats[b["id"]]["hit_count"] > 0 for b in FAKE_BACKENDS)
    hit_counts = {b["id"]: stats[b["id"]]["hit_count"] for b in FAKE_BACKENDS}
    checks.append(AssertionResult(
        all_hit,
        f"All 4 backends received traffic (counts: {hit_counts})",
    ))
    b1 = stats["backend_1"]["hit_count"]
    b4 = stats["backend_4"]["hit_count"]
    checks.append(AssertionResult(
        b1 > b4,
        f"backend_1 ({b1} hits) > backend_4 ({b4} hits) — faster backend serves more",
    ))
    total = sum(stats[b["id"]]["hit_count"] for b in FAKE_BACKENDS)
    checks.append(AssertionResult(
        total == 20,
        f"Total hits across all backends = 20 (got {total})",
    ))
    checks.append(AssertionResult(
        elapsed < 60.0,
        f"Elapsed {elapsed:.1f}s < 60s serialised ceiling (proves concurrency)",
    ))
    return checks


def verify_phase2(
    results: list[RequestResult],
    stats: dict[str, dict],
) -> list[AssertionResult]:
    """Run Phase 2 assertions: all 200s, per-model queue isolation holds."""
    checks: list[AssertionResult] = []
    failed = [r.status_code for r in results if r.status_code != 200]
    checks.append(AssertionResult(
        not failed,
        f"All 20 requests returned 200 (non-200 codes: {failed})",
    ))
    a_hits = stats["backend_1"]["hit_count"] + stats["backend_2"]["hit_count"]
    b_hits = stats["backend_3"]["hit_count"] + stats["backend_4"]["hit_count"]
    checks.append(AssertionResult(
        a_hits == 10,
        f"model-a backends (1+2) combined = 10 hits (got {a_hits})",
    ))
    checks.append(AssertionResult(
        b_hits == 10,
        f"model-b backends (3+4) combined = 10 hits (got {b_hits})",
    ))
    b3_models = {r["model"] for r in stats["backend_3"]["requests"]}
    b4_models = {r["model"] for r in stats["backend_4"]["requests"]}
    b12_models = {
        r["model"]
        for r in stats["backend_1"]["requests"] + stats["backend_2"]["requests"]
    }
    checks.append(AssertionResult(
        b3_models <= {"model-b"} and b4_models <= {"model-b"},
        f"model-a traffic never reached backend_3/4 (b3: {b3_models}, b4: {b4_models})",
    ))
    checks.append(AssertionResult(
        b12_models <= {"model-a"},
        f"model-b traffic never reached backend_1/2 (b1+b2 models: {b12_models})",
    ))
    return checks
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_starry_karp_verifier.py -v
```

Expected output:
```
tests/test_starry_karp_verifier.py::test_phase1_all_pass PASSED
tests/test_starry_karp_verifier.py::test_phase1_fails_when_not_200 PASSED
tests/test_starry_karp_verifier.py::test_phase1_fails_when_backend_not_hit PASSED
tests/test_starry_karp_verifier.py::test_phase1_fails_when_b1_not_faster_than_b4 PASSED
tests/test_starry_karp_verifier.py::test_phase2_all_pass PASSED
tests/test_starry_karp_verifier.py::test_phase2_fails_when_wrong_model_on_backend PASSED
tests/test_starry_karp_verifier.py::test_phase2_fails_when_hit_counts_wrong PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/starry_karp.py tests/test_starry_karp_verifier.py
git commit -m "feat(starry-karp): add Verifier with unit tests (TDD)"
```

---

## Task 5: Stats helpers, report printer, and `main()`

**Files:**
- Modify: `scripts/starry_karp.py` — append final sections

- [ ] **Step 1: Append stats helpers, report printer, and `main()` to `scripts/starry_karp.py`**

```python
# ── Stats helpers ─────────────────────────────────────────────────────────────


async def get_stats(client: httpx.AsyncClient, port: int) -> dict:
    resp = await client.get(f"http://127.0.0.1:{port}/stats", timeout=5.0)
    return dict(resp.json())


async def reset_stats(client: httpx.AsyncClient, port: int) -> None:
    await client.post(f"http://127.0.0.1:{port}/reset", timeout=5.0)


async def collect_stats(client: httpx.AsyncClient) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for b in FAKE_BACKENDS:
        result[b["id"]] = await get_stats(client, b["port"])
    return result


# ── Report ────────────────────────────────────────────────────────────────────


def _print_checks(checks: list[AssertionResult]) -> bool:
    all_pass = True
    for check in checks:
        icon = "✓" if check.passed else "✗"
        print(f"  {icon} {check.message}")
        if not check.passed:
            all_pass = False
    return all_pass


def print_report(
    phase1_checks: list[AssertionResult],
    phase2_checks: list[AssertionResult],
    phase1_stats: dict[str, dict],
) -> bool:
    print("\n====== Starry Karp Integration Test ======\n")
    print("Phase 1: Load Balancing (test-model, 4 backends)")
    p1_pass = _print_checks(phase1_checks)
    if p1_pass:
        dist = " ".join(
            f"{b['id']}={phase1_stats[b['id']]['hit_count']}" for b in FAKE_BACKENDS
        )
        print(f"  Hit distribution: {dist}")
    print()
    print("Phase 2: Queue Isolation (model-a ↔ backend_1/2, model-b ↔ backend_3/4)")
    p2_pass = _print_checks(phase2_checks)
    print()
    if p1_pass and p2_pass:
        print("All assertions passed. Fleet is battle-ready.")
    else:
        print("ASSERTIONS FAILED. Review the output above.")
    return p1_pass and p2_pass


# ── Main ──────────────────────────────────────────────────────────────────────


async def _start_fake_backends() -> tuple[list[FastAPI], list[tuple[uvicorn.Server, asyncio.Task]]]:
    apps = [make_fake_backend(b["id"], b["latency"]) for b in FAKE_BACKENDS]
    servers = []
    for app, b in zip(apps, FAKE_BACKENDS):
        server, task = await start_server(app, b["port"])
        servers.append((server, task))
    return apps, servers


async def _run_phase(
    client: httpx.AsyncClient,
    config_path: str,
    phase: int,
) -> tuple[list[RequestResult], dict[str, dict], float]:
    """Configure the agg server for the given phase, run requests, collect stats."""
    if phase == 1:
        write_phase1_config(config_path)
    else:
        write_phase2_config(config_path)
    os.environ["CONFIG_PATH"] = config_path

    from aggregate_server.router import app as agg_app

    print(f"Starting aggregate server (phase {phase})...")
    agg_server, agg_task = await start_server(agg_app, AGG_PORT)
    await wait_for_agg_server(client)
    print(f"  Ready. Running phase {phase} requests...")

    t0 = time.monotonic()
    results = await run_phase1(client) if phase == 1 else await run_phase2(client)
    elapsed = time.monotonic() - t0

    stats = await collect_stats(client)
    await stop_server(agg_server, agg_task)
    return results, stats, elapsed


async def main() -> int:
    print("Starting fake backends...")
    _, backend_servers = await _start_fake_backends()
    print("  All 4 backends ready.\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name

    phase1_results: list[RequestResult] = []
    phase2_results: list[RequestResult] = []
    phase1_stats: dict[str, dict] = {}
    phase2_stats: dict[str, dict] = {}
    elapsed = 0.0

    async with httpx.AsyncClient() as client:
        try:
            phase1_results, phase1_stats, elapsed = await _run_phase(
                client, config_path, phase=1
            )
            for b in FAKE_BACKENDS:
                await reset_stats(client, b["port"])

            phase2_results, phase2_stats, _ = await _run_phase(
                client, config_path, phase=2
            )
        finally:
            print("\nStopping fake backends...")
            for server, task in backend_servers:
                await stop_server(server, task)
            try:
                os.unlink(config_path)
            except OSError:
                pass

    p1_checks = verify_phase1(phase1_results, phase1_stats, elapsed)
    p2_checks = verify_phase2(phase2_results, phase2_stats)
    success = print_report(p1_checks, p2_checks, phase1_stats)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Verify full syntax**

```bash
python -c "import scripts.starry_karp; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Run existing unit tests to confirm no regressions**

```bash
pytest tests/test_starry_karp_verifier.py tests/ -v --ignore=tests/test_starry_karp_verifier.py -x -q
```

Expected: all existing 46 tests + 7 new verifier tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/starry_karp.py
git commit -m "feat(starry-karp): add stats helpers, report printer, and main orchestration"
```

---

## Task 6: End-to-end run and verification

- [ ] **Step 1: Run the integration test script**

```bash
python scripts/starry_karp.py
```

Expected output (approximate):
```
Starting fake backends...
  All 4 backends ready.

Starting aggregate server (phase 1)...
  Ready. Running phase 1 requests...
Starting aggregate server (phase 2)...
  Ready. Running phase 2 requests...

Stopping fake backends...

====== Starry Karp Integration Test ======

Phase 1: Load Balancing (test-model, 4 backends)
  ✓ All 20 requests returned 200 (non-200 codes: [])
  ✓ All 4 backends received traffic (counts: {...})
  ✓ backend_1 (N hits) > backend_4 (M hits) — faster backend serves more
  ✓ Total hits across all backends = 20 (got 20)
  ✓ Elapsed X.Xs < 60s serialised ceiling (proves concurrency)
  Hit distribution: backend_1=N backend_2=N backend_3=N backend_4=N

Phase 2: Queue Isolation (model-a ↔ backend_1/2, model-b ↔ backend_3/4)
  ✓ All 20 requests returned 200 (non-200 codes: [])
  ✓ model-a backends (1+2) combined = 10 hits (got 10)
  ✓ model-b backends (3+4) combined = 10 hits (got 10)
  ✓ model-a traffic never reached backend_3/4 (b3: {'model-b'}, b4: {'model-b'})
  ✓ model-b traffic never reached backend_1/2 (b1+b2 models: {'model-a'})

All assertions passed. Fleet is battle-ready.
```

Exit code should be 0: `echo $?` (bash) or `echo $LASTEXITCODE` (PowerShell).

- [ ] **Step 2: If any Phase 1 requests return 404 — diagnose the list/str mismatch**

The aggregate server's `resolve_model` returns `list[str]`, but `registry.has_backends_for_model` expects `str`. If this causes 404s, add a one-line fix in `aggregate_server/router.py`:

Find line:
```python
canonical = resolve_model(cfg, inbound_model)
```

The downstream code (`has_backends_for_model`, `PendingRequest.canonical_model`, `dispatcher._queues`) all expect a single `str`. Apply:
```python
canonical_list = resolve_model(cfg, inbound_model)
canonical = canonical_list[0] if canonical_list else inbound_model
```

Then re-run `pytest tests/ -v` to confirm all tests still pass, then re-run the script.

- [ ] **Step 3: Commit final state**

```bash
git add scripts/starry_karp.py tests/test_starry_karp_verifier.py
git commit -m "feat(starry-karp): complete integration test script — load balancing + queue isolation"
```

---

## Verification Summary

| Check | Command |
|-------|---------|
| Unit tests pass | `pytest tests/test_starry_karp_verifier.py -v` |
| Full test suite clean | `pytest tests/ -v` |
| Script syntax | `python -c "import scripts.starry_karp"` |
| End-to-end | `python scripts/starry_karp.py` — exit 0, all ✓ |
| Lint | `ruff check scripts/starry_karp.py` |
| Types | `mypy scripts/starry_karp.py` |
