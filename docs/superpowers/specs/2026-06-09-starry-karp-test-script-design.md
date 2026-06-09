# Integration Test Script — "Starry Karp"

**Date:** 2026-06-09  
**Status:** Approved  

---

## Context

The aggregate server has a queue + registry system that needs end-to-end validation beyond unit tests. The existing pytest suite mocks HTTP calls via `respx` — it never exercises real network I/O, real asyncio scheduling under load, or real backend latency effects. This script fills that gap by spinning up 4 fake OpenAI-compatible backends and the aggregate server in-process, then verifying that load balancing and per-model queue isolation behave correctly under realistic conditions.

---

## Approach

**Fully in-process asyncio** — all components run as `uvicorn.Server` asyncio tasks inside a single `asyncio.run(main())`. No subprocesses, no temp Python files. One temp `config.yaml` is written per phase. Fake backends stay running for both phases; only the aggregate server is restarted between phases.

---

## Components

### 1. FakeBackend (FastAPI app, one instance per backend)

Two endpoints:

- `POST /v1/chat/completions` — sleeps for fixed latency, appends request record, returns minimal OpenAI-compatible JSON response
- `GET /stats` — returns `{backend_id, hit_count, requests: [{request_id, model, received_at, responded_at}]}`
- `POST /reset` — clears the in-memory request log (called between phases)

Fixed latency assignments:

| Backend    | Port | Latency | Phase 1 model | Phase 2 model |
|------------|------|---------|---------------|---------------|
| backend_1  | 9001 | 0.5 s   | `test-model`  | `model-a`     |
| backend_2  | 9002 | 1.0 s   | `test-model`  | `model-a`     |
| backend_3  | 9003 | 2.0 s   | `test-model`  | `model-b`     |
| backend_4  | 9004 | 3.0 s   | `test-model`  | `model-b`     |

The response body is a minimal `ChatCompletion` JSON — enough to satisfy the aggregate server's response parsing and token counting. Must include `usage.prompt_tokens` and `usage.completion_tokens` (the forwarder reads these for `LogRecord`).

### 2. ServerHarness

- Writes a temp `config.yaml` to a `tempfile.NamedTemporaryFile`
- Starts the aggregate server via `uvicorn.Server` on port 8765
- Sets `CONFIG_PATH` env var before startup
- Exposes `start()` / `stop()` async methods
- Polls `GET /v1/models` until 200 (ready check, max 10s)

### 3. TestRunner

Sends requests in **4 waves of 5**, with 0.5s between waves (20 total per phase). Each wave fires all 5 requests concurrently via `asyncio.gather`. Uses `httpx.AsyncClient` targeting `http://localhost:8765`.

Request body:
```json
{
  "model": "<phase-dependent>",
  "messages": [{"role": "user", "content": "ping"}],
  "stream": false
}
```

### 4. Verifier

Queries `GET /stats` on each fake backend after each phase and runs assertions (see Verification section).

---

## Startup Sequence

```
1. Start FakeBackend tasks for all 4 backends
2. Wait until all 4 ports accept TCP connections (poll asyncio.open_connection, max 5s)
3. Write Phase 1 config.yaml (all 4 backends → test-model)
4. Start aggregate server (ServerHarness), wait for /v1/models → 200
5. Run Phase 1 (TestRunner + Verifier)
6. POST /reset to all 4 fake backends
7. Stop aggregate server
8. Write Phase 2 config.yaml (backends 1+2 → model-a, 3+4 → model-b)
9. Restart aggregate server, wait for /v1/models → 200
10. Run Phase 2 (TestRunner + Verifier)
11. Stop aggregate server
12. Stop all fake backend tasks
13. Print summary report
```

---

## Phase 1 — Load Balancing

**Config:** All 4 backends serve `test-model`.  
**Requests:** 20 × `test-model`, sent in 4 waves of 5.

**Assertions:**
- All 20 responses return HTTP 200
- All 4 backends were hit at least once
- `backend_1` hit count > `backend_4` hit count (faster backend accumulates more hits)
- Total elapsed time < 20 × 3.0s (proves concurrency, not serialisation)

---

## Phase 2 — Queue Isolation

**Config:** backend_1 + backend_2 serve `model-a`; backend_3 + backend_4 serve `model-b`.  
**Requests:** 20 requests interleaved — odd waves target `model-a`, even waves target `model-b` (10 each).

**Assertions:**
- All 20 responses return HTTP 200
- `model-a` requests only hit backend_1 or backend_2 (never backend_3 or backend_4)
- `model-b` requests only hit backend_3 or backend_4 (never backend_1 or backend_2)
- backend_1 + backend_2 combined hit count = 10
- backend_3 + backend_4 combined hit count = 10

---

## Output

The script prints a structured summary report on completion:

```
====== Starry Karp Integration Test ======

Phase 1: Load Balancing (test-model, 4 backends)
  ✓ All 20 requests returned 200
  ✓ All 4 backends received traffic
  ✓ backend_1 (8 hits) > backend_4 (2 hits)
  ✓ Elapsed: 6.2s (< 60s serialised ceiling)
  Hit distribution: backend_1=8 backend_2=5 backend_3=4 backend_4=3

Phase 2: Queue Isolation (model-a + model-b)
  ✓ All 20 requests returned 200
  ✓ model-a traffic stayed on backend_1, backend_2 (10 hits)
  ✓ model-b traffic stayed on backend_3, backend_4 (10 hits)

All assertions passed. Fleet is battle-ready.
```

---

## File Location

`scripts/starry_karp.py` — standalone script, no imports from the test suite.

---

## Verification

Run with:
```bash
python scripts/starry_karp.py
```

Expected: all assertions pass, summary report printed, script exits 0.  
On assertion failure: script prints the failing check and exits 1.

Dependencies already in `pyproject.toml`: `fastapi`, `uvicorn`, `httpx`, `pyyaml`.  
No new dependencies required.
