# PROJECT_CONTEXT.md

> Last updated: 2026-06-09 (rev 4) | [README](README.md)

---

## Project Description

`aggregate_server` is a Python proxy that exposes a single OpenAI-compatible `/v1/chat/completions` endpoint to trusted clients and fans requests out across multiple backend LLM servers (also OpenAI-compatible). It handles load distribution, backend failure, model routing, and streaming — all transparently to the caller.

## Objective / Problem Statement

Serve multiple concurrent LLM users from a single API URL whilst tolerating failures among backend endpoints. Clients need no API keys; the server holds backend credentials and injects them. Each backend serves exactly one model; the server maps inbound model names to appropriate backends via config.

**Non-goals:** embeddings, completions (legacy), auth on inbound requests.

**In-progress:** multi-model aliases — one alias key may map to multiple canonical backends (e.g. `qwen3.5` → `[qwen3.5-9b, qwen3.5-9b-q8]`). Config layer complete (Task 1); registry updated (Task 2); dispatcher updated (Task 3 — queues by model group, `PendingRequest.canonical_models: list[str]`); router adaptation pending (Tasks 4+).

## Architecture

```
Client (trusted)
    │  POST /v1/chat/completions
    ▼
┌─────────────────────────────┐
│  router.py  (FastAPI)        │  – resolves model alias, validates model exists,
│                              │    stamps request_id/timestamp, enqueues PendingRequest,
│                              │    awaits Future result; logs 404/503/504 via LogWriter
└──────────┬──────────────────┘
           │ per-model-group asyncio.Queue (key = sorted canonical_models joined by ","; max_queue_size → 503 if full)
           ▼
┌─────────────────────────────┐
│  dispatcher.py               │  – one run_for_model() loop per model group (queue key)
│                              │    polls BackendRegistry for a free backend across group,
│                              │    spawns _handle_request Task per request;
│                              │    emits LogRecord on success/error (non-streaming only)
└──────────┬──────────────────┘
           │ exhausts free backends on failure before 502
           ▼
┌─────────────────────────────┐
│  forwarder.py                │  – rewrites model field, injects API key,
│                              │    2 HTTP attempts per backend (httpx async)
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  registry.py                 │  – BackendEntry state: FREE → BUSY → FREE/FAILED
│  (BackendRegistry)           │    single asyncio.Lock, round-robin via last_used_at;
│                              │    acquire_backend accepts list[str] (multi-model aware)
└─────────────────────────────┘

Background tasks (started at lifespan):
  health.py     – startup probe of ALL backends; hourly probe of FAILED-only backends
  log_writer.py – drains asyncio.Queue, batches LogRecords into daily SQLite files

Dashboard (standalone, no aggregate_server imports):
  dashboard/data.py       – load_data() reads daily .db files into pd.DataFrame
  dashboard/regression.py – fit_response_model() via numpy lstsq
  dashboard/charts.py     – Altair chart helpers
  dashboard/app.py        – Streamlit layout (Overview + Per Backend tabs)
  dashboard.py            – entry point: streamlit run dashboard.py
```

## Process Flow

1. **Startup** (`router.py` lifespan): load `config.yaml` → build `BackendRegistry` → call
   `get_model_groups(cfg, registry.get_canonical_models())` → create `Dispatcher(groups=...)` →
   create shared `httpx.AsyncClient` → `health._check_all(failed_only=False)` probes all backends →
   start one `dispatcher.run_for_model(queue_key)` task per model group + hourly health task +
   `log_writer.run()` task. **NOTE: router lifespan not yet updated — pending Tasks 4+.**
2. **Request in** (`router.py`): parse body → stamp `request_id` (UUID4) + `timestamp` (unix) +
   `enqueue_at` (monotonic) → `resolve_model()` alias lookup (returns `list[str]`) → 404 if no backends
   (logs via LogWriter) → check queue depth → 503 if full (logs) → create `asyncio.Future` →
   `dispatcher.enqueue(PendingRequest(canonical_models=[...]))`. **NOTE: router not yet updated — pending Tasks 4+.**
3. **Dispatch** (`dispatcher.py`): `queue.get()` → poll `registry.acquire_backend(canonical_models)`
   every 0.1s (acquires any backend matching any model in the group) → backend acquired →
   `create_task(_handle_request(entry, pending))`.
4. **Forward** (`forwarder.py`): rewrite `body["model"]` to backend's model → inject `Authorization`
   header → `_post_once()` × 2 on failure → `ForwardResult` (stream open or full response).
5. **Escalation** (`dispatcher._handle_request`): `ForwardError` → `release(failed=True)` → acquire
   next free backend → repeat until exhausted → `future.set_exception(ForwardError(502))`.
6. **Logging** (`dispatcher._emit_success` / `_emit_error`): non-streaming requests only — build
   `LogRecord` with timing, token counts, backend_id; call `log_writer.enqueue()`.
7. **Log write** (`log_writer.run`): drain queue in batches → group by UTC date → write to
   `./data/logs/YYYY-MM-DD.db` via `aiosqlite`.
8. **Response back** (`router.py`): `asyncio.wait_for(future, queue_timeout)` → `StreamingResponse`
   or `JSONResponse`. Timeout → 504 (logs); `QueueFullError` → 503 (logs).
9. **Stream release**: `_tracked_stream(gen, on_done_fn)` — tracks `failed` flag via `except Exception`;
   calls `on_done_fn(failed)` in `finally`. `GeneratorExit` (client disconnect) is `BaseException`,
   so it marks backend FREE, not FAILED.
10. **Health** (`health.py`): every 3600s, `_check_all(failed_only=True)` probes FAILED backends
    with 1-token completion; 2xx → `restore_backend()` (FAILED → FREE).

## Key Files & Directories

| Path | Purpose |
|------|---------|
| `aggregate_server/registry.py` | Backend state machine; round-robin acquisition logic |
| `aggregate_server/dispatcher.py` | Per-model queues; request lifecycle; retry escalation; logging hooks |
| `aggregate_server/forwarder.py` | httpx forwarding; model/key rewrite; 2-attempt retry |
| `aggregate_server/health.py` | Startup + hourly backend probe; FAILED → FREE restoration |
| `aggregate_server/router.py` | FastAPI app, lifespan wiring, `/v1/chat/completions`, `/v1/models` |
| `aggregate_server/config.py` | YAML load + Pydantic v2 validation; `resolve_model()` → `list[str]`; `get_callable_models()`; `get_model_groups()` |
| `aggregate_server/log_writer.py` | `LogRecord` dataclass + `LogWriter` (queue → daily SQLite files) |
| `config.yaml` | Backend definitions, model aliases, timeout/queue tuning (not committed with secrets) |
| `dashboard/data.py` | `load_data(db_dir, days)` → pd.DataFrame from daily .db files (**implemented**) |
| `dashboard/regression.py` | `fit_response_model(df)` → `RegressionResult` via numpy lstsq (**implemented**) |
| `dashboard/charts.py` | Altair chart helpers: request count, tokens, error rate (**implemented**) |
| `dashboard/app.py` | Streamlit layout: Overview + Per Backend tabs (**implemented**) |
| `dashboard.py` | Entry point: `streamlit run dashboard.py` (**implemented**) |
| `docs/superpowers/plans/2026-06-07-sqlite-logging-dashboard.md` | Full 9-task implementation plan (complete) |
| `docs/superpowers/specs/2026-06-07-sqlite-logging-dashboard-design.md` | Design spec for logging/dashboard |
| `docs/superpowers/specs/2026-06-09-starry-karp-test-script-design.md` | Design spec for multi-model aliases |

## Encountered Errors & Solutions

- **2026-06-07 Error**: `TypeError: AsyncClient.send() got an unexpected keyword argument 'timeout'`
  in `forwarder._post_once`.
  **Cause**: `httpx.AsyncClient.send()` does not accept a `timeout` parameter in the installed
  version (0.27+). The timeout must be passed to higher-level methods like `client.post()`.
  **Fix**: Changed non-streaming path to use `client.post(..., timeout=httpx.Timeout(...))`
  instead of `client.send(req, timeout=...)`. Streaming path continues to use
  `client.send(req, stream=True)` with no per-call timeout override.

- **2026-06-07 Error**: `FastAPIError: Invalid args for response field` on
  `StreamingResponse | JSONResponse` return type.
  **Cause**: FastAPI cannot generate a Pydantic response schema from a union of two `Response`
  subclasses.
  **Fix**: Added `response_model=None` to the `@app.post("/v1/chat/completions")` decorator.

- **2026-06-07 Error**: `ModuleNotFoundError: No module named 'pandas'` when running
  `tests/test_dashboard_data.py`.
  **Cause**: `pandas` and `numpy` were not declared in `pyproject.toml`; only the base server
  deps were listed. The dashboard extras group had not been created yet.
  **Fix**: Added `[project.optional-dependencies] dashboard = ["pandas>=2.2.0", "numpy>=1.26.0"]`
  to `pyproject.toml` and ran `uv sync --extra dashboard --extra dev`.

- **2026-06-07 Error**: `ModuleNotFoundError: No module named 'altair'` when verifying Task 7.
  **Cause**: `altair`, `streamlit`, and `streamlit-autorefresh` were not listed in the `dashboard`
  optional-dependencies group in `pyproject.toml`; the task spec assumed they were pre-installed.
  **Fix**: Added the three packages to `pyproject.toml` dashboard extras and ran
  `uv sync --extra dashboard --extra dev`. The `pytest` module was also temporarily lost (same
  mechanism) and restored by re-syncing with `--extra dev`.

- **2026-06-07 Error**: mypy `unused-ignore` + wrong error code on `dashboard/charts.py` returns.
  **Cause**: `alt.layer().properties()` returns `LayerChart | FacetChart` (a concrete union, not `Any`),
  so `# type: ignore[no-any-return]` was the wrong suppression — leaving the real `[return-value]`
  error uncovered and adding a spurious `[unused-ignore]` error on top. Meanwhile, a single-chart
  chain's `.properties()` does return `Any`, correctly needing `[no-any-return]`.
  **Fix**: Applied `# type: ignore[return-value]` on the two `alt.layer(...)` returns and kept
  `# type: ignore[no-any-return]` only on the `alt.Chart(...)` chain return.

- **2026-06-07 Error**: four ruff errors in `dispatcher.py` and `tests/test_dispatcher_logging.py`
  found during Task 9 final lint pass.
  **Cause**: (1) `LogRecord` imported under `TYPE_CHECKING` but also imported locally inside
  `_emit_success`/`_emit_error`, making the module-level import unused (F401). (2) `try/except/pass`
  for JSON parsing (SIM105 — prefer `contextlib.suppress`). (3) `import contextlib` placed after
  third-party block instead of with stdlib (I001). (4) `pytest.raises(Exception)` too broad (B017).
  **Fix**: Removed `LogRecord` from TYPE_CHECKING block; replaced try/except/pass with
  `contextlib.suppress(Exception)`; moved `import contextlib` into stdlib block; tightened raises
  to `pytest.raises(ForwardError)`.

- **2026-06-07 Error**: five pre-existing mypy errors in `config.py` and `router.py` surfaced on
  first full `mypy aggregate_server/ dashboard/` run.
  **Cause**: `Path(path or fallback)` — mypy does not narrow `str | Path | None` through `or`
  (F401-style inference gap). Lifespan function missing return type. `ModelsResponse(data=...)`
  passed `list[dict]` where `list[ModelObject]` required. `StreamingResponse(result.stream_gen)`
  where `stream_gen` is `AsyncGenerator | None` without a None guard.
  **Fix**: Replaced `path or ...` with `path if path is not None else ...`; added
  `-> AsyncIterator[None]` to `lifespan`; fixed `ModelsResponse` call to pass `ModelObject` instances;
  added explicit `if result.stream_gen is None: raise RuntimeError(...)` guard before `StreamingResponse`.

- **2026-06-09 Error**: mypy type errors in `config.py` — `resolve_model` and `get_model_groups`
  had return-type mismatches because `model_aliases` was typed `dict[str, str | list[str]]` even
  though `_normalise_aliases` guaranteed all values were `list[str]` at runtime.
  **Cause**: `_normalise_aliases` ran as a `mode="after"` validator and mutated `self.model_aliases`
  in place, but Pydantic still inferred the field type from the annotation. Mypy saw the wider
  union throughout and reported mismatches on downstream functions that assumed `list[str]`.
  Additionally, `tests/test_config.py` had ruff I001 (unsorted import block).
  **Fix**: Changed field annotation to `dict[str, list[str]]`; moved string coercion into a
  `mode="before"` `@classmethod` validator so the narrowed type is established before field
  instantiation. Sorted the import block in `tests/test_config.py`.

- **2026-06-09 Error**: `test_router.py::test_non_streaming_roundtrip` and
  `test_alias_resolved_before_routing` started returning 404 after Task 1 config changes.
  **Cause**: `resolve_model()` return type changed from `str` to `list[str]`. `router.py` still
  passes the result directly to `registry.has_backends_for_model(canonical)`, which expects a `str`.
  A `list` never matches any backend model name, so every request is treated as model-not-found.
  **Fix**: Pending — router must be updated in Task 2 of the multi-model aliases plan to iterate
  over the resolved list. The 2 failing tests are a known, expected breakage from the Task 1 contract
  change and will be resolved in Task 2.

## Notable Points

- **`resolve_model()` returns `list[str]`**: as of 2026-06-09 (Task 1), the function returns a
  list. `registry.acquire_backend()` accepts `list[str]` (Task 2). `Dispatcher` now queues by
  model group using a sorted comma-joined key (Task 3). The router is NOT YET updated.
  `test_router.py` has 3 known failures until the router is updated in a later task.
- **`model_aliases` values are always normalised to `list[str]`**: the `_normalise_aliases`
  `mode="before"` class-method validator coerces single-string YAML values to a one-element list
  before field instantiation. The field annotation is `dict[str, list[str]]` — mypy can verify
  downstream callers without suppression. Using `mode="after"` for this coercion would preserve
  the wider `str | list[str]` annotation and cause return-type mismatches in `resolve_model` and
  `get_model_groups`.
- **`get_callable_models()` hides aliased canonicals**: only alias keys and un-aliased canonical
  models appear in the public model list. Raw canonical names that are members of an alias group
  (e.g. `qwen3.5-9b`) are filtered out so clients only call the group alias.
- **Per-model-group queues prevent head-of-line blocking**: a slow qwen3.5 group queue does not
  block llama3 requests. One `asyncio.Queue` + one dispatcher loop exists per model group, keyed
  by `",".join(sorted(canonical_models))`. Un-aliased models appear as single-element groups with
  a key equal to the model name itself.
- **Retry exhausts all free backends**: `_handle_request` loops through every free backend for the
  model before returning 502. Worst case: `2 HTTP attempts × N backends` total requests.
- **Streaming keeps backend BUSY for full duration**: the tracked stream wrapper holds the slot
  until the client finishes consuming or disconnects. This is intentional.
- **`GeneratorExit` is `BaseException`, not `Exception`**: client disconnect during streaming
  naturally marks the backend FREE (not FAILED) without special-casing.
- **Health check at startup probes FREE backends too**: the hourly loop only touches FAILED
  backends, but the startup call uses `failed_only=False` to validate all backends before
  accepting traffic.
- **No auth on inbound**: the server assumes a trusted network. API keys are per-backend in
  `config.yaml` and injected outbound only.
- **`queue_timeout` vs `backend_timeout`**: `queue_timeout` limits how long a request waits in
  queue (pre-dispatch); `backend_timeout` limits how long the httpx client waits for a backend
  response. Streaming requests have no `backend_timeout` after the stream opens (by design).
- **Streaming requests are never logged**: `LogWriter` only captures non-streaming requests where
  token usage is available in the JSON response body. Streaming responses cannot be inspected
  without consuming the stream, so they are deliberately excluded.
- **Daily SQLite files**: `LogWriter` writes to `./data/logs/YYYY-MM-DD.db` (UTC date). A single
  batch may span two files if records straddle midnight. The `_write_batch` method groups by date
  and opens each file separately.
- **Log queue is fire-and-forget**: `LogWriter.enqueue()` drops records silently (with a WARNING
  log) when the internal queue is full. Logging failures never propagate to the request path.
- **Dashboard is fully decoupled**: `dashboard/` imports only `sqlite3`, `pandas`, `numpy`,
  `altair`, and `streamlit`. Zero imports from `aggregate_server`. Can run against any compatible
  `.db` files independently.
- **Dashboard deps are an optional extras group**: `pyproject.toml` declares `[project.optional-dependencies]
  dashboard = [...]`. Run `uv sync --extra dashboard --extra dev` to install pandas, numpy, altair,
  streamlit, and streamlit-autorefresh without pulling them into the server's production environment.
  Always sync both `dashboard` and `dev` together to keep pytest available alongside dashboard tools.
- **`load_data` returns a typed empty DataFrame on missing dir**: when `db_dir` does not exist,
  `load_data` returns a zero-row DataFrame with the full column list (including `datetime`) rather
  than raising. Dashboard code can call `len(df) == 0` safely without special-casing the missing
  directory case.
- **Altair stubs return two distinct types**: `alt.layer().properties()` returns `LayerChart | FacetChart`
  (needs `# type: ignore[return-value]`) while a single-chain `alt.Chart(...).properties()` returns `Any`
  (needs `# type: ignore[no-any-return]`). Using the wrong code produces both the original error and
  a bonus `[unused-ignore]` error, making the problem appear doubled.
- **`fit_response_model` uses unconstrained OLS**: numpy `lstsq` can return negative coefficients
  (e.g. negative `latency_ms`) if the data is ill-conditioned or too homogeneous. Dashboard
  display code should guard for negative values before presenting them as physical estimates.

## Terms & Language

- **Canonical model**: the model name as defined in `config.yaml` under `backends[].model`.
  This is what backends expect. Distinct from inbound model names which may be aliases.
- **Model alias**: a mapping in `config.yaml → model_aliases` translating an inbound model name
  (e.g. `qwen3.5`) to one or more canonical model names (`[qwen3.5-9b, qwen3.5-9b-q8]`). Values
  are always stored as `list[str]` after normalisation — single-string YAML values are coerced
  automatically.
- **Model group**: the set of canonical backends that share an alias. `get_model_groups()` returns
  each group as a `list[str]`; un-aliased canonicals appear as single-element groups.
- **Callable model**: a model name a client may legally use — either an alias key or a canonical
  model that is not a member of any alias group. `get_callable_models()` returns this list.
- **PendingRequest**: the internal dataclass queued per incoming request; carries
  `canonical_models: list[str]` (the resolved model group), body, stream flag, `asyncio.Future`,
  plus `request_id`, `timestamp`, `inbound_model`, `enqueue_at` for logging.
- **Queue key**: the string key identifying a dispatcher queue, computed as
  `",".join(sorted(canonical_models))`. Single-model groups produce a key equal to the model name.
  Multi-model alias groups produce a key like `"qwen3.5-9b,qwen3.5-9b-q8"`.
- **ForwardResult**: returned by `forwarder.forward_request()`; holds either a complete
  `httpx.Response` or an async stream generator plus an `is_stream` flag.
- **ForwardError**: exception raised by `forwarder` after retries exhausted; carries `status_code`
  for downstream HTTP response.
- **_tracked_stream**: async generator wrapper that catches exceptions and calls
  `on_done_fn(failed: bool)` in `finally` to release the backend registry slot.
- **last_used_at**: monotonic timestamp on each `BackendEntry`; lower = earlier = preferred by
  round-robin. Reset to `time.monotonic()` on each release.
- **LogRecord**: dataclass capturing a single completed (non-streaming) request: request_id,
  timestamp, models, backend, status_code, timing (queue/backend/total ms), token counts,
  error_message.
- **LogWriter**: background asyncio task; accepts `LogRecord` via `enqueue()`, drains the queue
  in batches, and persists to daily SQLite files.
- **RegressionResult**: dataclass from `dashboard/regression.py`; holds fitted coefficients
  (latency_ms, pp_speed_ms_per_token, tg_speed_ms_per_token), R², and n_samples.

---

## Claude's Insights

> Independent observations — not highlighted by the user.

### User Tendencies

- Accepts well-reasoned recommendations quickly but engages seriously when challenged — the
  grill-me session produced 6 concrete plan improvements with no pushback on any recommendation.
- Prefers decisions to be forced into explicit choices rather than left open; responded well to
  forced-choice questions during interrogation.
- Has strong CLAUDE.md global standards (function length, complexity, line length) that suggest a
  disciplined engineering background. Apply these strictly — he will notice violations.
- Follows a strict TDD discipline as specified in the implementation plan: tests first, confirm
  failure, implement, confirm pass. Does not skip or reorder steps.
- Works through large multi-task plans one task at a time via subagent dispatch, rather than
  attempting everything in one shot. Each task is self-contained with explicit test + commit steps.
- Task specs sometimes assume packages are already installed when they are not (Tasks 7–8 assumed
  altair/streamlit were in the venv). A future Claude session should always verify the venv contents
  before treating a missing import as a code bug.
- Task specs are delivered as agent sub-tasks with explicit verification commands at the end.
  This means the agent is expected to run pytest + ruff + mypy and confirm clean before committing.
  Errors discovered at that point (as in this task) are fixed inline before the commit lands.
- Runs full lint + mypy at the end of a large feature, not per-task. This concentrated several
  fixable errors (ruff I001, SIM105, B017; mypy arg-type, no-untyped-def) into a single Task 9
  pass. They were all straightforward to fix but produced a larger-than-expected change set at the end.
- Implements features in strict layered plans (config → router → dispatcher → tests). Deliberately
  accepts known test failures between tasks — as seen in Task 1 of multi-model aliases, where the
  router breakage is expected and documented as a pending Task 2. Future sessions should not treat
  these as bugs to fix out of scope.

### Project Shortcomings

- **Streaming requests produce no logs**: token usage is unavailable mid-stream without consuming
  it, so streaming requests are silently excluded from all observability. High-volume streaming
  usage would leave the dashboard blind.
- **`asyncio.sleep(0.1)` polling in dispatcher**: introduces up to 100ms dispatch latency when all
  backends are busy. Acceptable for LLM inference but could be replaced with `asyncio.Condition`.
- **No config hot-reload**: adding or removing backends requires a server restart.
- **Test coverage gaps**: `test_router.py` relies heavily on `patch("aggregate_server.health.check_all")`
  to skip the startup probe, which means the startup probe itself is not integration-tested. The
  streaming end-to-end path has no test beyond confirming `is_stream=True` is returned.
- **Router has 3 known failing tests** as of 2026-06-09 (Task 3 done): `test_non_streaming_roundtrip`,
  `test_model_not_found_returns_404`, and `test_alias_resolved_before_routing` fail because `router.py`
  still references `canonical_models` (undefined variable — should be `canonical`) and constructs
  `PendingRequest` with the old `canonical_model=` keyword. This is an expected mid-plan state;
  Tasks 1–3 are complete; router update is in Tasks 4+.
- **`_next_untried_backend` gives up early on health-restore race**: if a health check restores a
  FAILED backend between retry iterations, `_next_untried_backend` returns `None` rather than
  trying again.
- **Dashboard regression model assumes linear relationship**: `fit_response_model` uses OLS
  (numpy `lstsq`) with no outlier removal and no non-negativity constraints. A single extremely
  slow response will skew coefficients, and sparse/homogeneous data can produce negative latency
  or speed estimates. The `r_squared` field signals fit quality but the caller must validate
  sign and plausibility before presenting results as physical measurements.
- **No index on SQLite `requests` table**: dashboard queries do `SELECT *` over entire files. At
  high request volume (>100k rows/day) this will become slow without at least a `timestamp` index.
  Now that the dashboard is fully implemented, this is the most pressing performance risk.
- **`st_autorefresh` emits a ScriptRunContext warning on bare import**: importing `dashboard.app`
  outside of `streamlit run` triggers a harmless warning from `streamlit_autorefresh`. This appears
  in import-verification commands but does not indicate a bug.
- **Dashboard is complete but untested with real data**: all 46 tests pass against mock/in-memory
  data. The dashboard has never been run against a live `./data/logs/` directory — first real use
  may surface edge cases in the Altair timezone handling or resampling.
- **Several pre-existing mypy errors in router.py were never caught**: the `lifespan` function
  lacked a return type, `ModelsResponse` was constructed with `list[dict]` instead of
  `list[ModelObject]`, and `StreamingResponse` was called without a None guard on `stream_gen`.
  None of these caused runtime failures (FastAPI/Pydantic absorbed them silently), which is how
  they survived through all prior tasks without being noticed. The project has now been fully
  mypy-clean since 2026-06-07.
- **Pydantic `mode="after"` validators do not narrow field types for mypy**: mutating `self.field`
  inside a `mode="after"` validator is invisible to mypy — the annotation remains as declared. Any
  coercion that changes the effective type of a field (e.g. `str | list[str]` → `list[str]`) must
  use a `mode="before"` class-method validator so the narrowed type is reflected in the annotation
  and mypy can verify downstream callers. This pattern was applied to `model_aliases` in `config.py`
  on 2026-06-09.

- **2026-06-09 Error**: `TypeError: unhashable type: 'list'` in `Dispatcher.__init__` when building
  the `_queues` dict after Task 3 tests were written but before the dispatcher was updated.
  **Cause**: The old `Dispatcher.__init__` iterated `canonical_models: list[str]` and used each
  string as a dict key. After Task 2 changed the contract to `list[list[str]]` (model groups),
  the inner lists were used as keys, which are unhashable.
  **Fix**: Updated `Dispatcher.__init__` to accept `canonical_model_groups: list[list[str]]` and
  compute string keys via `_key()` (`",".join(sorted(group))`). All queue lookups and enqueue
  operations now use the computed key rather than the raw list.

### Assumptions to Challenge

- **Backends are stable and low in number**: the O(n) lock acquisition and polling loops are fine
  for ≤20 backends but would degrade at 100+.
- **1-token probe is sufficient for health check**: some backends may respond to a 1-token request
  successfully but fail on larger context.
- **Trusted network is guaranteed**: no inbound auth is a hard assumption. If the network
  perimeter ever changes, this needs a fast follow.
- **Non-streaming responses fit in 300s `backend_timeout`**: very long non-streaming responses
  (large context windows) may still time out.
- **UTC date is the right partition key for SQLite files**: if the server runs across a UTC midnight
  boundary during a heavy batch, one batch's writes will span two files. This is handled correctly
  in `_write_batch`, but dashboard `load_data` must be given `days` large enough to cover the
  window of interest.

### Dependencies & Risks

- **httpx**: async HTTP client throughout. The `send()` vs `post()` timeout API already caused one
  bug during initial implementation; future httpx upgrades should be tested carefully.
- **aiosqlite**: new core dependency as of 2026-06-07. Wraps stdlib `sqlite3` with async context
  managers. No known issues, but it is a single-maintainer library — worth monitoring.
- **FastAPI `StreamingResponse`**: the stream release timing depends on how FastAPI/Starlette
  iterates the async generator. The `httpx2` migration warning from `TestClient` indicates
  Starlette is already deprecating httpx compatibility.
- **Backend LLM servers**: heterogeneous set (llama.cpp, Ollama, vLLM, etc.) — all assumed to
  implement `/v1/chat/completions` faithfully, including the `usage` field in non-streaming
  responses. If a backend omits `usage`, token counts will be `None` and regression data will
  be sparse.

### Potential Areas of Exploration

- **Complete multi-model aliases plan**: Tasks 1 (config), 2 (registry), and 3 (dispatcher) are
  done. Tasks 4+ (router adaptation, model-list filtering, integration testing) remain. The design
  spec is at `docs/superpowers/specs/2026-06-09-starry-karp-test-script-design.md`.
- **Index SQLite tables**: add `CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)`
  after table creation to keep dashboard queries fast as data grows.
- **Non-negativity constraint on regression**: replace `np.linalg.lstsq` with
  `scipy.optimize.nnls` or `scipy.optimize.lsq_linear` with bounds to guarantee physically
  meaningful coefficients (latency ≥ 0, both speeds ≥ 0) without changing the public API.
- **Log streaming token counts via `x-usage` header**: some backends (vLLM, llama.cpp) return
  token counts in response headers even for streaming; parse these to fill the logging gap.
- **Prometheus metrics endpoint** (`/metrics`): queue depth, backend state counts, p95 latency
  histograms — complementary to the SQLite dashboard.
- **Priority queues**: some users/models may warrant higher priority; `asyncio.PriorityQueue`
  is a drop-in replacement.
- **`asyncio.Condition` in dispatcher**: replace 0.1s polling with a condition variable notified
  on `release_backend` to eliminate dispatch latency under load.
- **Weighted round-robin**: backends with more VRAM could be assigned higher weight in selection
  rather than pure last-used ordering.
