# Multi-Model Aliases Design

> Date: 2026-06-09 | Status: approved

## Problem

Model aliases currently map one inbound name to exactly one canonical model (`dict[str, str]`).
A single alias must be able to map to multiple canonical backend models so that requests for
e.g. `qwen3.5` can be dispatched to any free backend serving `qwen3.5-9b`, `qwen3.5-9b-q8`,
or `qwen3.5-9b-alt` — round-robin across the combined pool.

---

## Config Layer (`config.py`)

### YAML format

`model_aliases` values may now be a string (existing) or a list of strings (new):

```yaml
model_aliases:
  qwen3.5-chat: qwen3.5               # string form — still valid
  qwen3.5: [qwen3.5-9b, qwen3.5-9b-q8, qwen3.5-9b-alt]  # list form
```

### `AppConfig`

```python
model_aliases: dict[str, str | list[str]] = {}
```

A `model_validator(mode="after")` normalises all values to `list[str]` at load time, so the rest
of the codebase always sees `dict[str, list[str]]`.

### `resolve_model`

```python
def resolve_model(config: AppConfig, inbound: str) -> list[str]:
    return config.model_aliases.get(inbound, [inbound])
```

Returns a one-element list for un-aliased canonicals (passthrough default preserved).

### `get_callable_models`

New helper returning what clients may pass as `model` in a request:

```python
def get_callable_models(config: AppConfig, registry: BackendRegistry) -> list[str]:
    aliased_canonicals = {c for targets in config.model_aliases.values() for c in targets}
    unaliased = [m for m in registry.get_canonical_models() if m not in aliased_canonicals]
    return sorted(config.model_aliases.keys()) + sorted(unaliased)
```

---

## Registry (`registry.py`)

### `acquire_backend`

Signature changes from `(canonical_model: str)` to `(canonical_models: list[str])`.
Candidate filter changes from equality to membership:

```python
candidates = [
    e for e in self._entries
    if e.state == BackendState.FREE and e.config.model in canonical_models
]
```

Round-robin selection (longest-idle) is unchanged.

### `has_backends_for_models`

Replaces `has_backends_for_model(str)`:

```python
def has_backends_for_models(self, canonical_models: list[str]) -> bool:
    return any(e.config.model in canonical_models for e in self._entries)
```

### Health checker

`health.py` uses `list_all()` and `restore_backend()` only — it never calls `acquire_backend`.
No changes required to `health.py`.

---

## Dispatcher (`dispatcher.py`)

### `PendingRequest`

`canonical_model: str` → `canonical_models: list[str]`.

```python
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
```

### Queue key

`",".join(sorted(canonical_models))` — one shared queue per alias group.

### `Dispatcher.__init__`

Receives `canonical_model_groups: list[list[str]]` instead of `canonical_models: list[str]`.
The router derives this from config at startup.

### `_acquire_with_poll`

Passes `pending.canonical_models` to `registry.acquire_backend`.

### Logging

- `_emit_success`: logs `entry.config.model` as `canonical_model` — the actually-selected backend.
- `_emit_error`: logs `pending.canonical_models[0]` as `canonical_model` fallback.

---

## Router (`router.py`)

Three call-site changes:

```python
canonical_models = resolve_model(cfg, inbound_model)

if not registry.has_backends_for_models(canonical_models):
    ...

pending = PendingRequest(canonical_models=canonical_models, ...)
```

`_emit_router_record` receives `canonical_models[0]` as `canonical_model` for 404/503/504 log
records (no backend was reached, so first resolved model is the best approximation).

### `/v1/models`

Returns `get_callable_models(cfg, registry)` — alias keys plus any un-aliased canonical models.
Clients never need to know internal canonical model names.

---

## Testing

### `test_config.py`

- List-form alias normalises to `list[str]`.
- String-form alias still normalises to `list[str]` (backward compat).
- `resolve_model` returns `list[str]` in both cases.
- `get_callable_models` returns alias keys + un-aliased canonicals; aliased canonicals excluded.

### `test_registry.py`

- `acquire_backend([A, B, C])` picks the longest-idle FREE backend across all three models.
- `has_backends_for_models` returns `True` when any model in the list has a backend.

### `test_dispatcher.py`

- `PendingRequest` constructed with `canonical_models=[...]`.
- Multi-model alias enqueues into the correct shared queue key.
- Requests route to any matching backend across the alias group.

### Existing tests

All existing `PendingRequest(canonical_model=...)` constructors renamed to `canonical_models=[...]`.
`conftest.py` fixtures updated accordingly.

---

## Non-goals

- Weighted or priority selection across models in an alias group (future work).
- Alias chaining (alias-of-alias).
- Hot-reload of aliases without server restart.
