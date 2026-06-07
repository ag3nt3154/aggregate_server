from __future__ import annotations

import asyncio

from aggregate_server.config import AppConfig
from aggregate_server.registry import BackendRegistry, BackendState


def _registry(cfg: AppConfig) -> BackendRegistry:
    return BackendRegistry(cfg.backends)


async def test_acquire_returns_free_backend(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend("qwen3.5")
    assert entry is not None
    assert entry.state == BackendState.BUSY


async def test_acquire_returns_none_when_none_free(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    await reg.acquire_backend("llama3")  # only one llama3 backend
    result = await reg.acquire_backend("llama3")
    assert result is None


async def test_round_robin_prefers_oldest(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    first = await reg.acquire_backend("qwen3.5")
    assert first is not None
    await reg.release_backend(first)

    # Release sets last_used_at to now; the OTHER backend has a lower last_used_at
    second = await reg.acquire_backend("qwen3.5")
    assert second is not None
    assert second.config.id != first.config.id


async def test_release_marks_free(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend("qwen3.5")
    assert entry is not None
    await reg.release_backend(entry)
    assert entry.state == BackendState.FREE


async def test_release_failed_marks_failed(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend("qwen3.5")
    assert entry is not None
    await reg.release_backend(entry, failed=True)
    assert entry.state == BackendState.FAILED


async def test_failed_backend_not_acquired(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    e1 = await reg.acquire_backend("qwen3.5")
    assert e1 is not None
    await reg.release_backend(e1, failed=True)
    # Only b2 is free now
    e2 = await reg.acquire_backend("qwen3.5")
    assert e2 is not None
    assert e2.config.id != e1.config.id


async def test_restore_backend(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    entry = await reg.acquire_backend("qwen3.5")
    assert entry is not None
    await reg.release_backend(entry, failed=True)
    await reg.restore_backend(entry)
    assert entry.state == BackendState.FREE


async def test_concurrent_acquire_no_double_claim(sample_config: AppConfig) -> None:
    reg = _registry(sample_config)
    results = await asyncio.gather(
        reg.acquire_backend("qwen3.5"),
        reg.acquire_backend("qwen3.5"),
    )
    busy = [r for r in results if r is not None]
    ids = [r.config.id for r in busy]
    assert len(ids) == len(set(ids)), "Same backend claimed twice concurrently"
