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
