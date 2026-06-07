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
            with contextlib.suppress(Exception):
                usage = result.response.json().get("usage", {})
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
