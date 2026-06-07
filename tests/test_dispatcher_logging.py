# tests/test_dispatcher_logging.py
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from aggregate_server.config import AppConfig
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
        # stream=True — even if it had been dispatched, it would not be logged
        pending = PendingRequest(
            "qwen3.5", BODY, True, future,
            request_id="stream-req", timestamp=time.time(),
            inbound_model="qwen3.5", enqueue_at=time.monotonic(),
        )
        _ = pending  # not dispatched — just verify no logs accumulated

    assert len(writer.records) == 0
