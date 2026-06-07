from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from aggregate_server.config import AppConfig
from aggregate_server.dispatcher import Dispatcher, PendingRequest, QueueFullError
from aggregate_server.forwarder import ForwardError, ForwardResult
from aggregate_server.registry import BackendRegistry

RESPONSE_JSON = {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
BODY = {"model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}


def _make_dispatcher(cfg: AppConfig, client: httpx.AsyncClient) -> Dispatcher:
    registry = BackendRegistry(cfg.backends)
    models = registry.get_canonical_models()
    return Dispatcher(
        registry, client, models,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
    ), registry


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
        pending = PendingRequest("qwen3.5", BODY, False, future)

        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        dispatcher.enqueue(pending)
        result = await asyncio.wait_for(future, timeout=2.0)
        task.cancel()

    assert not result.is_stream
    assert result.response is not None


async def test_queue_full_sets_exception(sample_config: AppConfig) -> None:
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client)
        # Fill the queue without running the dispatcher loop
        loop = asyncio.get_running_loop()
        for _ in range(sample_config.max_queue_size):
            future: asyncio.Future[ForwardResult] = loop.create_future()
            dispatcher.enqueue(PendingRequest("qwen3.5", BODY, False, future))

        overflow_future: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest("qwen3.5", BODY, False, overflow_future))

        assert overflow_future.done()
        with pytest.raises(QueueFullError):
            overflow_future.result()


async def test_unknown_model_sets_error(sample_config: AppConfig) -> None:
    async with httpx.AsyncClient() as client:
        dispatcher, _ = _make_dispatcher(sample_config, client)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ForwardResult] = loop.create_future()
        dispatcher.enqueue(PendingRequest("no-such-model", BODY, False, future))

        assert future.done()
        with pytest.raises(ForwardError) as exc_info:
            future.result()
        assert exc_info.value.status_code == 404


@respx.mock
async def test_escalates_to_next_backend_on_failure(sample_config: AppConfig) -> None:
    # b1 always fails, b2 succeeds
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
        dispatcher.enqueue(PendingRequest("qwen3.5", BODY, False, future))

        task = asyncio.create_task(dispatcher.run_for_model("qwen3.5"))
        result = await asyncio.wait_for(future, timeout=3.0)
        task.cancel()

    assert result.response is not None
