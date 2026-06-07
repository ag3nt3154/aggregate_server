from __future__ import annotations

import json

import httpx
import pytest
import respx

from aggregate_server.config import BackendConfig
from aggregate_server.forwarder import ForwardError, forward_request
from aggregate_server.registry import BackendEntry, BackendState


def _entry(model: str = "qwen3.5", backend_id: str = "b1") -> BackendEntry:
    cfg = BackendConfig(
        id=backend_id, url="http://backend1:8080", api_key="test-key", model=model
    )
    return BackendEntry(config=cfg, state=BackendState.FREE)


CHAT_URL = "http://backend1:8080/v1/chat/completions"
BODY = {"model": "qwen-chat", "messages": [{"role": "user", "content": "hi"}]}
RESPONSE_JSON = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}


@respx.mock
async def test_rewrites_model_field() -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    async with httpx.AsyncClient() as client:
        result = await forward_request(client, _entry("qwen3.5"), BODY, stream=False)

    assert result.is_stream is False
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen3.5"


@respx.mock
async def test_injects_api_key() -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    async with httpx.AsyncClient() as client:
        await forward_request(client, _entry(), BODY, stream=False)

    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"


@respx.mock
async def test_retries_on_500_and_succeeds() -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(500, text="error"),
            httpx.Response(200, json=RESPONSE_JSON),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await forward_request(client, _entry(), BODY, stream=False)

    assert result.response is not None
    assert len(route.calls) == 2


@respx.mock
async def test_raises_after_two_500s() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="err"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ForwardError) as exc_info:
            await forward_request(client, _entry(), BODY, stream=False)
    assert exc_info.value.status_code == 502


@respx.mock
async def test_no_retry_on_400() -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(400, text="bad"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ForwardError) as exc_info:
            await forward_request(client, _entry(), BODY, stream=False)
    assert exc_info.value.status_code == 400
    assert len(route.calls) == 1


@respx.mock
async def test_raises_on_connection_error() -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ForwardError):
            await forward_request(client, _entry(), BODY, stream=False)


@respx.mock
async def test_streaming_returns_generator() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, content=b"data: chunk\n\n")
    )
    async with httpx.AsyncClient() as client:
        result = await forward_request(client, _entry(), BODY, stream=True)
    assert result.is_stream is True
    assert result.stream_gen is not None
