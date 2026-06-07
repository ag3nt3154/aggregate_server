from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx

from aggregate_server.registry import BackendEntry


@dataclass
class ForwardResult:
    response: httpx.Response | None = None
    stream_gen: AsyncGenerator[bytes, None] | None = None
    is_stream: bool = False


class ForwardError(Exception):
    """Raised when forwarding fails after all allowed retries."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _rewrite_body(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Return a shallow copy of body with the model field replaced."""
    return {**body, "model": model}


def _build_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _backend_url(entry: BackendEntry) -> str:
    return str(entry.config.url).rstrip("/") + "/v1/chat/completions"


async def _post_once(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    stream: bool,
    timeout: float,
) -> httpx.Response:
    if stream:
        req = client.build_request("POST", url, json=body, headers=headers)
        return await client.send(req, stream=True)
    return await client.post(
        url, json=body, headers=headers,
        timeout=httpx.Timeout(timeout, connect=10.0),
    )


async def _stream_bytes(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


async def tracked_stream(
    gen: AsyncGenerator[bytes, None],
    on_done: Callable[[bool], Coroutine[Any, Any, None]],
) -> AsyncGenerator[bytes, None]:
    """
    Wrap a stream generator to call on_done(failed) when the stream ends.
    GeneratorExit (client disconnect) is BaseException — not caught here,
    so the backend is released as free rather than failed.
    """
    failed = False
    try:
        async for chunk in gen:
            yield chunk
    except Exception:
        failed = True
        raise
    finally:
        await on_done(failed)


async def forward_request(
    client: httpx.AsyncClient,
    entry: BackendEntry,
    body: dict[str, Any],
    *,
    stream: bool,
    backend_timeout: float = 300.0,
) -> ForwardResult:
    """
    Forward a chat completion request to the backend, retrying once on transient failure.
    Raises ForwardError after 2 failed attempts or immediately on 4xx.
    """
    url = _backend_url(entry)
    headers = _build_headers(entry.config.api_key)
    rewritten = _rewrite_body(body, entry.config.model)

    last_error: ForwardError | None = None
    for _ in range(2):
        try:
            response = await _post_once(
                client, url, headers, rewritten, stream=stream, timeout=backend_timeout
            )
        except httpx.RequestError as exc:
            last_error = ForwardError(f"Backend {entry.config.id} connection error: {exc}", 502)
            continue

        if response.status_code >= 500:
            await response.aclose()
            last_error = ForwardError(
                f"Backend {entry.config.id} returned {response.status_code}", 502
            )
            continue

        if response.status_code >= 400:
            if stream:
                await response.aread()
            body_text = response.content.decode(errors="replace")
            await response.aclose()
            raise ForwardError(
                f"Backend {entry.config.id} client error {response.status_code}: {body_text}",
                response.status_code,
            )

        if stream:
            return ForwardResult(is_stream=True, stream_gen=_stream_bytes(response))
        return ForwardResult(is_stream=False, response=response)

    raise last_error or ForwardError(f"Backend {entry.config.id} failed after 2 attempts")
