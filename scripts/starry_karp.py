#!/usr/bin/env python3
"""
Starry Karp — end-to-end integration test for aggregate_server.

Runs 4 fake OpenAI-compatible backends + the aggregate server in-process,
then verifies load balancing (Phase 1) and per-model queue isolation (Phase 2).

Usage:
    python scripts/starry_karp.py
Exit codes:
    0 — all assertions passed
    1 — one or more assertions failed
"""
from __future__ import annotations

import asyncio
import os  # noqa: F401
import sys  # noqa: F401
import tempfile  # noqa: F401
import time
from dataclasses import dataclass

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────

FAKE_BACKENDS: list[dict[str, object]] = [
    {"id": "backend_1", "port": 9001, "latency": 0.5},
    {"id": "backend_2", "port": 9002, "latency": 1.0},
    {"id": "backend_3", "port": 9003, "latency": 2.0},
    {"id": "backend_4", "port": 9004, "latency": 3.0},
]
AGG_PORT = 8765

# ── FakeBackend ───────────────────────────────────────────────────────────────


@dataclass
class RequestRecord:
    model: str
    received_at: float
    responded_at: float


def make_fake_backend(backend_id: str, latency: float) -> FastAPI:
    """Return a FastAPI app that simulates an OpenAI-compatible backend."""
    app: FastAPI = FastAPI()
    requests_log: list[RequestRecord] = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        model: str = body.get("model", "")
        received = time.monotonic()
        await asyncio.sleep(latency)
        responded = time.monotonic()
        requests_log.append(
            RequestRecord(
                model=model,
                received_at=received,
                responded_at=responded,
            )
        )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        )

    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(
            {
                "backend_id": backend_id,
                "hit_count": len(requests_log),
                "requests": [
                    {
                        "model": r.model,
                        "received_at": r.received_at,
                        "responded_at": r.responded_at,
                    }
                    for r in requests_log
                ],
            }
        )

    @app.post("/reset")
    async def reset() -> JSONResponse:
        requests_log.clear()
        return JSONResponse({"status": "ok"})

    return app


# ── Server lifecycle ──────────────────────────────────────────────────────────


async def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Poll until a TCP port accepts connections or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"Port {port} not ready after {timeout:.0f}s")


async def start_server(
    app: FastAPI, port: int
) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """Start a uvicorn server on the given port and wait until it accepts connections."""
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", loop="none"
    )
    server = uvicorn.Server(config)
    task: asyncio.Task[None] = asyncio.create_task(server.serve())
    await wait_for_port("127.0.0.1", port)
    return server, task


async def stop_server(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    """Signal the server to exit and wait for the task to finish."""
    server.should_exit = True
    await task


# ── Config writers ────────────────────────────────────────────────────────────


def _backend_entry(backend_id: str, port: int, model: str) -> dict[str, object]:
    return {
        "id": backend_id,
        "url": f"http://127.0.0.1:{port}",
        "api_key": "test-key",
        "model": model,
    }


def write_phase1_config(path: str) -> None:
    """All 4 backends serve 'test-model'."""
    cfg = {
        "backends": [
            _backend_entry(b["id"], b["port"], "test-model")
            for b in FAKE_BACKENDS
        ],
        "queue_timeout": 30,
        "backend_timeout": 60,
        "max_queue_size": 50,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def write_phase2_config(path: str) -> None:
    """Backends 1+2 serve 'model-a', backends 3+4 serve 'model-b'."""
    model_map: dict[str, str] = {
        "backend_1": "model-a",
        "backend_2": "model-a",
        "backend_3": "model-b",
        "backend_4": "model-b",
    }
    cfg = {
        "backends": [
            _backend_entry(b["id"], b["port"], model_map[b["id"]])
            for b in FAKE_BACKENDS
        ],
        "queue_timeout": 30,
        "backend_timeout": 60,
        "max_queue_size": 50,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


async def wait_for_agg_server(
    client: httpx.AsyncClient, timeout: float = 20.0
) -> None:
    """Poll /v1/models until the aggregate server returns 200."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = await client.get(
                f"http://127.0.0.1:{AGG_PORT}/v1/models", timeout=2.0
            )
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError("Aggregate server did not become ready in time")
