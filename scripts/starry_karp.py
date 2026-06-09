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
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────

FAKE_BACKENDS: list[dict] = [
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
        received = time.monotonic()
        await asyncio.sleep(latency)
        responded = time.monotonic()
        requests_log.append(
            RequestRecord(
                model=body.get("model", ""),
                received_at=received,
                responded_at=responded,
            )
        )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", ""),
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
