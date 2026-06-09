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
import contextlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import TypedDict

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────


class _BackendSpec(TypedDict):
    id: str
    port: int
    latency: float


FAKE_BACKENDS: list[_BackendSpec] = [
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
    try:
        await wait_for_port("127.0.0.1", port)
    except Exception:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
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


# ── TestRunner ────────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    status_code: int
    body: dict[str, object]


async def _send_one(client: httpx.AsyncClient, model: str) -> RequestResult:
    try:
        resp = await client.post(
            f"http://127.0.0.1:{AGG_PORT}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "ping"}]},
            timeout=60.0,
        )
        return RequestResult(status_code=resp.status_code, body=resp.json())
    except Exception as exc:
        return RequestResult(status_code=0, body={"error": str(exc)})


async def send_wave(
    client: httpx.AsyncClient, model: str, count: int
) -> list[RequestResult]:
    """Fire `count` requests concurrently, all for the same model."""
    return list(
        await asyncio.gather(*[_send_one(client, model) for _ in range(count)])
    )


async def run_phase1(client: httpx.AsyncClient) -> list[RequestResult]:
    """4 waves of 5 requests, all targeting 'test-model'."""
    results: list[RequestResult] = []
    for _ in range(4):
        results.extend(await send_wave(client, "test-model", 5))
        await asyncio.sleep(0.5)
    return results


async def run_phase2(client: httpx.AsyncClient) -> list[RequestResult]:
    """4 waves of 5 requests, alternating between 'model-a' and 'model-b'."""
    results: list[RequestResult] = []
    for i in range(4):
        model = "model-a" if i % 2 == 0 else "model-b"
        results.extend(await send_wave(client, model, 5))
        await asyncio.sleep(0.5)
    return results


# ── Verifier ──────────────────────────────────────────────────────────────────


@dataclass
class AssertionResult:
    passed: bool
    message: str


def verify_phase1(
    results: list[RequestResult],
    stats: dict[str, dict[str, object]],
    elapsed: float,
) -> list[AssertionResult]:
    """Run Phase 1 assertions: all 200s, all backends hit, faster > slower."""
    checks: list[AssertionResult] = []
    failed = [r.status_code for r in results if r.status_code != 200]
    checks.append(AssertionResult(
        not failed,
        f"All 20 requests returned 200 (non-200 codes: {failed})",
    ))
    all_hit = all(stats[b["id"]]["hit_count"] > 0 for b in FAKE_BACKENDS)
    hit_counts = {b["id"]: stats[b["id"]]["hit_count"] for b in FAKE_BACKENDS}
    checks.append(AssertionResult(
        all_hit,
        f"All 4 backends received traffic (counts: {hit_counts})",
    ))
    b1 = stats["backend_1"]["hit_count"]
    b4 = stats["backend_4"]["hit_count"]
    checks.append(AssertionResult(
        b1 > b4,  # type: ignore[operator]
        f"backend_1 ({b1} hits) > backend_4 ({b4} hits) — faster backend serves more",
    ))
    total = sum(int(stats[b["id"]]["hit_count"]) for b in FAKE_BACKENDS)  # type: ignore[arg-type]
    checks.append(AssertionResult(
        total == 20,
        f"Total hits across all backends = 20 (got {total})",
    ))
    checks.append(AssertionResult(
        elapsed < 60.0,
        f"Elapsed {elapsed:.1f}s < 60s serialised ceiling (proves concurrency)",
    ))
    return checks


def verify_phase2(
    results: list[RequestResult],
    stats: dict[str, dict[str, object]],
) -> list[AssertionResult]:
    """Run Phase 2 assertions: all 200s, per-model queue isolation holds."""
    checks: list[AssertionResult] = []
    failed = [r.status_code for r in results if r.status_code != 200]
    checks.append(AssertionResult(
        not failed,
        f"All 20 requests returned 200 (non-200 codes: {failed})",
    ))
    a_hits = int(stats["backend_1"]["hit_count"]) + int(stats["backend_2"]["hit_count"])  # type: ignore[arg-type]
    b_hits = int(stats["backend_3"]["hit_count"]) + int(stats["backend_4"]["hit_count"])  # type: ignore[arg-type]
    checks.append(AssertionResult(
        a_hits == 10,
        f"model-a backends (1+2) combined = 10 hits (got {a_hits})",
    ))
    checks.append(AssertionResult(
        b_hits == 10,
        f"model-b backends (3+4) combined = 10 hits (got {b_hits})",
    ))
    b3_reqs = stats["backend_3"]["requests"]
    b4_reqs = stats["backend_4"]["requests"]
    b12_reqs = list(stats["backend_1"]["requests"]) + list(stats["backend_2"]["requests"])  # type: ignore[operator]
    b3_models = {r["model"] for r in b3_reqs} if isinstance(b3_reqs, list) else set()  # type: ignore[union-attr]
    b4_models = {r["model"] for r in b4_reqs} if isinstance(b4_reqs, list) else set()  # type: ignore[union-attr]
    b12_models = {r["model"] for r in b12_reqs}  # type: ignore[index]
    checks.append(AssertionResult(
        b3_models <= {"model-b"} and b4_models <= {"model-b"},
        f"model-a traffic never reached backend_3/4 (b3: {b3_models}, b4: {b4_models})",
    ))
    checks.append(AssertionResult(
        b12_models <= {"model-a"},
        f"model-b traffic never reached backend_1/2 (b1+b2 models: {b12_models})",
    ))
    return checks


# ── Stats helpers ─────────────────────────────────────────────────────────────


async def get_stats(client: httpx.AsyncClient, port: int) -> dict[str, object]:
    resp = await client.get(f"http://127.0.0.1:{port}/stats", timeout=5.0)
    return dict(resp.json())


async def reset_stats(client: httpx.AsyncClient, port: int) -> None:
    await client.post(f"http://127.0.0.1:{port}/reset", timeout=5.0)


async def collect_stats(client: httpx.AsyncClient) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for b in FAKE_BACKENDS:
        result[b["id"]] = await get_stats(client, b["port"])
    return result


# ── Report ────────────────────────────────────────────────────────────────────


def _print_checks(checks: list[AssertionResult]) -> bool:
    all_pass = True
    for check in checks:
        icon = "✓" if check.passed else "✗"
        print(f"  {icon} {check.message}")
        if not check.passed:
            all_pass = False
    return all_pass


def print_report(
    phase1_checks: list[AssertionResult],
    phase2_checks: list[AssertionResult],
    phase1_stats: dict[str, dict[str, object]],
) -> bool:
    print("\n====== Starry Karp Integration Test ======\n")
    print("Phase 1: Load Balancing (test-model, 4 backends)")
    p1_pass = _print_checks(phase1_checks)
    if p1_pass:
        dist = " ".join(
            f"{b['id']}={phase1_stats[b['id']]['hit_count']}" for b in FAKE_BACKENDS
        )
        print(f"  Hit distribution: {dist}")
    print()
    print("Phase 2: Queue Isolation (model-a ↔ backend_1/2, model-b ↔ backend_3/4)")
    p2_pass = _print_checks(phase2_checks)
    print()
    if p1_pass and p2_pass:
        print("All assertions passed. Fleet is battle-ready.")
    else:
        print("ASSERTIONS FAILED. Review the output above.")
    return p1_pass and p2_pass


# ── Main ──────────────────────────────────────────────────────────────────────


async def _start_fake_backends() -> list[tuple[uvicorn.Server, asyncio.Task[None]]]:
    servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
    for b in FAKE_BACKENDS:
        app = make_fake_backend(b["id"], float(b["latency"]))
        server, task = await start_server(app, int(b["port"]))
        servers.append((server, task))
    return servers


async def _run_phase(
    client: httpx.AsyncClient,
    config_path: str,
    phase: int,
) -> tuple[list[RequestResult], dict[str, dict[str, object]], float]:
    """Configure and start the aggregate server for the given phase, run requests."""
    if phase == 1:
        write_phase1_config(config_path)
    else:
        write_phase2_config(config_path)
    os.environ["CONFIG_PATH"] = config_path

    from aggregate_server.router import app as agg_app

    print(f"Starting aggregate server (phase {phase})...")
    agg_server, agg_task = await start_server(agg_app, AGG_PORT)
    await wait_for_agg_server(client)
    print(f"  Ready. Running phase {phase} requests...")

    t0 = time.monotonic()
    results = await run_phase1(client) if phase == 1 else await run_phase2(client)
    elapsed = time.monotonic() - t0

    stats = await collect_stats(client)
    await stop_server(agg_server, agg_task)
    return results, stats, elapsed


async def main() -> int:
    print("Starting fake backends...")
    backend_servers = await _start_fake_backends()
    print("  All 4 backends ready.\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name

    phase1_results: list[RequestResult] = []
    phase2_results: list[RequestResult] = []
    phase1_stats: dict[str, dict[str, object]] = {}
    phase2_stats: dict[str, dict[str, object]] = {}
    elapsed = 0.0

    async with httpx.AsyncClient() as client:
        try:
            phase1_results, phase1_stats, elapsed = await _run_phase(
                client, config_path, phase=1
            )
            for b in FAKE_BACKENDS:
                await reset_stats(client, b["port"])

            phase2_results, phase2_stats, _ = await _run_phase(
                client, config_path, phase=2
            )
        finally:
            print("\nStopping fake backends...")
            for server, task in backend_servers:
                await stop_server(server, task)
            with contextlib.suppress(OSError):
                os.unlink(config_path)

    p1_checks = verify_phase1(phase1_results, phase1_stats, elapsed)
    p2_checks = verify_phase2(phase2_results, phase2_stats)
    success = print_report(p1_checks, p2_checks, phase1_stats)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
