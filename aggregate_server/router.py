# aggregate_server/router.py
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aggregate_server.config import AppConfig, load_config, resolve_model
from aggregate_server.dispatcher import Dispatcher, PendingRequest, QueueFullError
from aggregate_server.forwarder import ForwardError, ForwardResult
from aggregate_server.health import check_all, run_health_checks
from aggregate_server.log_writer import LogRecord, LogWriter
from aggregate_server.models import ModelObject, ModelsResponse
from aggregate_server.registry import BackendRegistry

logger = logging.getLogger(__name__)


def _error_response(message: str, code: str, status: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "code": code, **extra}},
    )


def _emit_router_record(
    log_writer: LogWriter,
    request_id: str,
    timestamp: float,
    inbound_model: str,
    canonical_model: str,
    enqueue_at: float,
    status_code: int,
    error_message: str,
) -> None:
    queue_ms = (time.monotonic() - enqueue_at) * 1000
    log_writer.enqueue(LogRecord(
        request_id=request_id,
        timestamp=timestamp,
        inbound_model=inbound_model,
        canonical_model=canonical_model,
        backend_id=None,
        status_code=status_code,
        queue_time_ms=queue_ms,
        backend_time_ms=None,
        total_time_ms=queue_ms,
        input_tokens=None,
        output_tokens=None,
        error_message=error_message,
    ))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    registry = BackendRegistry(cfg.backends)
    client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.backend_timeout, connect=10.0))
    log_writer = LogWriter()

    logger.info("Probing all backends before accepting traffic...")
    await check_all(registry, client, failed_only=False)

    models = registry.get_canonical_models()
    dispatcher = Dispatcher(
        registry, client, models,
        max_queue_size=cfg.max_queue_size,
        backend_timeout=cfg.backend_timeout,
        log_writer=log_writer,
    )

    dispatch_tasks = [asyncio.create_task(dispatcher.run_for_model(m)) for m in models]
    health_task = asyncio.create_task(run_health_checks(registry, client))
    log_task = asyncio.create_task(log_writer.run())

    app.state.config = cfg
    app.state.registry = registry
    app.state.dispatcher = dispatcher
    app.state.log_writer = log_writer

    yield

    for task in [*dispatch_tasks, health_task, log_task]:
        task.cancel()
    await asyncio.gather(*dispatch_tasks, health_task, log_task, return_exceptions=True)
    await client.aclose()
    logger.info("Aggregate server shut down cleanly")


app = FastAPI(title="Aggregate Server", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    registry: BackendRegistry = request.app.state.registry
    models = [ModelObject(id=m) for m in sorted(registry.get_canonical_models())]
    return JSONResponse(ModelsResponse(data=models).model_dump())


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    body: dict[str, Any] = await request.json()
    cfg: AppConfig = request.app.state.config
    registry: BackendRegistry = request.app.state.registry
    dispatcher: Dispatcher = request.app.state.dispatcher
    log_writer: LogWriter = request.app.state.log_writer

    enqueue_at = time.monotonic()
    timestamp = time.time()
    request_id = str(uuid.uuid4())
    stream = bool(body.get("stream", False))
    inbound_model: str = body.get("model", "")
    canonical = resolve_model(cfg, inbound_model)

    if not registry.has_backends_for_model(canonical):
        available = sorted(registry.get_canonical_models())
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 404, f"model '{inbound_model}' not found",
            )
        return _error_response(
            f"The model '{inbound_model}' does not exist. Available: {available}",
            code="model_not_found",
            status=404,
            available_models=available,
        )

    future: asyncio.Future[ForwardResult] = asyncio.get_running_loop().create_future()
    pending = PendingRequest(
        canonical_model=canonical,
        body=body,
        stream=stream,
        result_future=future,
        request_id=request_id,
        timestamp=timestamp,
        inbound_model=inbound_model,
        enqueue_at=enqueue_at,
    )
    dispatcher.enqueue(pending)

    try:
        result = await asyncio.wait_for(asyncio.shield(future), timeout=cfg.queue_timeout)
    except TimeoutError:
        future.cancel()
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 504, "request timed out waiting for a free backend",
            )
        return _error_response(
            "Request timed out waiting for a free backend", "server_error", 504
        )
    except QueueFullError as exc:
        if not stream:
            _emit_router_record(
                log_writer, request_id, timestamp, inbound_model, canonical,
                enqueue_at, 503, str(exc),
            )
        return _error_response(str(exc), "rate_limit_exceeded", 503)
    except ForwardError as exc:
        return _error_response(str(exc), "server_error", exc.status_code)

    if result.is_stream:
        if result.stream_gen is None:
            raise RuntimeError("Stream ForwardResult has no generator")
        return StreamingResponse(result.stream_gen, media_type="text/event-stream")
    if result.response is None:
        raise RuntimeError("Non-stream ForwardResult has no response body")
    return JSONResponse(result.response.json(), status_code=result.response.status_code)
