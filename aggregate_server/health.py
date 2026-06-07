from __future__ import annotations

import asyncio
import logging

import httpx

from aggregate_server.registry import BackendEntry, BackendRegistry, BackendState

logger = logging.getLogger(__name__)


async def _probe_backend(
    registry: BackendRegistry,
    client: httpx.AsyncClient,
    entry: BackendEntry,
) -> None:
    """Send a 1-token probe to the backend. Restore to FREE on 2xx; leave FAILED otherwise."""
    probe = {
        "model": entry.config.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    url = str(entry.config.url).rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {entry.config.api_key}"}
    try:
        resp = await client.post(url, json=probe, headers=headers, timeout=15.0)
        await resp.aclose()
        if resp.status_code < 300:
            await registry.restore_backend(entry)
            logger.info("Backend %s restored to FREE", entry.config.id)
        else:
            logger.debug("Backend %s probe returned %d", entry.config.id, resp.status_code)
    except httpx.RequestError as exc:
        logger.debug("Backend %s probe error: %s", entry.config.id, exc)


async def check_all(
    registry: BackendRegistry,
    client: httpx.AsyncClient,
    *,
    failed_only: bool,
) -> None:
    """Probe backends in parallel. Pass failed_only=False for the startup probe."""
    entries = await registry.list_all()
    targets = [e for e in entries if not failed_only or e.state == BackendState.FAILED]
    if not targets:
        return
    await asyncio.gather(*[_probe_backend(registry, client, e) for e in targets],
                         return_exceptions=True)


async def run_health_checks(
    registry: BackendRegistry,
    client: httpx.AsyncClient,
    *,
    interval: float = 3600.0,
) -> None:
    """Background task: probe all FAILED backends every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        logger.info("Running scheduled health checks on FAILED backends")
        await check_all(registry, client, failed_only=True)
