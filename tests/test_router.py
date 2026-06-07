from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from aggregate_server.router import app

RESPONSE_JSON = {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
CHAT_BODY = {"model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def config_path(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
backends:
  - id: b1
    url: http://backend1:8080
    api_key: key1
    model: qwen3.5
model_aliases:
  qwen-chat: qwen3.5
queue_timeout: 5
backend_timeout: 10
max_queue_size: 10
""")
    return str(cfg)


@respx.mock
def test_non_streaming_roundtrip(config_path: str) -> None:
    respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=CHAT_BODY)

    assert resp.status_code == 200
    assert "choices" in resp.json()


@respx.mock
def test_model_not_found_returns_404(config_path: str) -> None:
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                               json={"model": "no-such-model", "messages": []})

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "model_not_found"
    assert "available_models" in body["error"]


@respx.mock
def test_alias_resolved_before_routing() -> None:
    route = respx.post("http://backend1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=RESPONSE_JSON)
    )
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[BackendConfig(id="b1", url="http://backend1:8080",
                                   api_key="k1", model="qwen3.5")],
            model_aliases={"qwen-chat": "qwen3.5"},
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                               json={"model": "qwen-chat", "messages": []})

    assert resp.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen3.5"


def test_list_models() -> None:
    with patch("aggregate_server.router.load_config") as mock_load, \
         patch("aggregate_server.health.check_all", new_callable=AsyncMock):
        from aggregate_server.config import AppConfig, BackendConfig
        mock_load.return_value = AppConfig(
            backends=[
                BackendConfig(id="b1", url="http://b1:8080", api_key="k", model="qwen3.5"),
                BackendConfig(id="b2", url="http://b2:8080", api_key="k", model="llama3"),
            ],
            queue_timeout=5.0, backend_timeout=10.0, max_queue_size=10,
        )
        with TestClient(app) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "qwen3.5" in ids
    assert "llama3" in ids
