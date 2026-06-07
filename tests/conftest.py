from __future__ import annotations

import pytest

from aggregate_server.config import AppConfig, BackendConfig


@pytest.fixture
def sample_config() -> AppConfig:
    return AppConfig(
        backends=[
            BackendConfig(id="b1", url="http://backend1:8080", api_key="k1", model="qwen3.5"),
            BackendConfig(id="b2", url="http://backend2:8080", api_key="k2", model="qwen3.5"),
            BackendConfig(id="b3", url="http://backend3:8080", api_key="k3", model="llama3"),
        ],
        model_aliases={"qwen-chat": "qwen3.5"},
        queue_timeout=5.0,
        backend_timeout=10.0,
        max_queue_size=10,
    )
