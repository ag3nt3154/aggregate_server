from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from aggregate_server.config import AppConfig, BackendConfig, load_config, resolve_model


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_config_valid(tmp_path: Path) -> None:
    path = _write_config(tmp_path, """
        backends:
          - id: b1
            url: http://host:8080
            api_key: key1
            model: gpt4
        model_aliases:
          gpt-4: gpt4
        queue_timeout: 30
    """)
    cfg = load_config(path)
    assert len(cfg.backends) == 1
    assert cfg.backends[0].id == "b1"
    assert cfg.model_aliases == {"gpt-4": "gpt4"}
    assert cfg.queue_timeout == 30.0


def test_load_config_defaults(tmp_path: Path) -> None:
    path = _write_config(tmp_path, """
        backends:
          - id: b1
            url: http://host:8080
            api_key: key
            model: m
    """)
    cfg = load_config(path)
    assert cfg.queue_timeout == 60.0
    assert cfg.backend_timeout == 300.0
    assert cfg.max_queue_size == 100


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate backend IDs"):
        AppConfig(
            backends=[
                BackendConfig(id="same", url="http://a:8080", api_key="k", model="m"),
                BackendConfig(id="same", url="http://b:8080", api_key="k", model="m"),
            ]
        )


def test_resolve_model_with_alias(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "qwen-chat") == "qwen3.5"


def test_resolve_model_no_alias(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "qwen3.5") == "qwen3.5"


def test_resolve_model_unknown_passthrough(sample_config: AppConfig) -> None:
    assert resolve_model(sample_config, "unknown-model") == "unknown-model"
