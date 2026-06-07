from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import AnyHttpUrl, BaseModel, model_validator


class BackendConfig(BaseModel):
    id: str
    url: AnyHttpUrl
    api_key: str
    model: str


class AppConfig(BaseModel):
    backends: list[BackendConfig]
    model_aliases: dict[str, str] = {}
    queue_timeout: float = 60.0
    backend_timeout: float = 300.0
    max_queue_size: int = 100

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> AppConfig:
        ids = [b.id for b in self.backends]
        if len(ids) != len(set(ids)):
            dupes = {x for x in ids if ids.count(x) > 1}
            raise ValueError(f"Duplicate backend IDs: {dupes}")
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.yaml. Raises FileNotFoundError or ValidationError on failure."""
    resolved = Path(path if path is not None else os.getenv("CONFIG_PATH", "config.yaml"))
    raw = yaml.safe_load(resolved.read_text())
    return AppConfig.model_validate(raw)


def resolve_model(config: AppConfig, inbound_model: str) -> str:
    """Resolve an inbound model name through aliases to its canonical form."""
    return config.model_aliases.get(inbound_model, inbound_model)
