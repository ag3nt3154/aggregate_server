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
    model_aliases: dict[str, str | list[str]] = {}
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

    @model_validator(mode="after")
    def _normalise_aliases(self) -> AppConfig:
        self.model_aliases = {
            k: ([v] if isinstance(v, str) else list(v))
            for k, v in self.model_aliases.items()
        }
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.yaml. Raises FileNotFoundError or ValidationError on failure."""
    resolved = Path(path if path is not None else os.getenv("CONFIG_PATH", "config.yaml"))
    raw = yaml.safe_load(resolved.read_text())
    return AppConfig.model_validate(raw)


def resolve_model(config: AppConfig, inbound_model: str) -> list[str]:
    """Resolve an inbound model name through aliases to its canonical form(s)."""
    return config.model_aliases.get(inbound_model, [inbound_model])


def get_callable_models(config: AppConfig, canonical_models: list[str]) -> list[str]:
    """Models a client may name: alias keys + un-aliased canonicals."""
    aliased = {c for targets in config.model_aliases.values() for c in targets}
    unaliased = sorted(m for m in canonical_models if m not in aliased)
    return sorted(config.model_aliases.keys()) + unaliased


def get_model_groups(config: AppConfig, canonical_models: list[str]) -> list[list[str]]:
    """All dispatch groups: each alias's target list + single-element groups for un-aliased."""
    aliased = {c for targets in config.model_aliases.values() for c in targets}
    unaliased = [[m] for m in canonical_models if m not in aliased]
    return list(config.model_aliases.values()) + unaliased
