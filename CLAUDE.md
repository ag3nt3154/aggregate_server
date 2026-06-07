# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`aggregate_server` is a Python-based aggregation server. The repository is in early scaffolding stage — refer to `PROJECT_CONTEXT.md` (once created) for evolving architecture details.

## Language & Tooling

- **Language:** Python
- **Linter/Formatter:** Ruff (`.ruff_cache/` is gitignored — prefer `ruff` over flake8/black)
- **Type checker:** mypy or pyright (`.mypy_cache/` is gitignored)
- **Package manager:** uv, poetry, or pdm (none committed yet — confirm with `pyproject.toml` or `Pipfile` once present)

## Commands

Commands will be established once the project scaffolding is in place. Typical entrypoints to expect:

```bash
# Install dependencies (update once package manager is confirmed)
uv sync            # if using uv
poetry install     # if using poetry

# Run the server
python -m aggregate_server

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name -v

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .
```

## Coding Standards

- Functions: ≤ 100 lines
- Cyclomatic complexity: ≤ 8
- Positional parameters: ≤ 5
- Line length: 100 characters
- Files: ≤ 500 lines

## Architecture

Architecture documentation will live in `PROJECT_CONTEXT.md` at the repo root. Update it after every significant task using the `update-project-context` skill.
