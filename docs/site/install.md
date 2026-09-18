# Installation

## PyPI

```console
pip install mut-var
```

## Local Editable Install

```console
uv sync --locked --extra dev
```

## Quality Gates

Run the same checks used in CI:

```console
uv run --frozen ruff check src tests scripts setup.py
uv run --frozen ruff format --check src tests scripts setup.py
uv run --frozen ty check src tests scripts
uv run --frozen pytest -p no:capture
```

See [Contributing](contributing.md) for documentation builds, package artifacts, and releases.
