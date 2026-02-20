# Installation

## PyPI

```console
pip install mut-var
```

## Local Editable Install

```console
pip install -e .
```

## Quality Gates

Run the same checks used in CI:

```console
ruff check src/mut_var tests
mypy src/mut_var tests
pytest -p no:capture
```
