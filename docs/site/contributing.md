# Contributing

## Development environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then synchronize
an editable installation with the development and documentation tools:

```bash
uv sync --locked --extra dev --extra docs
```

Python 3.10 or newer and a C compiler are required when building from source.
uv supplies the isolated build dependencies (Cython, NumPy, SciPy, and setuptools).
The Cython extension calls SciPy's BLAS interface; it is exercised by the numerical tests.

`uv.lock` records the resolved environments across supported Python versions.
After editing dependencies, run `uv lock` and commit both the lockfile and
`pyproject.toml`. CI uses `--locked` to reject a stale lockfile.

## Checks

Run the same quality gates as CI:

```bash
uv run --frozen ruff check src tests scripts setup.py
uv run --frozen ruff format --check src tests scripts setup.py
uv run --frozen ty check src tests scripts
uv run --frozen pytest -p no:capture
```

Install the matching Git hooks with `uv run pre-commit install`. The local hooks
run tools from this project's uv environment rather than separate tool environments.

## Documentation

```bash
uv run --frozen zensical build --strict
uv run --frozen zensical serve
```

Configuration lives in `zensical.toml`, pages in `docs/site/`, and theme overrides
in `docs_theme/`. Generated HTML goes to the ignored root `site/` directory.
The documentation source directory is tracked. API pages inspect the installed
package, so the Cython extension must build before generating the reference.

Use `--clean` only in a disposable checkout: Zensical may clear `.cache/`, which
can contain unrelated local artifacts. CI builds documentation in fresh checkouts.

## Distributions

```bash
uv build
```

This produces an sdist and a platform-specific wheel under `dist/`. uv runs
setuptools as the build backend because this package contains a Cython extension.
Versions come from Git tags through setuptools-scm; release checkouts need full
Git history. Source archives carry the generated version metadata and Cython sources.

CI builds and installs the sdist separately. cibuildwheel builds and tests wheels
for CPython 3.10–3.14 on Linux x86_64 and macOS Intel/Apple Silicon. Each installed
artifact runs `scripts/smoke_test.py`, which exercises the compiled solver, a curve
fit, and the `mutvar` CLI. These platforms are the binary release targets; other
platforms may need a source build.

## Release setup

The workflows follow the same separation as jaxqtl: CI, package builds, releases,
and documentation deployment. CI and package builds run for pull requests and
pushes to `main`. Publishing a GitHub release triggers tested distributions and
PyPI publication; documentation deploys on a release or manual dispatch.

Repository administrators must configure:

- A PyPI trusted publisher for `quattro/mut_var`, workflow `release.yml`,
  environment `pypi`, and the `mut-var` project. No PyPI token is stored in CI.
- GitHub Pages with **GitHub Actions** as the source and a `github-pages`
  environment permitting manual deployments from `main` and release tags.
- Any desired required reviews on the `pypi` and `github-pages` environments.

The migration only defines these workflows; it does not publish a package,
create a release, or change repository/environment settings.
