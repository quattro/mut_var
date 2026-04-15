# mut_var

`mut_var` provides reproducible mutation-variance inference pipelines with explicit failure states for both CLI and Python callers.

## What Is Stable

- CLI entrypoint: `mutvar` with `infer`, `curve`, and `simulate` subcommands.
- Pipeline APIs: `run_inference_pipeline`, `run_curve_pipeline`, and `run_simulation_pipeline`.
- Numerics status contract: `mut_var.types.Solution.result` is the canonical success/failure signal.

## Project Documentation Layout

- Published MkDocs pages are stored in `docs/site/`.
- Design and implementation artifacts are kept under `docs/design-plans/` and `docs/reviews/`.

## Next Steps

- Start with [Installation](install.md).
- Follow [Quickstart](quickstart.md) for end-to-end CLI usage.
- See [Python API](python-api.md) for package-level integration.
