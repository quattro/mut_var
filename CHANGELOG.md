# Changelog

## vX.Y.0 - Breaking Hardening Release

### Breaking Changes

- CLI now enforces boundary validation before numerics execution.
- Public workflow API is canonicalized around package-root exports and numerics workflow modules.
- Curve workflow now supports explicit fit-only mode and isolates plotting side effects.
- CI and release gates now require contract tests, stability tests, and benchmark evidence.

### Removed / Deprecated Behaviors

- Implicit validation and numerics side effects in CLI internals.
- Legacy internal call patterns that bypass canonical workflow entrypoints.
- Unscoped objective/model changes without explicit design and sign-off.

### Migration Guide

1. Use package-root exports documented in `docs/api.md`.
2. Run inference via `run_inference_pipeline` and curve fitting via `run_curve_workflow`.
3. Expect explicit `RESULTS` statuses for failures (`invalid_input`, `empty_subset`, `nonfinite_objective`, etc.).
4. Update automation to use CI/local quality gates:
   - `ruff check src/mut_var tests`
   - `mypy src/mut_var tests`
   - `pytest -p no:capture`
   - `python benchmarks/infer_runtime.py --config benchmarks/config/runtime_baseline.json --output benchmarks/results/latest.json`

### Release-Blocking Performance Gate

Release is blocked unless benchmark evidence is present and steady-state improvement is >= 20%.

- Benchmark artifact: `benchmarks/results/latest.json`
- Gate command: `python scripts/check_release_gate.py --report benchmarks/results/latest.json`
- Representativeness review: `docs/reviews/benchmark-representativeness.md`

### Cross-Reference

See `README.md` migration and contract sections for user-facing guidance.
Human migration review artifact: `docs/reviews/migration-guide-signoff.md`.
