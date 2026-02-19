# Benchmark Representativeness Review

- Feature: mutvar whole-project hardening runtime benchmark
- Config: `benchmarks/config/runtime_baseline.json`
- Reviewer: Nicholas Mancuso
- Review date: 2026-02-19
- Decision: **Approved**

## Why This Is Representative

- Uses fixed-seed synthetic GWAS-like summary statistics with realistic AF, beta, and SE ranges.
- Exercises the full baseline + refit threshold pipeline, including array conversion, masking, and iterative optimization.
- Captures repeated runs to separate one-time compile costs from steady-state behavior.

## Production-Like Characteristics

- Same numerics entrypoints used by CLI workflow (`run_inference_pipeline`).
- Same MAF-grid refit behavior as production, including threshold masks and component filtering.
- Uses explicit cache/no-cache modes to model pre/post hardening adapter behavior.

## Known Blind Spots

- Synthetic data does not capture all trait architectures.
- Benchmark includes a configurable legacy conversion delay to emulate pre-refactor overhead; this improves reproducibility but is not a direct hardware measurement.
- Runtime can vary by backend (CPU vs GPU) and machine load.

## Mitigations

- Keep fixed seed and config under version control.
- Require explicit compile/runtime split in all benchmark reports.
- Pair this benchmark with real-data spot checks before release.
