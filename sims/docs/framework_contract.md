# Sims Framework Contract

## Scope
This framework is fully independent from mut_var internals and writes outputs only under:
- `sims/results/`
- `sims/plots/`

It must remain compatible with mut_var inference by file contract.

## Methods-doc readability requirement
Any methods/design Markdown in `sims/docs/` must also be rendered to PDF for human review.

Canonical command:

`python sims/scripts/render_methods_pdf.py --docs-dir sims/docs --out-dir sims/results/docs_pdf`

This keeps generated human-readable artifacts in `sims/results/`.

## Canonical mut_var entrypoints
- Inference: `mutvar infer <observed.tsv> -o <infer.tsv>`
- Curve fit: `mutvar curve <infer.tsv> --fit-only -o <curve.tsv>`

No changes are made to `src/`, `tests/`, packaging, or dependencies.

## Required observed schema
Observed TSV must include these exact columns:
1. `effect_allele_frequency`
2. `beta`
3. `standard_error`

Additional columns are allowed but optional.

## Core simulation controls
The simulator must expose and record:
- `frequency.demography_mode` (currently `equilibrium` implemented)
- `effect.selection_mode` in `{hd, 1d}`
- `ascertainment.ascertainment_statistic` with primary default `threshold_on_maf` and optional alternative modes `{threshold_on_v_true, threshold_on_v_hat}`
- `ascertainment.maf_min` for MAF-threshold runs
- `ascertainment.v_s_cutoff` and `ascertainment.p_value_threshold` for variance-threshold runs
- fixed `n_ascertained` target per run
- `truth_reference_n` for distribution-evaluation truth sampling (should be larger than `n_ascertained`)

Runtime caution: variance-threshold scenarios are intentionally retained for comparison checks but can be extremely sensitive to `v_s_cutoff` and may run for a long time if configured too strictly. For routine workloads, `threshold_on_maf` is the operational default in this framework.

The observed `beta` column is on the selection-scaled effect axis (`beta_s`), and metadata must record the canonical scaled coefficient definition `S_ud = 2Ne*s_ud`.

## Domain and quality constraints
For every row in observed TSV:
- `effect_allele_frequency` is finite and in `[0, 1]`
- `beta` is finite
- `standard_error` is finite and strictly `> 0`
- No nulls in required columns
- File is tab-separated with a header row

## Simulator output artifacts per run
Given `run_id`, simulator writes:
- `sims/results/<run_id>.observed.tsv` (mut_var input)
- `sims/results/<run_id>.truth.tsv` (latent variables and generating params; may use an independent larger truth-reference sample)
- `sims/results/<run_id>.meta.tsv` (run metadata and summary stats)
- `sims/results/<run_id>.manifest.json` (reproducibility metadata)

Downstream mut_var outputs:
- `sims/results/<run_id>.infer.tsv`
- `sims/results/<run_id>.curve.tsv`

## Recovery objective
Primary study question:
- How well mut_var's inferred mixture captures the underlying DFE-generated effect-size distribution.

Evaluation compares:
- Truth-side unconditional effect distribution summaries
- mut_var inferred mixture summaries reconstructed from `infer.tsv` (`mu0`, `var0`, and mixture weights)
- Distribution distances and tail-fidelity metrics on $|\beta_s|$

AF-conditioned diagnostics may be computed as optional secondary outputs, but they are not the primary criterion in this workflow.

## Reproducibility requirements
Each run must be reproducible from:
- random seed
- scenario configuration
- exact commands executed
- artifact paths and basic checks (row counts and key summary values)
