# Pi-Only Refactor Human Test Plan

## Preconditions

- Editable install is active and `mutvar` is on `PATH`.
- `tests/fixtures/sumstats_valid.tsv` is available.
- `/tmp` or another writable output directory is available.

## Phase Checks

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | `python -c "from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig"` | Imports succeed with no error |
| 2 | `python -c "from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays"` | Imports succeed with no error |
| 3 | `python -c "from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline"` | Imports succeed with no error |
| 4 | `python -c "import mut_var.contracts"` and `python -c "import mut_var.numerics.baseline"` | Both raise `ModuleNotFoundError` |
| 5 | `python - <<'PY'` smoke JIT calls for `prepare_fit_state`, `fit_baseline`, and `fit_refit_step` on valid arrays | No tracing errors; results are `successful` or `max_steps_reached` as appropriate |
| 6 | `mutvar infer /tmp/sumstats_valid.tsv -k 10 -m 50 -f 1e-6 --lowest 1e-5 --highest 1e-2 --num-breaks 5 -o /tmp/out.tsv` | Exit `0`; output TSV exists and has long-format columns |
| 7 | `mutvar curve /tmp/out.tsv --fit-only` and `mutvar simulate --output-prefix /tmp/sim` | Both exit `0`; curve writes coefficients; simulate writes `.truth.tsv`, `.observed.tsv`, `.meta.tsv` |
| 8 | `mutvar infer /tmp/missing.tsv` and `python -c "from mut_var.pipelines import run_inference_pipeline; run_inference_pipeline('missing.tsv', lowest=1e-3, highest=5e-3, num_breaks=2)"` | CLI exits `2`; Python raises `FileNotFoundError` |

## End-To-End Scenarios

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Infer-to-curve-to-simulate smoke | Run supported-flag `mutvar infer`, feed output into `mutvar curve --fit-only`, then run `mutvar simulate` with an output prefix | All commands exit `0` and produce the expected artifacts |
| Ingress and failure-path smoke | Run missing-path `mutvar infer`, then the Python missing-path pipeline call, then legacy import checks | Proper `2`/`FileNotFoundError`/`ModuleNotFoundError` behavior with no custom exceptions |
| Numerics boundary smoke | Run JIT and eager `prepare_fit_state`, `fit_baseline`, and `fit_refit_step` on a small synthetic dataset | Valid inputs succeed; invalid inputs surface explicit `RESULTS` failures |

## Traceability

| AC ID | Automated Evidence | Human Step |
|------|---------------------|------------|
| AC1.1-AC1.6 | `tests/test_infer_opt.py`; `pytest -p no:capture` | Numerics smoke with valid, invalid, empty, singleton-cluster, max-iter, and non-finite objective cases |
| AC2.1-AC2.8 | `tests/test_types_io_imports.py`; `pytest -p no:capture` | Import smoke for current modules plus deletion checks for legacy modules |
| AC3.1-AC3.8 | `tests/test_infer.py`, `tests/test_cli_contracts.py`; `pytest -p no:capture` | CLI infer/curve/simulate smoke plus missing-path Python and CLI checks |
| AC4.1-AC4.3 | `ruff check src/mut_var tests`, `mypy src/mut_var tests`, `pytest -p no:capture` | Re-run the canonical repo checks at the branch tip |
