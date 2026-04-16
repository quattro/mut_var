# Pi-Only Refactor Implementation Plan — Phase 4

**Goal:** Update `cli.py` and `__init__.py` to use the new import paths from `pipelines/`; remove four solver-internal CLI flags from `mutvar infer`; update `AGENTS.md` contracts to reflect the final module layout.

**Architecture:** Pure wiring and documentation pass. No new logic. All three pipeline functions move from flat-module imports to `mut_var.pipelines`. Four argparse flags (`--step-size`/`-r`, `--penalty`, `--seed`/`-s`, `--maf-threshold`/`-t`) removed from `mutvar infer`. `run_infer_pipeline` in `cli.py` hands `args.sumstats` directly to `run_inference_pipeline` instead of pre-loading a DataFrame.

**Tech Stack:** Python stdlib (`argparse`) — no external dep changes.

**Scope:** Phase 4 of 4 from design plan.

**Codebase verified:** 2026-04-15

---

## Acceptance Criteria Coverage

This phase implements and tests:

### pi-only-refactor.AC3: Public API surface and CLI updated correctly

- **pi-only-refactor.AC3.1 Success:** `run_inference_pipeline(path, ...)` (path string) returns a long-format `pl.DataFrame` with columns `mu0`, `var0`, `maf`, `name`, `value` for a valid TSV path.
- **pi-only-refactor.AC3.3 Success:** `mutvar infer <path> -k 10 -m 50 -f 1e-6 --lowest 1e-5 --highest 1e-2 --num-breaks 5` exits with code 0.
- **pi-only-refactor.AC3.4 Success:** `mutvar curve <path>` and `mutvar simulate --output-prefix foo` exit with code 0 using valid inputs.
- **pi-only-refactor.AC3.5 Failure:** `mutvar infer <path> --step-size 0.1` exits with code 2 (unrecognized argument).
- **pi-only-refactor.AC3.6 Failure:** `mutvar infer <path> --seed 42` exits with code 2 (unrecognized argument).
- **pi-only-refactor.AC3.7 Failure:** `mutvar infer missing.tsv` exits with code 2 (file not found).
- **pi-only-refactor.AC3.8 Failure:** `run_inference_pipeline("missing.tsv", ...)` raises `FileNotFoundError`.

### pi-only-refactor.AC4: Quality gate passes end-to-end

- **pi-only-refactor.AC4.1 Success:** `ruff check src/mut_var tests` exits with code 0.
- **pi-only-refactor.AC4.2 Success:** `mypy src/mut_var tests` exits with code 0 (no type errors).
- **pi-only-refactor.AC4.3 Success:** `pytest -p no:capture` exits with code 0 (all tests pass).

---

## Design Adjustments

1. **`test_cli_contracts.py` fix must precede Phase 2 Task 5.** Phase 2 Task 5 deletes `baseline.py`, but Phase 2 Task 4 does not fix `test_cli_contracts.py`, which imports `import mut_var.numerics.baseline as baseline_module`. After deletion, pytest collection of `test_cli_contracts.py` fails with `ModuleNotFoundError`, causing Phase 2's quality gate to fail. When executing sequentially, apply the `test_cli_contracts.py` changes from Task 1 of this phase **during Phase 2 Task 4** (before running Phase 2 Task 5's quality gate). They are written here in Phase 4 for coherence with the CLI cleanup scope but must be pulled forward in execution order.

2. **`--num_breaks` renamed to `--num-breaks`.** The AC (pi-only-refactor.AC3.3) explicitly uses `--num-breaks`. All other infer flags already use hyphens (e.g., `--num-clusters`, `--max-iter`). The rename is trivial: argparse converts `--num-breaks` to `dest='num_breaks'`, so no function-body changes are needed. The test `test_cli_contracts.py` (line 109) must also be updated from `--num_breaks` to `--num-breaks`.

3. **`run_infer_pipeline` drops explicit loading log lines.** The current handler logs `"infer: loading data from '%s'"` before calling `read_sumstats`, and `"infer: data loaded (%d rows)"` after. After Phase 4, loading is internal to `run_inference_pipeline`. These two log calls are removed. The corresponding assertion in `test_cli_infer_success_writes_dataframe` is updated from `"infer: loading data"` to `"infer: starting inference pipeline"`.

---

## Codebase Verification Findings

- ✓ `cli.py` imports: `from mut_var.curve import run_curve_pipeline`, `from mut_var.infer import InferenceConfig, run_inference_pipeline`, `from mut_var.io import read_sumstats`, `from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig` — all four must change.
- ✓ `cli.py` flags to remove: `-t`/`--maf-threshold` (lines 88–94), `-r`/`--step-size` (lines 110–115), `-s`/`--seed` (line 116, infer subcommand only), `--penalty` (lines 124–129).
- ✓ `cli.py` flag to rename: `--num_breaks` (line 134) → `--num-breaks`; `args.num_breaks` dest is unchanged (argparse converts hyphen to underscore automatically).
- ✓ `cli.py::run_infer_pipeline` uses `args.seed`, `args.step_size`, `args.penalty` — must be removed from `InferenceConfig` construction and the function call.
- ✓ `cli.py::run_infer_pipeline` calls `read_sumstats(args.sumstats)` then passes `df` to `run_inference_pipeline` — after Phase 4, passes `args.sumstats` directly.
- ✓ `__init__.py` imports from `.curve`, `.infer`, `.simulate` — all three flat modules are deleted in Phase 3; must change to `.pipelines` and `.types`.
- ✓ `test_cli_contracts.py` imports `mut_var.numerics.baseline as baseline_module` and patches `baseline_module.fit_baseline` — must change to `mixture_fit_module` / `prepare_fit_state`.
- ✓ `test_cli_contracts.py` uses `--num_breaks` (line 109) — must change to `--num-breaks`.
- ✓ `test_cli_infer_success_writes_dataframe` checks `"infer: loading data" in err` (line 147) — must change to `"infer: starting inference pipeline"`.
- ✓ `AGENTS.md` references `mut_var.contracts.*` and `mut_var.numerics.InferenceArrays/InferenceConfig` — must update to `mut_var.types.*` and `mut_var.io.InferenceArrays`.
- ✓ `numerics/AGENTS.md` references `mut_var.contracts`, `baseline.py`, `refit.py` — must update to `mut_var.types`, `mixture_fit.py`.
- + `mutvar simulate` subcommand also has `--seed` (line 200) — this flag stays; only remove `--seed` from `mutvar infer`.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Update `test_cli_contracts.py` — fix log assertion after CLI simplification

> **EXECUTION ORDER NOTE:** Changes 1–3 below were already applied in **Phase 2 Task 4** (before `baseline.py` was deleted). Only change 4 remains for this task, since it depends on Phase 4's removal of the `"infer: loading data"` log line from `run_infer_pipeline`.

**Verifies:** prerequisite for AC4.3

**Files:**
- Modify: `tests/test_cli_contracts.py`

**Implementation:**

**Change 4 only** (changes 1–3 were applied in Phase 2 Task 4):

**`test_cli_infer_success_writes_dataframe`** — update log assertion to match the simplified `run_infer_pipeline` which no longer calls `read_sumstats` and logs `"infer: loading data"`:
```python
# Before (line 147):
    assert "infer: loading data" in err
# After:
    assert "infer: starting inference pipeline" in err
```

**Verification:**

```bash
pytest tests/test_cli_contracts.py -p no:capture
```
Expected: all tests pass.

**Commit:** `refactor: update test_cli_contracts.py log assertion for simplified run_infer_pipeline`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `src/mut_var/cli.py` — remove four flags, update imports, simplify `run_infer_pipeline`

**Verifies:** pi-only-refactor.AC3.3, AC3.5, AC3.6, AC3.7

**Files:**
- Modify: `src/mut_var/cli.py`

**Implementation:**

**Step A — Update imports** (lines 13–17). Replace:
```python
from mut_var.curve import run_curve_pipeline
from mut_var.infer import InferenceConfig, run_inference_pipeline
from mut_var.io import read_sumstats
from mut_var.numerics import SimulationNumericsConfig
from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig
```
with:
```python
from mut_var.numerics import SimulationNumericsConfig
from mut_var.pipelines import run_curve_pipeline, run_inference_pipeline, run_simulation_pipeline
from mut_var.types import InferenceConfig, SimulationPipelineConfig
```

**Step B — Remove four argument definitions from `_build_infer_subcommand`.**

Remove the `-t`/`--maf-threshold` block:
```python
    model_group.add_argument(
        "-t",
        "--maf-threshold",
        type=float,
        default=0.01,
        help="Reserved MAF threshold parameter for model controls.",
    )
```

Remove the `-r`/`--step-size` block:
```python
    model_group.add_argument(
        "-r",
        "--step-size",
        type=float,
        default=0.01,
        help="Optimization step size.",
    )
```

Remove the `-s`/`--seed` line:
```python
    model_group.add_argument("-s", "--seed", type=int, default=0, help="PRNG seed.")
```

Remove the `--penalty` block:
```python
    model_group.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="Penalty weight for objective regularization.",
    )
```

**Step C — Rename `--num_breaks` to `--num-breaks`** in the grid group:
```python
# Before:
    grid_group.add_argument("--num_breaks", type=int, default=10, help="Number of MAF grid breakpoints.")

# After:
    grid_group.add_argument("--num-breaks", type=int, default=10, help="Number of MAF grid breakpoints.")
```

No other change needed — `args.num_breaks` still works because argparse converts `--num-breaks` to dest `num_breaks`.

**Step D — Simplify `run_infer_pipeline`** (currently lines 308–355). Replace the entire function body with:

```python
def run_infer_pipeline(args: ap.Namespace, log: logging.Logger) -> int:
    r"""Run the CLI inference workflow.

    **Arguments:**

    - `args`: Parsed CLI arguments for `infer`.
    - `log`: Logger used for diagnostics.

    **Returns:**

    - Exit code (`0` success, `2` usage/input errors, `1` runtime failures).
    """
    try:
        log.info("infer: starting inference pipeline")
        result_df = run_inference_pipeline(
            args.sumstats,
            af_col=args.af_col,
            beta_col=args.beta_col,
            se_col=args.se_col,
            lowest=args.lowest,
            highest=args.highest,
            num_breaks=args.num_breaks,
            config=InferenceConfig(
                num_clusters=args.num_clusters,
                max_iter=args.max_iter,
                filter_threshold=args.filter,
            ),
            log=log,
        )
        log.info("infer: inference pipeline completed")
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    log.info("infer: writing output to '%s'", _output_target(args.output))
    result_df.write_csv(args.output, separator="\t")
    log.info("infer: finished writing output")
    return 0
```

Key changes vs current:
- Removed `df = read_sumstats(args.sumstats)` and the two log lines around it.
- Changed first argument from `df` to `args.sumstats`.
- Removed `seed=args.seed` from the call.
- Removed `step_size=args.step_size` and `penalty=args.penalty` from `InferenceConfig`.

**Verification:**

```bash
python -c "import mut_var.cli; print('import OK')"
```

```bash
python -m mut_var.cli infer --help | grep -E "step-size|penalty|seed|maf-threshold"
```
Expected: no output (all four flags removed).

```bash
python -m mut_var.cli infer --help | grep "num-breaks"
```
Expected: prints the `--num-breaks` help line.

**Commit:** `refactor: cli.py — remove step-size/penalty/seed/maf-threshold flags; update imports`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Update `src/mut_var/__init__.py`

**Verifies:** prerequisite for AC3.1

**Files:**
- Modify: `src/mut_var/__init__.py`

**Implementation:**

Replace the three import lines (lines 8–10) with imports from the new module locations:

```python
# Before:
from .curve import run_curve_pipeline
from .infer import run_inference_pipeline
from .simulate import run_simulation_pipeline, SimulationArtifacts, SimulationPipelineConfig

# After:
from .pipelines import run_curve_pipeline, run_inference_pipeline, run_simulation_pipeline, SimulationArtifacts
from .types import SimulationPipelineConfig
```

The `__all__` list and the version block stay unchanged.

**Verification:**

```bash
python -c "
from mut_var import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline
from mut_var import SimulationPipelineConfig, SimulationArtifacts
print('__init__.py imports OK')
"
```
Expected: prints `__init__.py imports OK`.

**Commit:** `refactor: update __init__.py imports from pipelines/ and types`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

---

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: Update `AGENTS.md` and `src/mut_var/numerics/AGENTS.md`

**Verifies:** pi-only-refactor documentation completeness (prerequisite for AC4.3 — mypy clean)

**Files:**
- Modify: `AGENTS.md`
- Modify: `src/mut_var/numerics/AGENTS.md`

**Implementation:**

**`AGENTS.md`** — make the following targeted replacements:

1. **Contracts → Exposes** section — update the contract types list:
   - `mut_var.contracts.RESULTS` → `mut_var.types.RESULTS`
   - `mut_var.contracts.Solution` → `mut_var.types.Solution`
   - `mut_var.numerics.InferenceArrays` → `mut_var.io.InferenceArrays`
   - `mut_var.numerics.InferenceConfig` → `mut_var.types.InferenceConfig`
   - `mut_var.SimulationPipelineConfig` stays (still exported from package root)
   - `mut_var.SimulationArtifacts` stays (still exported from package root)

2. **Contracts → Exposes** — update pipeline API signature:
   - `run_inference_pipeline` now accepts `path: str` (was `df: pl.DataFrame`)

3. **Contracts → Guarantees** — add:
   - `run_inference_pipeline` accepts a file path string and calls `io.load_inference_arrays` internally; the intermediate DataFrame is not retained alongside JAX arrays.

4. **Dependencies** — update:
   - Remove `mut_var.adapters` from any mention
   - Add note that pipeline APIs are in `mut_var.pipelines`

5. **Key Decisions** — update or add:
   - `contracts.py` renamed to `types.py`; `adapters/` deleted; `io.py` now owns all ingress
   - Pipeline orchestration lives in `src/mut_var/pipelines/`

6. **Commands** section — update `mutvar infer` example, removing `--step-size`/`--seed`/`--penalty`/`--maf-threshold`, and changing `--num_breaks` to `--num-breaks`.

**`src/mut_var/numerics/AGENTS.md`** — make the following targeted replacements:

1. **Contracts → Exposes** — replace baseline/refit surface with mixture_fit surface:
   ```
   # Before:
   - `fit_baseline(beta_hat, s2, key, config, verbose=False) -> Solution`
   - `fit_refit_grid(beta_hat, s2, maf_masks, init, config, verbose=False) -> Solution`

   # After:
   - `prepare_fit_state(beta_hat, s2, config) -> Solution`
   - `fit_baseline(state: FitState, config: InferenceConfig, verbose=False) -> Solution`
   - `fit_refit_step(L_sub: Array, prev_params: Params, config: InferenceConfig, verbose=False) -> Solution`
   ```

2. **Dependencies → Uses** — change `mut_var.contracts` to `mut_var.types`.

3. **Key Decisions** — update:
   - `MutVarSolver` centralizes result mapping from `optx.RESULTS` to `mut_var.types.RESULTS` (was `mut_var.contracts.RESULTS`)
   - Add: `baseline.py` and `refit.py` merged into `mixture_fit.py`; likelihood matrix pre-computed once in `prepare_fit_state`

4. **Key Files** — replace `baseline.py` and `refit.py` entries with:
   - `src/mut_var/numerics/mixture_fit.py` — pi-only solver: `prepare_fit_state`, `fit_baseline`, `fit_refit_step`, `FitState`, `Params`.

5. **Gotchas** — remove references to `baseline_objective_lse`; update any references to old module names.

**Commit:** `docs: update AGENTS.md and numerics/AGENTS.md for new module layout`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Run full quality gate and verify AC3 CLI contracts

**Verifies:** pi-only-refactor.AC3.1, AC3.3, AC3.4, AC3.5, AC3.6, AC3.7, AC3.8, AC4.1, AC4.2, AC4.3

**Files:**
- No code changes; verification only.

**Verification — module layout (AC2.3 regression check):**
```bash
python -c "
from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline
print('AC2.3 OK: pipelines imports work')
"
```

**Verification — CLI flags removed (AC3.5, AC3.6):**

```bash
# Setup: copy a valid TSV to /tmp for quick CLI tests
cp tests/fixtures/sumstats_valid.tsv /tmp/sumstats_valid.tsv

python -c "
import sys
import mut_var.cli as cli
code = cli.run_cli(['infer', '/tmp/sumstats_valid.tsv', '--step-size', '0.1'])
assert code == 2, f'Expected exit code 2, got {code}'
print('AC3.5 OK: --step-size returns code 2')
"

python -c "
import sys
import mut_var.cli as cli
code = cli.run_cli(['infer', '/tmp/sumstats_valid.tsv', '--seed', '42'])
assert code == 2, f'Expected exit code 2, got {code}'
print('AC3.6 OK: --seed returns code 2')
"
```

**Verification — missing file (AC3.7, AC3.8):**

```bash
python -c "
import mut_var.cli as cli
code = cli.run_cli(['infer', 'missing.tsv'])
assert code == 2, f'Expected exit code 2, got {code}'
print('AC3.7 OK: missing file returns code 2')
"

python -c "
from mut_var.pipelines import run_inference_pipeline
try:
    run_inference_pipeline('missing.tsv')
    print('FAIL: expected FileNotFoundError')
except FileNotFoundError:
    print('AC3.8 OK: FileNotFoundError raised')
"
```

**Verification — help text shows --num-breaks (AC3.3):**
```bash
python -c "
import mut_var.cli as cli, sys
from io import StringIO
stdout = StringIO()
import sys
old_stdout = sys.stdout
sys.stdout = stdout
try:
    cli.run_cli(['infer', '--help'])
except SystemExit:
    pass
sys.stdout = old_stdout
help_text = stdout.getvalue()
assert '--num-breaks' in help_text, 'FAIL: --num-breaks not in help'
assert '--step-size' not in help_text, 'FAIL: --step-size still in help'
assert '--penalty' not in help_text, 'FAIL: --penalty still in help'
assert '--maf-threshold' not in help_text, 'FAIL: --maf-threshold still in help'
print('AC3.3 OK: help text correct')
"
```

**Full quality gate:**

```bash
ruff check src/mut_var tests
```
Expected: exits code 0.

```bash
mypy src/mut_var tests
```
Expected: exits code 0.

```bash
pytest -p no:capture
```
Expected: all tests pass.

**Commit:** `refactor: pi-only-refactor phase 4 complete — CLI and public API cleanup`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->
