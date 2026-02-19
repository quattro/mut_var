# Public API

## Canonical Workflow Entrypoints

Import from package root:

```python
from mut_var import (
    InferenceArrays,
    InferenceConfig,
    RESULTS,
    Solution,
    fit_baseline,
    fit_curve,
    fit_refit_grid,
    run_curve_workflow,
    run_inference_pipeline,
    run_profiled_inference_pipeline,
)
```

Supported workflow APIs:

- `run_inference_pipeline`: infer mixture weights across MAF thresholds.
- `run_curve_workflow`: fit-only or fit+plot curve workflow.
- `run_profiled_inference_pipeline`: compile/steady-state profiling helper.

Supported contract types:

- `RESULTS`
- `Solution`
- `InferenceArrays`
- `InferenceConfig`

## Deprecated Internal Patterns

These internal call patterns are deprecated and not part of the supported API contract:

- Importing objectives/optimizers from `mut_var.cli`
- Calling legacy script internals in `fit.curve.new.py` directly
- Reaching into adapter internals when canonical workflow entrypoints are available

Use package-root exports and workflow modules documented above for forward compatibility.
