# Public API

## Canonical Pipeline Entrypoints

Import from package root:

```python
from mut_var import (
    run_curve_pipeline,
    run_inference_pipeline,
)
```

Supported pipeline APIs:

- `run_inference_pipeline`: validated dataframe ingress -> long-form coefficients `polars.DataFrame`.
- `run_curve_pipeline`: fit-only or fit+plot curve pipeline -> coefficient `polars.DataFrame`.

Supported numerics APIs:

- `mut_var.numerics.run_inference_pipeline`: numerics pipeline with `Solution` status/result payload.
- `mut_var.numerics.run_profiled_inference_pipeline`: compile/steady-state profiling helper.
- `mut_var.numerics.fit_baseline`
- `mut_var.numerics.fit_curve`
- `mut_var.numerics.fit_refit_grid`

Supported contract types:

- `mut_var.contracts.RESULTS`
- `mut_var.contracts.Solution`
- `mut_var.numerics.pipeline.InferenceArrays`
- `mut_var.numerics.pipeline.InferenceConfig`

## Deprecated Internal Patterns

These internal call patterns are deprecated and not part of the supported API contract:

- Importing objectives/optimizers from `mut_var.cli`
- Reaching into adapter internals when canonical pipeline entrypoints are available

Use package-root exports and pipeline modules documented above for forward compatibility.
