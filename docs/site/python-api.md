# Python API

## Package-Root Pipeline APIs

```python
from mut_var import run_curve_pipeline, run_inference_pipeline
```

Supported orchestration entrypoints:

- `run_inference_pipeline(df, ...) -> polars.DataFrame`
- `run_curve_pipeline(input_path, generate_plots=...) -> polars.DataFrame`

## Numerics APIs

Numerics implementations are available under `mut_var.numerics`:

- `mut_var.numerics.fit_baseline`
- `mut_var.numerics.fit_curve`
- `mut_var.numerics.fit_refit_grid`

Numerics-level status is reported through:

- `mut_var.contracts.Solution`
- `mut_var.contracts.RESULTS`

Use `Solution.result` as the canonical success/failure signal.

## Example

```python
import polars as pl

from mut_var import run_inference_pipeline

df = pl.read_csv("data/bmi_exwas.tsv.gz", separator="\t")
result_df = run_inference_pipeline(df)
print(result_df.head())
```
