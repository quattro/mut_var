# Python API

## Toplevel APIs

::: mut_var.run_inference_pipeline

-----

::: mut_var.run_curve_pipeline
    options:
        show_bases: true

-----
## Numerics APIs

Numerics implementations are available under `mut_var.numerics`:

::: mut_var.numerics.fit_baseline

-----

::: mut_var.numerics.fit_curve

-----

::: mut_var.numerics.fit_refit_grid

-----

Numerics-level status is reported through:

- `mut_var.contracts.Solution`
- `mut_var.contracts.RESULTS`

Use `Solution.result` as the canonical success/failure signal.

## Configuration Classes

::: mut_var.infer.InferenceConfig
    options:
        members:
            - __init__
            - to_baseline_config
            - to_refit_config

::: mut_var.numerics.baseline.BaselineConfig
    options:
        members:
            - __init__

::: mut_var.numerics.refit.RefitConfig
    options:
        members:
            - __init__

## Example

```python
import polars as pl

from mut_var import run_inference_pipeline

df = pl.read_csv("data/bmi_exwas.tsv.gz", separator="\t")
result_df = run_inference_pipeline(df)
print(result_df.head())
```
