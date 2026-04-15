# Python API

## Workflow APIs

::: mut_var.run_inference_pipeline

-----

::: mut_var.run_curve_pipeline
    options:
        show_bases: true

-----

::: mut_var.run_simulation_pipeline

-----

## Public Configs

::: mut_var.InferenceConfig

-----

::: mut_var.SimulationConfig

-----

## Numerics APIs

Numerics implementations are available under `mut_var.numerics`. These are lower-level,
array-oriented entrypoints intended for direct experimentation and testing rather than the
primary CLI-facing workflow.

Shared fit-state preparation:

::: mut_var.numerics.prepare_fit_state

-----

::: mut_var.numerics.fit_baseline

-----

::: mut_var.numerics.fit_refit_step

-----

::: mut_var.numerics.fit_curve_model

-----

::: mut_var.numerics.evaluate_curve_fit

-----

Simulation numerics:

::: mut_var.numerics.simulate_mixture_data

-----

Numerics-level status is reported through `mut_var.types`:

- `mut_var.types.Solution`
- `mut_var.types.RESULTS`

Use `Solution.result` as the canonical success/failure signal.

## Pipeline Types

::: mut_var.pipelines.InferenceArrays

-----

::: mut_var.SimulationArtifacts

## Example

```python
from mut_var import InferenceConfig, run_inference_pipeline

result_df = run_inference_pipeline(
    "data/bmi_exwas.tsv.gz",
    config=InferenceConfig(num_clusters=30),
)
print(result_df.head())
```

Curve fitting returns a method-neutral parameter dataframe:

```python
from mut_var import run_curve_pipeline

fit_df = run_curve_pipeline(
    "data/curve-output.tsv",
    generate_plots=False,
    method="sigmoid",
)
print(fit_df.head())
```
