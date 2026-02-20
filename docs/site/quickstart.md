# Quickstart

## Run Inference

```console
mutvar infer data/bmi_exwas.tsv.gz -o mutvar-output.tsv
```

The inference input must contain:

- `effect_allele_frequency` (or `--af-col`)
- `beta` (or `--beta-col`)
- `standard_error` (or `--se-col`)

## Run Curve Fitting

Fit coefficients only:

```console
mutvar curve mutvar-output.tsv --fit-only -o curve-coefficients.tsv
```

Fit plus PNG generation:

```console
mutvar curve mutvar-output.tsv -o curve-coefficients.tsv
```
