# Quickstart

## Run Inference

```console
mutvar infer data/bmi_exwas.tsv.gz -o mutvar-output.tsv
```

The inference input path must point to a tab-separated file containing:

- `effect_allele_frequency` (or `--af-col`)
- `beta` (or `--beta-col`)
- `standard_error` (or `--se-col`)

Override column names when needed:

```console
mutvar infer data/custom.tsv \
  --af-col eaf \
  --beta-col effect \
  --se-col se \
  -o mutvar-output.tsv
```

## Run Curve Fitting

Fit coefficients only:

```console
mutvar curve mutvar-output.tsv --fit-only -o curve-coefficients.tsv
```

Fit plus PNG generation:

```console
mutvar curve mutvar-output.tsv -o curve-coefficients.tsv
```

## Run Simulation

```console
mutvar simulate \
  --output-dir out \
  --output-prefix demo
```

This writes:

- `out/demo.truth.tsv`
- `out/demo.observed.tsv`
- `out/demo.meta.tsv`
