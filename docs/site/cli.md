# CLI

## Commands

- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [options]`

## `infer` Options

Input and output:

- `sumstats` input path
- `-o, --output` output TSV path (defaults to stdout)

Column overrides:

- `--af-col` (default: `effect_allele_frequency`)
- `--beta-col` (default: `beta`)
- `--se-col` (default: `standard_error`)

Model and optimizer controls:

- `-k, --num-clusters`
- `-m, --max-iter`
- `-r, --step-size`
- `-s, --seed`
- `-f, --filter`
- `--penalty`

MAF grid controls:

- `--lowest`
- `--highest`
- `--num_breaks`

Logging:

- `-v, --verbose` enables debug logging.

## `curve` Options

- `data` input TSV from `mutvar infer`
- `-o, --output` output TSV path (defaults to stdout)
- `--fit-only` skips PNG generation

## Exit Codes

- `0`: success
- `2`: usage/input errors (`ValueError`, `FileNotFoundError`)
- `1`: runtime failures (`RuntimeError`)
