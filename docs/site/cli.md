# CLI

## Commands

- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [options]`
- `mutvar simulate --output-prefix <prefix> [options]`

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
- `-f, --filter`
- `--tol`

MAF grid controls:

- `--lowest`
- `--highest`
- `--num-breaks`

Logging:

- `-v, --verbose` enables debug logging.

## `curve` Options

- `data` input TSV from `mutvar infer`
- `-o, --output` output TSV path (defaults to stdout)
- `--method` curve fitting method (`sigmoid` default, `isotonic` alternative)
- `--fit-only` skips PNG generation

## `simulate` Options

Output:

- `--output-prefix` required filename prefix
- `--output-dir` output directory

Core:

- `--n-rows`
- `--seed`

Mixture:

- `--weights`
- `--log-var-scales`

Link and sampling:

- `--variance-link`
- `--theta`
- `--link-eps`
- `--link-shift`
- `--af-model`
- `--af-clip-min`
- `--af-uniform-low`
- `--af-uniform-high`
- `--af-beta-a`
- `--af-beta-b`
- `--se-model`
- `--se-constant`
- `--sample-size`
- `--se-scale`

## Exit Codes

- `0`: success
- `2`: usage/input errors (`ValueError`, `FileNotFoundError`)
- `1`: runtime failures (`RuntimeError`)
