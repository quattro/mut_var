# CLI

## Commands

- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [options]`
- `mutvar simulate --output-prefix <prefix> [options]`

## `infer` Options

Input and output:

- `sumstats` input path
- `-o, --output` output TSV path (defaults to stdout)

File output is written atomically after successful processing. The output must
not refer to the input file, including through a symlink or hardlink. These
rules also apply to `curve` output.

Column overrides:

- `--af-col` (default: `effect_allele_frequency`)
- `--beta-col` (default: `beta`)
- `--se-col` (default: `standard_error`)

Model and optimizer controls:

- `-k, --num-clusters`
- `-m, --max-iter`
- `-f, --filter`
- `--atol` and `--rtol` (each defaults to `1e-6`; finite, nonnegative solver tolerances)
- `--constrain-spike` (opt into spike constraints during refitting)

Cluster counts must be integers at least 2, iteration limits positive integers,
and the filtering threshold finite and within `[0, 1]`.

MAF grid controls:

- `--lowest` (defaults to the minimum positive observed MAF)
- `--highest`
- `--num-breaks`

Logging:

- `-v, --verbose` enables debug logging.

## `curve` Options

- `data` input TSV from `mutvar infer`
- `-o, --output` output TSV path (defaults to stdout)
- `--method`: `sigmoid` (default), `isotonic`, `mono_spline`, `invlog_linear`,
  `invlog_logit`, or `invlog_sigmoid`
- `--fit-only` skips PNG generation

## `simulate` Options

Output:

- `--output-prefix` required filename prefix
- `--output-dir` output directory

Core:

- `--n-rows`
- `--seed` (nonnegative integer)

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
