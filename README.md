# mut_var

[![PyPI - Version](https://img.shields.io/pypi/v/mut-var.svg)](https://pypi.org/project/mut-var)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mut-var.svg)](https://pypi.org/project/mut-var)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install mut-var
```

## Validation Contract

`mutvar` validates inputs before running any fitting routine.

- Required columns:
  - `effect_allele_frequency` (or `--af-col`)
  - `beta` (or `--beta-col`)
  - `standard_error` (or `--se-col`)
- Domain rules:
  - `effect_allele_frequency` must be within `[0, 1]`
  - `standard_error` must be strictly greater than `0`
- MAF grid rules:
  - `0 < lowest < highest <= 0.5`
  - `num_breaks >= 2`
- Failure contract:
  - Invalid CLI arguments or invalid input data return a deterministic non-zero exit code.
  - Errors are emitted as actionable stderr messages.

## License

`mut-var` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
