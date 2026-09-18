# pattern: Imperative Shell
"""Check an installed distribution, including its Cython/BLAS extension and CLI."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

import numpy as np

import mut_var

from mut_var.numerics import evaluate_curve_fit, fit_curve_model
from mut_var.numerics.mixsqp import mix_sqp
from mut_var.types import RESULTS


def main() -> None:
    weights, info = mix_sqp(np.tile([1.0, 2.0], (10, 1)))
    assert info["converged"]
    np.testing.assert_allclose(weights, [0.0, 1.0], atol=1e-6)
    maf = np.geomspace(1e-6, 0.01, 10)
    values = 0.2 + 0.1 / -np.log(maf)
    fit = fit_curve_model(maf, values, method="invlog_linear")
    assert fit.result is RESULTS.successful
    np.testing.assert_allclose(evaluate_curve_fit(fit.value, maf), values)
    executable = Path(sys.executable).with_name("mutvar")
    subprocess.run([str(executable), "--help"], check=True, stdout=subprocess.DEVNULL)
    print(f"Validated mut-var {mut_var.__version__} from {mut_var.__file__}")


if __name__ == "__main__":
    main()
