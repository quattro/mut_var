# pattern: Imperative Shell
from mut_var.numerics._solver._common import default_verbose, map_optimistix_result
from mut_var.numerics._solver._gradient import (
    MutVarSolver,
    RiemannianGradientDescent,
    RiemannianGradientDescentState,
)
from mut_var.numerics._solver._truncated_cg import TruncatedCG
from mut_var.numerics._solver._trust_region import RiemannianTrustRegion

__all__ = [
    "MutVarSolver",
    "RiemannianGradientDescent",
    "RiemannianGradientDescentState",
    "RiemannianTrustRegion",
    "TruncatedCG",
    "default_verbose",
    "map_optimistix_result",
]
