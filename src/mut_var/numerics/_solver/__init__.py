# pattern: Imperative Shell
from mut_var.numerics._solver._common import default_verbose, map_optimistix_result
from mut_var.numerics._solver._gradient import RiemannianGradientDescent, RiemannianSteepestDescent
from mut_var.numerics._solver._truncated_cg import TruncatedCG
from mut_var.numerics._solver._trust_region import RiemannianHessianBuilder, RiemannianTrustRegion

__all__ = [
    "RiemannianGradientDescent",
    "RiemannianHessianBuilder",
    "RiemannianSteepestDescent",
    "RiemannianTrustRegion",
    "TruncatedCG",
    "default_verbose",
    "map_optimistix_result",
]
