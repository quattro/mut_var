# _core.pxd — Cython declarations for _core.pyx
# Import this file in other .pyx modules to call _core functions at C speed.

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t
