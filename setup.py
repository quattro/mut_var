"""
setup.py — builds the numerics _core Cython extension for mut-var.

Run:
    pip install -e .            # editable install, builds extension in-place
    python setup.py build_ext --inplace  # explicit in-place build
"""

import numpy as np

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="mut_var.numerics._core",
        sources=["src/mut_var/numerics/_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2", "-ffast-math"],
        language="c",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=False,
    ),
)
