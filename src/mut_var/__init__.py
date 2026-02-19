# SPDX-FileCopyrightText: 2025-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .contracts import RESULTS, Solution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = ["RESULTS", "Solution", "__version__"]
