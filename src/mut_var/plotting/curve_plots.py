from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def render_curve_plot(
    maf,
    value,
    maf_space,
    fitted_values,
    title: str,
    output_path: str | Path,
) -> Path:
    r"""Render and save a semilog fitted-curve plot, returning the written path."""
    path = Path(output_path)

    plt.figure(figsize=(6, 4))
    plt.semilogx(maf, value, "o", label="data")
    plt.semilogx(maf_space, fitted_values, "-", label="fit")
    plt.xlabel("MAF")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()

    return path
