from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(maf, value, "o", label="data")
    ax.semilogx(maf_space, fitted_values, "-", label="fit")
    ax.set_xlabel("MAF")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3e"))
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path
