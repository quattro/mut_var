#!/usr/bin/env python3
"""Create simulation-only diagnostic plots before inference workflows.

Outputs:
1) DFE over log10(S_ud), using the configured tabulated DFE weights.
2) beta_s_true distributions across MAF cutoff configurations, using truth TSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation diagnostics (DFE and beta distributions).")
    parser.add_argument("--results-dir", default="sims/results")
    parser.add_argument("--plots-dir", default="sims/plots")
    parser.add_argument("--base-config", default="sims/config/dfe_scenarios.json")
    parser.add_argument(
        "--maf-configs",
        nargs="+",
        default=[
            "sims/config/maf_cutoffs/m0p001_n10000.json",
            "sims/config/maf_cutoffs/m0p005_n10000.json",
            "sims/config/maf_cutoffs/m0p01_n10000.json",
            "sims/config/maf_cutoffs/m0p05_n10000.json",
        ],
    )
    parser.add_argument("--n-ascertained", type=int, default=12000)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id_from_config(config_path: Path) -> str:
    payload = _load_json(config_path)
    return str(payload["default"]["run_id"])


def _maf_min_from_config(config_path: Path) -> float:
    payload = _load_json(config_path)
    return float(payload["default"]["ascertainment"]["maf_min"])


def _plot_dfe(base_config_path: Path, plots_dir: Path) -> None:
    payload = _load_json(base_config_path)
    dfe = payload["default"]["dfe"]
    ne = float(payload["default"]["frequency"]["N_e_ancestral"])

    log10_s = np.asarray(dfe["log10_s_grid"], dtype=float)
    weights = np.asarray(dfe["weight_grid"], dtype=float)
    weights = np.maximum(weights, 0.0)
    if np.sum(weights) <= 0.0:
        raise ValueError("DFE weight grid must have positive total mass")
    weights = weights / np.sum(weights)

    log10_S = np.log10(2.0 * ne) + log10_s

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.2,
            "lines.linewidth": 3.0,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    color_main = "#0072B2"
    color_fill = "#56B4E9"

    ax.plot(log10_S, weights, color=color_main)
    ax.fill_between(log10_S, 0.0, weights, color=color_fill, alpha=0.35)
    ax.set_xlabel("log10(S_ud)")
    ax.set_ylabel("DFE probability mass")
    ax.set_title("Configured DFE on the S scale")
    ax.grid(alpha=0.18)
    fig.tight_layout()

    out_pdf = plots_dir / "simulation_diagnostics.dfe_logS.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)


def _read_beta_truth(path: Path) -> np.ndarray:
    frame = pl.read_csv(path, separator="\t")
    return frame.get_column("beta_s_true").to_numpy()


def _plot_beta_by_maf(results_dir: Path, maf_configs: list[Path], plots_dir: Path) -> None:
    palette = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#4E79A7"]

    series: list[tuple[float, np.ndarray, str]] = []
    for idx, config_path in enumerate(maf_configs):
        run_id = _run_id_from_config(config_path)
        maf_min = _maf_min_from_config(config_path)
        truth_path = results_dir / f"{run_id}.truth.tsv"
        if not truth_path.exists():
            raise FileNotFoundError(
                f"Missing truth file for MAF diagnostic: {truth_path}. "
                "Run simulate_dfe for this config first."
            )
        beta = _read_beta_truth(truth_path)
        series.append((maf_min, beta, palette[idx % len(palette)]))

    series = sorted(series, key=lambda x: x[0])

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    bins_log = np.linspace(-5.0, 1.0, 120)
    pooled_log_beta: list[np.ndarray] = []
    ecdf_series: list[tuple[np.ndarray, str, str]] = []
    density_series: list[tuple[np.ndarray, np.ndarray]] = []

    for maf_min, beta, color in series:
        label = f"MAF cutoff = {maf_min:g}"
        beta_pos = np.abs(beta)
        beta_pos = beta_pos[beta_pos > 0.0]

        log_beta = np.log10(np.maximum(beta_pos, 1e-12))
        pooled_log_beta.append(log_beta)
        ecdf_series.append((np.sort(log_beta), label, color))
        log_density, log_edges = np.histogram(log_beta, bins=bins_log, density=True)
        log_mids = 0.5 * (log_edges[:-1] + log_edges[1:])
        density_series.append((log_mids, log_density))
        axes[0].plot(log_mids, log_density, color=color, label=label)

    pooled = np.concatenate(pooled_log_beta)
    full_xmin = float(np.quantile(pooled, 0.001))
    full_xmax = float(np.quantile(pooled, 0.999))
    full_xmin = max(full_xmin, -5.0)
    full_xmax = min(full_xmax, 1.0)
    rhs_xmin = -2.0
    rhs_xmax = full_xmax
    if rhs_xmax <= rhs_xmin + 0.2:
        rhs_xmin = rhs_xmax - 1.5

    ax_rhs, ax_ecdf = axes

    ax_rhs.set_xlim(max(rhs_xmin, full_xmin), rhs_xmax)
    ax_rhs.set_xlabel("log10(|beta_s|)")
    ax_rhs.set_ylabel("Density")
    ax_rhs.grid(alpha=0.18)

    rhs_xmin_eff, rhs_xmax_eff = ax_rhs.get_xlim()

    rhs_peak = 0.0
    for x_vals, y_vals in density_series:
        rhs_mask = (x_vals >= rhs_xmin_eff) & (x_vals <= rhs_xmax_eff)
        if np.any(rhs_mask):
            rhs_peak = max(rhs_peak, float(np.max(y_vals[rhs_mask])))

    if rhs_peak > 0.0:
        ax_rhs.set_ylim(0.0, rhs_peak * 1.08)

    for x_sorted, label, color in ecdf_series:
        y = np.arange(1, x_sorted.size + 1, dtype=float) / float(x_sorted.size)
        ax_ecdf.plot(x_sorted, y, color=color, label=label)
    ax_ecdf.set_xlim(full_xmin, full_xmax)
    ax_ecdf.set_ylim(0.0, 1.0)
    ax_ecdf.set_xlabel("log10(|beta_s|)")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.grid(alpha=0.18)

    handles, labels = ax_rhs.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))

    out_pdf = plots_dir / "simulation_diagnostics.beta_log_by_maf.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    base_config_path = Path(args.base_config)
    maf_configs = [Path(path) for path in args.maf_configs]

    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_dfe(base_config_path=base_config_path, plots_dir=plots_dir)
    _plot_beta_by_maf(results_dir=results_dir, maf_configs=maf_configs, plots_dir=plots_dir)

    print(f"wrote: {plots_dir / 'simulation_diagnostics.dfe_logS.pdf'}")
    print(f"wrote: {plots_dir / 'simulation_diagnostics.beta_log_by_maf.pdf'}")


if __name__ == "__main__":
    main()