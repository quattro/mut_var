#!/usr/bin/env python3
"""Render a publication-style grid for MAF-threshold distribution recovery.

For each run_id, this script overlays true, observed, and fitted |beta_s| summaries
in a multi-panel grid with component and tail diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser(description="Plot publication-quality MAF cutoff recovery grid.")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=[
            "dfe_maf_cutoff_m0p05_n10000",
            "dfe_maf_cutoff_m0p01_n10000",
            "dfe_maf_cutoff_m0p005_n10000",
            "dfe_maf_cutoff_m0p001_n10000",
        ],
        help="Run IDs to include in the grid, ordered left-to-right, top-to-bottom.",
    )
    parser.add_argument("--results-dir", default="sims/results")
    parser.add_argument("--plots-dir", default="sims/plots")
    parser.add_argument("--metrics-config", default="sims/config/eval_metrics.json")
    parser.add_argument("--sample-size", type=int, default=120000)
    parser.add_argument("--bins", type=int, default=140)
    return parser.parse_args()


def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def first_present_key(rows, candidates):
    if not rows:
        raise ValueError("no rows found when resolving column candidates")
    header = set(rows[0].keys())
    for key in candidates:
        if key in header:
            return key
    raise KeyError(f"none of the expected columns found: {candidates}")


def extract_reference_mixture(infer_rows, component_name, maf_selector):
    chosen_rows = [r for r in infer_rows if r.get("name") == component_name]
    resolved_name = component_name
    if not chosen_rows:
        chosen_rows = infer_rows
        resolved_name = "all"

    mafs = np.asarray([float(r["maf"]) for r in chosen_rows], dtype=float)
    unique_mafs = np.unique(mafs)
    if unique_mafs.size == 0:
        raise ValueError("infer rows did not contain valid `maf` values")

    selector = str(maf_selector).lower()
    if selector in ("min", "lowest", "smallest"):
        maf_ref = float(np.min(unique_mafs))
    elif selector in ("median", "middle"):
        maf_ref = float(np.quantile(unique_mafs, 0.5))
    else:
        maf_ref = float(np.max(unique_mafs))

    at_ref = [r for r in chosen_rows if math.isclose(float(r["maf"]), maf_ref, rel_tol=0.0, abs_tol=1e-15)]
    if not at_ref:
        idx = int(np.argmin(np.abs(unique_mafs - maf_ref)))
        maf_ref = float(unique_mafs[idx])
        at_ref = [r for r in chosen_rows if math.isclose(float(r["maf"]), maf_ref, rel_tol=0.0, abs_tol=1e-15)]

    mu = np.asarray([float(r["mu0"]) for r in at_ref], dtype=float)
    var = np.asarray([float(r["var0"]) for r in at_ref], dtype=float)
    wt = np.asarray([max(float(r["value"]), 0.0) for r in at_ref], dtype=float)
    if mu.size == 0 or var.size == 0 or wt.size == 0:
        raise ValueError("no mixture components found at selected reference MAF")

    var = np.maximum(var, 0.0)
    wsum = float(np.sum(wt))
    wt = (wt / wsum) if wsum > 0.0 else np.full_like(wt, 1.0 / wt.size)
    return mu, var, wt, maf_ref, resolved_name


def sample_mixture(rng, mu, var, wt, n):
    comp = rng.choice(np.arange(mu.size), size=n, p=wt)
    return rng.normal(loc=mu[comp], scale=np.sqrt(var[comp]), size=n)


def folded_component_density(x, mu, sigma):
    z1 = (x - mu) / sigma
    z2 = (x + mu) / sigma
    return (np.exp(-0.5 * z1 * z1) + np.exp(-0.5 * z2 * z2)) / (sigma * np.sqrt(2.0 * np.pi))


def meta_to_dict(meta_path):
    rows = read_tsv(meta_path)
    out = {}
    for row in rows:
        out[row["key"]] = row["value"]
    return out


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.metrics_config).read_text(encoding="utf-8"))
    component_name = str(cfg.get("reference_component_name", "pi0"))
    maf_selector = str(cfg.get("reference_maf_selector", "max"))
    seed = int(cfg.get("random_seed", 20260304))

    panel_data = []
    all_abs = []
    nz_threshold = 1e-4

    for idx, run_id in enumerate(args.run_ids):
        observed_path = results_dir / f"{run_id}.observed.tsv"
        truth_path = results_dir / f"{run_id}.truth.tsv"
        infer_path = results_dir / f"{run_id}.infer.tsv"
        meta_path = results_dir / f"{run_id}.meta.tsv"

        for path in (observed_path, truth_path, infer_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"missing required file for grid plot: {path}")

        observed_rows = read_tsv(observed_path)
        truth_rows = read_tsv(truth_path)
        infer_rows = read_tsv(infer_path)
        meta = meta_to_dict(meta_path)

        beta_obs = np.asarray([float(r["beta"]) for r in observed_rows], dtype=float)
        abs_obs = np.abs(beta_obs)
        beta_true_key = first_present_key(truth_rows, ["beta_s_true", "beta_true"])
        beta_true = np.asarray([float(r[beta_true_key]) for r in truth_rows], dtype=float)
        abs_true = np.abs(beta_true)

        mu, var, wt, maf_ref, resolved_name = extract_reference_mixture(infer_rows, component_name, maf_selector)
        sigma = np.sqrt(np.maximum(var, 1e-16))
        rng = np.random.default_rng(seed + idx)
        beta_fit = sample_mixture(rng, mu, var, wt, int(args.sample_size))
        abs_fit = np.abs(beta_fit)

        all_abs.extend([abs_true, abs_obs, abs_fit])

        k_nonzero = int(np.sum(wt > nz_threshold))

        n_true_rows = int(abs_true.size)
        n_obs_rows = int(abs_obs.size)
        n_fit_rows = int(abs_fit.size)
        maf_min = meta.get("maf_min", "NA")

        panel_data.append(
            {
                "run_id": run_id,
                "maf_min": maf_min,
                "abs_true": abs_true,
                "abs_obs": abs_obs,
                "abs_fit": abs_fit,
                "mu": mu,
                "sigma": sigma,
                "wt": wt,
                "k_nonzero": k_nonzero,
                "n_true": n_true_rows,
                "n_obs": n_obs_rows,
                "n_fit": n_fit_rows,
                "ref_maf": maf_ref,
                "ref_name": resolved_name,
            }
        )

    pooled_abs = np.concatenate(all_abs)
    xmax = float(np.quantile(pooled_abs, 0.997))
    xmax = max(xmax, 1e-4)
    body_xmin = float(np.quantile(pooled_abs, 0.02))
    body_xmax = float(np.quantile(pooled_abs, 0.98))
    body_xmin = max(body_xmin, 1e-4)
    body_xmax = min(body_xmax, xmax)
    if body_xmax <= body_xmin * 1.05:
        body_xmin = max(xmax * 0.02, 1e-4)
        body_xmax = xmax
    bins = np.linspace(0.0, xmax, max(40, int(args.bins)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 170,
        }
    )

    n = len(panel_data)
    ncols = 2
    nrows = n
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.8, 2.6 * nrows),
        sharex="col",
        gridspec_kw={"wspace": 0.16},
    )
    if nrows == 1:
        axes = np.asarray([axes])

    color_true = "#E69F00"
    color_obs = "#1f4e79"
    color_fit = "#b22222"
    color_mix = "#117733"
    color_comp = "#8c6bb1"
    component_alpha = 0.45

    for i, panel in enumerate(panel_data):
        ax_density = axes[i, 0]
        ax_tail = axes[i, 1]

        abs_true = panel["abs_true"]
        abs_obs = panel["abs_obs"]
        abs_fit = panel["abs_fit"]
        mu = panel["mu"]
        sigma = panel["sigma"]
        wt = panel["wt"]

        dens_true, edges = np.histogram(abs_true, bins=bins, density=True)
        dens_obs, edges = np.histogram(abs_obs, bins=bins, density=True)
        dens_fit, _ = np.histogram(abs_fit, bins=bins, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])
        x_grid = np.linspace(0.0, xmax, 500)

        comp_densities = []
        for j in range(mu.size):
            comp = wt[j] * folded_component_density(x_grid, mu[j], sigma[j])
            comp_densities.append(comp)
        mix_density = np.sum(np.vstack(comp_densities), axis=0)

        ax_density.plot(mids, dens_true, color=color_true, lw=2.2)
        ax_density.plot(mids, dens_obs, color=color_obs, lw=2.2)
        ax_density.plot(mids, dens_fit, color=color_fit, lw=1.9, ls="--")
        ax_density.plot(x_grid, mix_density, color=color_mix, lw=2.0)
        for comp in comp_densities:
            ax_density.plot(x_grid, comp, color=color_comp, lw=1.0, alpha=component_alpha)

        body_mask_mid = (mids >= body_xmin) & (mids <= body_xmax)
        body_mask_grid = (x_grid >= body_xmin) & (x_grid <= body_xmax)
        body_peak = 0.0
        if np.any(body_mask_mid):
            body_peak = max(
                body_peak,
                float(np.nanmax(dens_true[body_mask_mid])),
                float(np.nanmax(dens_obs[body_mask_mid])),
                float(np.nanmax(dens_fit[body_mask_mid])),
            )
        if np.any(body_mask_grid):
            body_peak = max(body_peak, float(np.nanmax(mix_density[body_mask_grid])))
        if body_peak > 0.0:
            ax_density.set_ylim(0.0, body_peak * 1.08)

        ax_density.grid(alpha=0.18, lw=0.6)

        tail_true = np.where(dens_true > 0.0, dens_true, np.nan)
        tail_obs = np.where(dens_obs > 0.0, dens_obs, np.nan)
        tail_fit = np.where(dens_fit > 0.0, dens_fit, np.nan)
        tail_mix = np.where(mix_density > 0.0, mix_density, np.nan)
        ax_tail.plot(mids, tail_true, color=color_true, lw=2.2)
        ax_tail.plot(mids, tail_obs, color=color_obs, lw=2.2)
        ax_tail.plot(mids, tail_fit, color=color_fit, lw=1.9, ls="--")
        ax_tail.plot(x_grid, tail_mix, color=color_mix, lw=2.0)
        ax_tail.set_yscale("log")
        ax_tail.grid(alpha=0.18, lw=0.6)

        txt = f"k_nonzero={panel['k_nonzero']}"
        ax_density.text(
            0.98,
            0.96,
            txt,
            transform=ax_density.transAxes,
            va="top",
            ha="right",
            fontsize=8.3,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.86, "edgecolor": "#bbbbbb"},
        )
        ax_density.text(0.01, 0.96, f"MAF cutoff = {panel['maf_min']}", transform=ax_density.transAxes, va="top", ha="left", fontsize=10.5, fontweight="bold")

    for i in range(nrows):
        axes[i, 0].set_xlim(body_xmin, body_xmax)
        axes[i, 1].set_xlim(0.0, xmax)
        axes[i, 0].set_ylabel("Density")
        axes[i, 1].set_ylabel("Density (log)")

    axes[-1, 0].set_xlabel("|beta_s|")
    axes[-1, 1].set_xlabel("|beta_s|")
    axes[0, 0].set_title("Body view with mixture components", fontsize=11)
    axes[0, 1].set_title("Tail view (log y-axis)", fontsize=11)

    n_obs_unique = sorted({int(panel["n_obs"]) for panel in panel_data})
    obs_note = str(n_obs_unique[0]) if len(n_obs_unique) == 1 else ",".join(str(v) for v in n_obs_unique)

    legend_handles = [
        Line2D([0], [0], color=color_true, lw=2.2, label="True |beta|"),
        Line2D([0], [0], color=color_obs, lw=2.2, label="Observed noisy |beta|"),
        Line2D([0], [0], color=color_fit, lw=1.9, ls="--", label="Sampled fitted |beta|"),
        Line2D([0], [0], color=color_mix, lw=2.0, label="Analytic fitted mixture"),
        Line2D([0], [0], color=color_comp, lw=1.2, alpha=component_alpha, label="Mixture components"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
    )
    fig.text(0.5, 0.955, f"Shared fit input size across runs: n_obs={obs_note}", ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.925, hspace=0.22)

    out_pdf = plots_dir / "maf_cutoff_distribution_grid.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"wrote: {out_pdf}")


if __name__ == "__main__":
    main()
