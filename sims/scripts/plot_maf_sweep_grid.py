#!/usr/bin/env python3
"""Render a publication-style grid for MAF-threshold distribution recovery.

For each run_id, this script overlays truth and fitted-mixture |beta_s| densities
in a multi-panel grid and annotates fit metrics (KS, Wasserstein, variance error).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot publication-quality MAF sweep recovery grid.")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=[
            "dfe_maf_sweep_m0p005_n10000",
            "dfe_maf_sweep_m0p01_n10000",
            "dfe_maf_sweep_m0p02_n10000",
            "dfe_maf_sweep_m0p05_n10000",
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

    for idx, run_id in enumerate(args.run_ids):
        truth_path = results_dir / f"{run_id}.truth.tsv"
        infer_path = results_dir / f"{run_id}.infer.tsv"
        recovery_path = results_dir / f"{run_id}.recovery.json"
        meta_path = results_dir / f"{run_id}.meta.tsv"

        for path in (truth_path, infer_path, recovery_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"missing required file for grid plot: {path}")

        truth_rows = read_tsv(truth_path)
        infer_rows = read_tsv(infer_path)
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        meta = meta_to_dict(meta_path)

        beta_key = first_present_key(truth_rows, ["beta_s_true", "beta_true"])
        beta_true = np.asarray([float(r[beta_key]) for r in truth_rows], dtype=float)
        abs_truth = np.abs(beta_true)

        mu, var, wt, maf_ref, resolved_name = extract_reference_mixture(infer_rows, component_name, maf_selector)
        rng = np.random.default_rng(seed + idx)
        beta_fit = sample_mixture(rng, mu, var, wt, int(args.sample_size))
        abs_fit = np.abs(beta_fit)

        all_abs.extend([abs_truth, abs_fit])

        ks = float(recovery["distribution_capture"]["distance_metrics_abs_beta_s"]["ks_distance"])
        w1 = float(recovery["distribution_capture"]["distance_metrics_abs_beta_s"]["wasserstein_1"])
        var_err = float(recovery["distribution_capture"]["error_metrics"]["abs_error_var_beta_s"])

        n_truth_rows = recovery.get("sample_sizes", {}).get("n_truth_rows", len(truth_rows))
        n_fit_rows = recovery.get("sample_sizes", {}).get("n_fit_rows", None)
        maf_min = meta.get("maf_min", "NA")

        panel_data.append(
            {
                "run_id": run_id,
                "maf_min": maf_min,
                "abs_truth": abs_truth,
                "abs_fit": abs_fit,
                "ks": ks,
                "w1": w1,
                "var_err": var_err,
                "n_truth": n_truth_rows,
                "n_fit": n_fit_rows,
                "ref_maf": maf_ref,
                "ref_name": resolved_name,
            }
        )

    xmax = float(np.quantile(np.concatenate(all_abs), 0.997))
    xmax = max(xmax, 1e-4)
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
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 8.5), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    color_truth = "#1f4e79"
    color_fit = "#b22222"

    for i, panel in enumerate(panel_data):
        ax = axes_arr[i]
        abs_truth = panel["abs_truth"]
        abs_fit = panel["abs_fit"]

        dens_truth, edges = np.histogram(abs_truth, bins=bins, density=True)
        dens_fit, _ = np.histogram(abs_fit, bins=bins, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])

        ax.plot(mids, dens_truth, color=color_truth, lw=2.2, label="Simulated truth")
        ax.plot(mids, dens_fit, color=color_fit, lw=2.0, ls="--", label="Fitted mixture")

        ax.set_title(f"MAF cutoff = {panel['maf_min']}", fontsize=11, pad=6)
        ax.grid(alpha=0.18, lw=0.6)

        txt = (
            f"KS = {panel['ks']:.3f}\n"
            f"W1 = {panel['w1']:.3f}\n"
            f"|ΔVar| = {panel['var_err']:.3f}\n"
            f"n_truth={panel['n_truth']}, n_fit={panel['n_fit']}"
        )
        ax.text(
            0.98,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#bbbbbb"},
        )

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    for ax in axes_arr[:n]:
        ax.set_xlim(0.0, xmax)
        ax.set_xlabel("|beta_s|")
        ax.set_ylabel("Density")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        "DFE Simulation Recovery Across MAF Ascertainment Thresholds\n"
        "Simulated truth distribution vs mut_var fitted mixture",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    out_png = plots_dir / "maf_sweep_distribution_grid.png"
    out_pdf = plots_dir / "maf_sweep_distribution_grid.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"wrote: {out_png}")
    print(f"wrote: {out_pdf}")


if __name__ == "__main__":
    main()
