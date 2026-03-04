#!/usr/bin/env python3
"""Plot fitted-mixture versus simulated effect-size distributions.

This script compares the simulated truth distribution (from `<run_id>.truth.tsv`)
against a reconstructed mut_var mixture distribution (from `<run_id>.infer.tsv`).
Plots are written under `sims/plots/` by default.
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
    parser = argparse.ArgumentParser(description="Plot fitted versus simulated effect-size distributions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="sims/results")
    parser.add_argument("--plots-dir", default="sims/plots")
    parser.add_argument("--metrics-config", default="sims/config/eval_metrics.json")
    parser.add_argument("--sample-size", type=int, default=None, help="Override mixture sample size.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins for |beta| density.")
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
    if wsum <= 0.0:
        wt = np.full_like(wt, 1.0 / wt.size)
    else:
        wt = wt / wsum

    return mu, var, wt, maf_ref, resolved_name


def sample_mixture(rng, mu, var, wt, n):
    comp = rng.choice(np.arange(mu.size), size=n, p=wt)
    return rng.normal(loc=mu[comp], scale=np.sqrt(var[comp]), size=n)


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    truth_path = results_dir / f"{args.run_id}.truth.tsv"
    observed_path = results_dir / f"{args.run_id}.observed.tsv"
    infer_path = results_dir / f"{args.run_id}.infer.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth file: {truth_path}")
    if not infer_path.exists():
        raise FileNotFoundError(f"missing infer file: {infer_path}")

    cfg = json.loads(Path(args.metrics_config).read_text(encoding="utf-8"))
    component_name = str(cfg.get("reference_component_name", "pi0"))
    maf_selector = str(cfg.get("reference_maf_selector", "max"))
    seed = int(cfg.get("random_seed", 20260304))
    sample_n = int(cfg.get("mixture_sample_size", 50000))
    if args.sample_size is not None:
        sample_n = int(args.sample_size)

    truth_rows = read_tsv(truth_path)
    observed_rows = read_tsv(observed_path) if observed_path.exists() else []
    infer_rows = read_tsv(infer_path)

    beta_key = first_present_key(truth_rows, ["beta_s_true", "beta_true"])
    beta_true = np.asarray([float(r[beta_key]) for r in truth_rows], dtype=float)
    abs_truth = np.abs(beta_true)

    mu, var, wt, maf_ref, resolved_name = extract_reference_mixture(infer_rows, component_name, maf_selector)
    rng = np.random.default_rng(seed)
    beta_fit = sample_mixture(rng, mu, var, wt, sample_n)
    abs_fit = np.abs(beta_fit)

    ecdf_truth_x, ecdf_truth_y = ecdf(abs_truth)
    ecdf_fit_x, ecdf_fit_y = ecdf(abs_fit)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    max_x = float(np.quantile(np.concatenate((abs_truth, abs_fit)), 0.995))
    bins = np.linspace(0.0, max(max_x, 1e-6), max(20, int(args.bins)))
    axes[0].hist(abs_truth, bins=bins, density=True, alpha=0.45, label="Truth |beta_s|", color="#1f77b4")
    axes[0].hist(abs_fit, bins=bins, density=True, alpha=0.45, label="Fitted mixture |beta_s|", color="#d62728")
    axes[0].set_xlabel("|beta_s|")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Density overlay")
    axes[0].legend(frameon=False)

    axes[1].plot(ecdf_truth_x, ecdf_truth_y, color="#1f77b4", linewidth=2.0, label="Truth")
    axes[1].plot(ecdf_fit_x, ecdf_fit_y, color="#d62728", linewidth=2.0, label="Fitted mixture")
    axes[1].set_xlabel("|beta_s|")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Empirical CDF")
    axes[1].legend(frameon=False)

    surv_truth = 1.0 - ecdf_truth_y
    surv_fit = 1.0 - ecdf_fit_y
    axes[2].plot(ecdf_truth_x, np.clip(surv_truth, 1e-6, 1.0), color="#1f77b4", linewidth=2.0, label="Truth")
    axes[2].plot(ecdf_fit_x, np.clip(surv_fit, 1e-6, 1.0), color="#d62728", linewidth=2.0, label="Fitted mixture")
    axes[2].set_xlabel("|beta_s|")
    axes[2].set_ylabel("1 - ECDF")
    axes[2].set_yscale("log")
    axes[2].set_title("Tail survival (log scale)")
    axes[2].legend(frameon=False)

    fig.suptitle(
        f"Distribution recovery: {args.run_id}"
        f"\nref_name={resolved_name}, ref_maf={maf_ref:.6g}, "
        f"n_truth={beta_true.size}, n_fit_data={len(observed_rows) if observed_rows else 'NA'}, n_fit_draw={sample_n}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_png = plots_dir / f"{args.run_id}.distribution_recovery.png"
    out_pdf = plots_dir / f"{args.run_id}.distribution_recovery.pdf"
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"wrote: {out_png}")
    print(f"wrote: {out_pdf}")


if __name__ == "__main__":
    main()
