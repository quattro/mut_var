#!/usr/bin/env python3
"""Evaluate recovery of effect-size distribution under DFE simulation.

Reads simulator truth and mutvar infer outputs, then writes distribution-first
recovery metrics. AF-dependent diagnostics are optional and secondary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute recovery metrics for a simulation run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="sims/results")
    parser.add_argument("--metrics-config", default="sims/config/eval_metrics.json")
    return parser.parse_args()


def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_float_array(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def first_present_key(rows, candidates):
    if not rows:
        raise ValueError("no rows found when resolving column candidates")
    header = set(rows[0].keys())
    for key in candidates:
        if key in header:
            return key
    raise KeyError(f"none of the expected columns found: {candidates}")


def linfit(x, y):
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xx = float(np.sum((x - x_mean) ** 2))
    if xx <= 0.0:
        return 0.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / xx)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def relative_error(estimate, truth):
    return float(abs(estimate - truth) / max(abs(truth), 1e-12))


def ks_distance(x, y):
    if x.size == 0 or y.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    grid = np.sort(np.concatenate((x_sorted, y_sorted)))
    cdf_x = np.searchsorted(x_sorted, grid, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, grid, side="right") / y_sorted.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_1d(x, y, n_q):
    if x.size == 0 or y.size == 0:
        return 0.0
    qq = np.linspace(0.001, 0.999, max(8, int(n_q)))
    qx = np.quantile(x, qq)
    qy = np.quantile(y, qq)
    return float(np.mean(np.abs(qx - qy)))


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
    if np.any(var < 0.0):
        var = np.maximum(var, 0.0)

    wsum = float(np.sum(wt))
    if wsum <= 0.0:
        wt = np.full_like(wt, 1.0 / wt.size)
    else:
        wt = wt / wsum
    return mu, var, wt, maf_ref, resolved_name


def sample_mixture(rng, mu, var, wt, n):
    comp = rng.choice(np.arange(mu.size), size=n, p=wt)
    return rng.normal(loc=mu[comp], scale=np.sqrt(np.maximum(var[comp], 0.0)), size=n)


def compute_af_dependence_truth(beta_true, af, bins):
    mids: list[float] = []
    logv: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (af >= lo) & (af < hi)
        if np.sum(mask) < 50:
            continue
        v = float(np.var(beta_true[mask]))
        xmid = float(0.5 * (lo + hi))
        maf_term = max(2.0 * xmid * (1.0 - xmid), 1e-12)
        mids.append(np.log(maf_term))
        logv.append(np.log(max(v, 1e-20)))

    if len(mids) >= 2:
        x = np.asarray(mids, dtype=float)
        y = np.asarray(logv, dtype=float)
        slope, intercept = linfit(x, y)
        yhat = intercept + slope * x
        sst = float(np.sum((y - np.mean(y)) ** 2))
        sse = float(np.sum((y - yhat) ** 2))
        r2 = 1.0 - sse / sst if sst > 0.0 else 0.0
    else:
        slope = 0.0
        r2 = 0.0

    return {
        "slope_log_varbeta_vs_log_2x1mx": float(slope),
        "r2_log_varbeta_vs_log_2x1mx": float(r2),
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    truth_path = results_dir / f"{args.run_id}.truth.tsv"
    observed_path = results_dir / f"{args.run_id}.observed.tsv"
    curve_path = results_dir / f"{args.run_id}.curve.tsv"
    infer_path = results_dir / f"{args.run_id}.infer.tsv"
    out_path = results_dir / f"{args.run_id}.recovery.json"

    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth file: {truth_path}")
    if not infer_path.exists():
        raise FileNotFoundError(f"missing infer file: {infer_path}")

    truth_rows = read_tsv(truth_path)
    observed_rows = read_tsv(observed_path) if observed_path.exists() else []
    af = safe_float_array(truth_rows, "effect_allele_frequency")
    beta_truth_col = first_present_key(truth_rows, ["beta_s_true", "beta_true"])
    beta_true = safe_float_array(truth_rows, beta_truth_col)
    infer_rows = read_tsv(infer_path)

    n_truth_rows = int(len(truth_rows))
    n_fit_rows = int(len(observed_rows)) if observed_rows else None
    truth_exceeds_fit = (n_truth_rows > n_fit_rows) if n_fit_rows is not None else None
    if truth_exceeds_fit is False:
        print(
            "warning: truth sample-size is not larger than observed fit sample-size; "
            "set truth_reference_n > n_ascertained for lower-noise truth diagnostics"
        )

    metrics_cfg = json.loads(Path(args.metrics_config).read_text(encoding="utf-8"))
    quantiles = np.asarray(metrics_cfg.get("distribution_quantiles", [0.9, 0.95, 0.99]), dtype=float)
    tail_probs = np.asarray(metrics_cfg.get("tail_mass_probs", [0.95, 0.99]), dtype=float)
    mc_samples = int(metrics_cfg.get("mixture_sample_size", max(50000, int(beta_true.size * 5))))
    rng_seed = int(metrics_cfg.get("random_seed", 20260304))
    component_name = str(metrics_cfg.get("reference_component_name", "pi0"))
    maf_selector = str(metrics_cfg.get("reference_maf_selector", "max"))
    n_wass_quantiles = int(metrics_cfg.get("wasserstein_quantiles", 512))

    compute_af_secondary = bool(metrics_cfg.get("compute_af_dependence_secondary", False))
    af_bins = np.asarray(metrics_cfg.get("af_bins", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999]), dtype=float)

    mu, var, wt, maf_ref, resolved_name = extract_reference_mixture(infer_rows, component_name, maf_selector)
    rng = np.random.default_rng(rng_seed)
    n_draw = max(mc_samples, int(beta_true.size))
    beta_mixture = sample_mixture(rng, mu, var, wt, n_draw)

    abs_beta = np.abs(beta_true)
    abs_beta_mix = np.abs(beta_mixture)

    true_mean = float(np.mean(beta_true))
    true_var = float(np.var(beta_true))
    true_mean_abs = float(np.mean(abs_beta))

    mix_mean = float(np.sum(wt * mu))
    mix_second = float(np.sum(wt * (var + mu * mu)))
    mix_var = float(max(mix_second - mix_mean * mix_mean, 0.0))
    mix_mean_abs = float(np.mean(abs_beta_mix))

    truth_metrics = {
        "n": int(beta_true.size),
        "mean_beta_s_true": true_mean,
        "var_beta_s_true": true_var,
        "mean_abs_beta_s_true": true_mean_abs,
    }

    q_labels = [f"q{int(round(100 * q))}" for q in quantiles]
    truth_abs_q = {label: float(np.quantile(abs_beta, q)) for label, q in zip(q_labels, quantiles)}
    mixture_abs_q = {label: float(np.quantile(abs_beta_mix, q)) for label, q in zip(q_labels, quantiles)}

    tail_labels = [f"above_truth_q{int(round(100 * p))}" for p in tail_probs]
    truth_tail_mass: dict[str, float] = {}
    mix_tail_mass: dict[str, float] = {}
    for label, p in zip(tail_labels, tail_probs):
        threshold = float(np.quantile(abs_beta, p))
        truth_tail_mass[label] = float(np.mean(abs_beta > threshold))
        mix_tail_mass[label] = float(np.mean(abs_beta_mix > threshold))

    distribution_capture = {
        "truth_summary": truth_metrics,
        "inferred_mixture_summary": {
            "n_components": int(mu.size),
            "reference_component_name": resolved_name,
            "reference_maf": float(maf_ref),
            "mean_beta_s_mixture": mix_mean,
            "var_beta_s_mixture": mix_var,
            "mean_abs_beta_s_mixture": mix_mean_abs,
        },
        "quantiles_abs_beta_s": {
            "truth": truth_abs_q,
            "mixture": mixture_abs_q,
        },
        "tail_mass_abs_beta_s": {
            "truth": truth_tail_mass,
            "mixture": mix_tail_mass,
        },
        "error_metrics": {
            "abs_error_mean_beta_s": float(abs(mix_mean - true_mean)),
            "abs_error_var_beta_s": float(abs(mix_var - true_var)),
            "abs_error_mean_abs_beta_s": float(abs(mix_mean_abs - true_mean_abs)),
            "rel_error_mean_beta_s": relative_error(mix_mean, true_mean),
            "rel_error_var_beta_s": relative_error(mix_var, true_var),
            "rel_error_mean_abs_beta_s": relative_error(mix_mean_abs, true_mean_abs),
            "abs_quantile_errors": {
                label: float(abs(mixture_abs_q[label] - truth_abs_q[label])) for label in q_labels
            },
            "abs_tail_mass_errors": {
                label: float(abs(mix_tail_mass[label] - truth_tail_mass[label])) for label in tail_labels
            },
        },
        "distance_metrics_abs_beta_s": {
            "ks_distance": ks_distance(abs_beta, abs_beta_mix),
            "wasserstein_1": wasserstein_1d(abs_beta, abs_beta_mix, n_q=n_wass_quantiles),
        },
        "sampling": {
            "mixture_sample_size": int(n_draw),
            "random_seed": int(rng_seed),
        },
    }

    af_secondary = None
    if compute_af_secondary:
        af_secondary = compute_af_dependence_truth(beta_true, af, af_bins)

    curve_summary = {}
    if curve_path.exists():
        curve_rows = read_tsv(curve_path)
        if curve_rows:
            coef_rate = np.asarray([float(r["coef_rate"]) for r in curve_rows], dtype=float)
            curve_summary = {
                "n_curve_rows": int(len(curve_rows)),
                "mean_coef_rate": float(np.mean(coef_rate)),
                "median_coef_rate": float(np.median(coef_rate)),
            }

    recovery = {
        "run_id": args.run_id,
        "sample_sizes": {
            "n_truth_rows": n_truth_rows,
            "n_fit_rows": n_fit_rows,
            "truth_exceeds_fit": truth_exceeds_fit,
        },
        "distribution_capture": distribution_capture,
        "af_dependence_secondary": af_secondary,
        "mutvar_curve": curve_summary,
        "files": {
            "observed": str(observed_path) if observed_path.exists() else None,
            "truth": str(truth_path),
            "infer": str(infer_path),
            "curve": str(curve_path) if curve_path.exists() else None,
        },
    }

    out_path.write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
