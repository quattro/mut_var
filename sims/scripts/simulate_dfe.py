#!/usr/bin/env python3
"""Simulate DFE summary statistics compatible with mutvar inference.

This simulator follows the equilibrium DFE pathway used in the original `df.py`
workflow: sample selection coefficients from a tabulated DFE (model name: SSD), sample allele
frequencies from an underdominant SFS conditional on selection, sample effects
under one-dimensional or high-dimensional stabilizing assumptions, and apply
ascertainment on variance contribution with optional GWAS noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special

@dataclass
class Scenario:
    run_id: str
    n_ascertained: int
    truth_reference_n: int
    seed: int
    truth_reference_seed: int | None
    dfe_log10_s_grid: np.ndarray
    dfe_weight_grid: np.ndarray
    demography_mode: str
    min_x: float
    configured_min_x: float
    generation_min_x_source: str
    n_x: int
    ne: float
    batch_size: int
    x_min: float
    x_max: float
    selection_mode: str
    ascertainment_mode: str
    maf_min: float | None
    v_cutoff: float | None
    p_threshold: float | None
    neff_override: float | None


def parse_args():
    parser = argparse.ArgumentParser(description="DFE simulator for mut_var compatibility studies.")
    parser.add_argument(
        "--config",
        default="sims/config/dfe_scenarios.json",
        help="Path to scenario JSON.",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="Scenario key in the config file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id override.",
    )
    parser.add_argument(
        "--n-ascertained",
        type=int,
        default=None,
        help="Optional n_ascertained override.",
    )
    parser.add_argument(
        "--truth-reference-n",
        type=int,
        default=None,
        help="Optional truth reference sample-size override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override.",
    )
    parser.add_argument(
        "--outdir",
        default="sims/results",
        help="Output directory for artifacts.",
    )
    return parser.parse_args()


def _read_scenario(config_path, scenario_key):
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if scenario_key not in payload:
        raise KeyError(f"scenario '{scenario_key}' not found in {config_path}")

    sc = payload[scenario_key]
    frequency_cfg = sc["frequency"]
    effect_cfg = sc["effect"]
    ascertainment_cfg = sc["ascertainment"]

    selection_mode = str(effect_cfg.get("selection_mode", "hd")).lower()
    if selection_mode in ("high_dimensional", "high-dimensional", "pleiotropic"):
        selection_mode = "hd"
    if selection_mode in ("one_dimensional", "one-dimensional", "single_trait", "single-trait"):
        selection_mode = "1d"

    ascertainment_mode = str(
        ascertainment_cfg.get(
            "ascertainment_statistic",
            ascertainment_cfg.get("mode", "noisy_hat_v"),
        )
    ).lower()
    if ascertainment_mode in ("threshold_on_maf", "maf", "maf_cutoff"):
        ascertainment_mode = "threshold_on_maf"
    if ascertainment_mode in ("threshold_on_v_true", "true_v", "none_true_v"):
        ascertainment_mode = "none_true_v"
    if ascertainment_mode in ("threshold_on_v_hat", "hat_v", "noisy_hat_v"):
        ascertainment_mode = "noisy_hat_v"

    maf_min = ascertainment_cfg.get("maf_min", ascertainment_cfg.get("min_maf"))
    v_cutoff = ascertainment_cfg.get("v_s_cutoff", ascertainment_cfg.get("v_cutoff"))
    p_threshold = ascertainment_cfg.get("p_value_threshold", ascertainment_cfg.get("p_threshold"))
    configured_min_x = float(frequency_cfg["min_x"])
    x_min_clip = float(frequency_cfg["x_min"])
    x_max_clip = float(frequency_cfg["x_max"])

    if ascertainment_mode == "threshold_on_maf":
        if maf_min is None:
            raise ValueError("ascertainment `threshold_on_maf` requires `maf_min`.")
        maf_min = float(maf_min)
        if not np.isfinite(maf_min) or maf_min <= 0.0 or maf_min >= 0.5:
            raise ValueError("ascertainment `maf_min` must satisfy 0 < maf_min < 0.5.")
        if x_min_clip > maf_min:
            raise ValueError(
                "frequency.x_min exceeds ascertainment.maf_min; this would break single-threshold "
                "MAF generation/ascertainment coupling."
            )
        if x_max_clip < (1.0 - maf_min):
            raise ValueError(
                "frequency.x_max is below 1 - ascertainment.maf_min; this would break single-threshold "
                "MAF generation/ascertainment coupling."
            )
        generation_min_x = maf_min
        generation_min_x_source = "ascertainment.maf_min"
    elif v_cutoff is None:
        raise ValueError("ascertainment requires `v_s_cutoff` (or legacy `v_cutoff`).")
    else:
        generation_min_x = configured_min_x
        generation_min_x_source = "frequency.min_x"
    if ascertainment_mode != "threshold_on_maf" and p_threshold is None:
        raise ValueError("ascertainment requires `p_value_threshold` (or legacy `p_threshold`).")

    n_eff_override = ascertainment_cfg.get("n_eff_override", ascertainment_cfg.get("neff_override"))

    return Scenario(
        run_id=str(sc["run_id"]),
        n_ascertained=int(sc["n_ascertained"]),
        truth_reference_n=int(sc.get("truth_reference_n", max(int(sc["n_ascertained"]) * 5, int(sc["n_ascertained"])))),
        seed=int(sc["seed"]),
        truth_reference_seed=(None if sc.get("truth_reference_seed") is None else int(sc.get("truth_reference_seed"))),
        dfe_log10_s_grid=np.asarray(sc["dfe"]["log10_s_grid"], dtype=float),
        dfe_weight_grid=np.asarray(sc["dfe"]["weight_grid"], dtype=float),
        demography_mode=str(frequency_cfg["demography_mode"]),
        min_x=float(generation_min_x),
        configured_min_x=configured_min_x,
        generation_min_x_source=generation_min_x_source,
        n_x=int(frequency_cfg["n_x"]),
        ne=float(frequency_cfg.get("N_e_ancestral", frequency_cfg.get("ne"))),
        batch_size=int(frequency_cfg["batch_size"]),
        x_min=x_min_clip,
        x_max=x_max_clip,
        selection_mode=selection_mode,
        ascertainment_mode=ascertainment_mode,
        maf_min=(None if maf_min is None else float(maf_min)),
        v_cutoff=(None if v_cutoff is None else float(v_cutoff)),
        p_threshold=(None if p_threshold is None else float(p_threshold)),
        neff_override=(None if n_eff_override is None else float(n_eff_override)),
    )


def _trad_x_set(min_x, n_points):
    lo = np.log(min_x / (1.0 - min_x))
    hi = np.log((1.0 - min_x) / min_x)
    return 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, n_points)))


def _sample_log10_s(rng, s_grid, w_grid, n):
    dense = np.linspace(float(np.min(s_grid)), float(np.max(s_grid)), 4000)
    dens = np.interp(dense, xp=s_grid, fp=np.maximum(w_grid, 0.0))
    dens = dens / np.trapezoid(dens, dense)
    dx = np.diff(dense)
    cdf = np.concatenate(([0.0], np.cumsum((dens[:-1] + dens[1:]) * 0.5 * dx)))
    cdf = cdf / cdf[-1]
    uu = rng.uniform(0.0, 1.0, size=n)
    return np.interp(uu, cdf, dense)


def _sfs_ud_params_log(xx, theta, s_ud_scaled):
    s_ud_scaled = np.abs(s_ud_scaled) + 1e-8
    root = np.sqrt(s_ud_scaled)
    non_erf_term = np.log(theta) - np.log(xx * (1.0 - xx)) - s_ud_scaled * xx * (1.0 - xx)
    ratio = special.erf(root * (0.5 - xx)) / special.erf(root / 2.0)
    ratio = np.clip(ratio, -1.0 + 1e-15, None)
    erf_term_low = np.log1p(ratio)

    a_term = special.log_ndtr(np.sqrt(2.0 * s_ud_scaled) * (0.5 - xx))
    b_term = special.log_ndtr(-np.sqrt(s_ud_scaled / 2.0))
    diff = -np.exp(b_term - a_term)
    diff = np.clip(diff, -1.0 + 1e-15, 0.0)
    erf_term_high = np.log(2.0) - np.log(special.erf(root / 2.0)) + a_term + np.log1p(diff)
    return np.where(xx < 0.5, non_erf_term + erf_term_low, non_erf_term + erf_term_high)


def _equilibrium_sample_x(rng, nn, sc):
    x_set = _trad_x_set(sc.min_x, sc.n_x)
    m0 = np.trapezoid(np.exp(_sfs_ud_params_log(x_set, 1.0, np.zeros_like(x_set))), x_set)

    out_x: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    have = 0
    proposal_batch = max(250, min(sc.batch_size, nn))
    while have < nn:
        s_sample = _sample_log10_s(rng, sc.dfe_log10_s_grid, sc.dfe_weight_grid, proposal_batch)
        s_ud = np.power(10.0, s_sample)
        s_ud_scaled = 2.0 * sc.ne * s_ud
        sfs_log = _sfs_ud_params_log(x_set[None, :], 1.0, (2.0 * s_ud_scaled)[:, None])
        sfs_mass = np.trapezoid(np.exp(sfs_log), x_set, axis=1)
        keep_prob = np.clip(sfs_mass / max(m0, 1e-300), 0.0, 1.0)
        keep = rng.uniform(size=proposal_batch) < keep_prob
        if not np.any(keep):
            continue

        kept_log = s_sample[keep]
        kept_sfs = np.exp(sfs_log[keep, :] - np.log(sfs_mass[keep, None]))
        cdf_grid = np.cumsum(
            (kept_sfs[:, 1:] * np.diff(x_set) + kept_sfs[:, :-1] * np.diff(x_set)) / 2.0,
            axis=1,
        )
        cdf_grid = np.pad(cdf_grid, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
        u = rng.uniform(size=kept_log.shape[0])
        x_sample = np.array([np.interp(u[i], cdf_grid[i, :], x_set) for i in range(kept_log.shape[0])], dtype=float)
        out_x.append(x_sample)
        out_s.append(kept_log)
        have += x_sample.shape[0]

    x_all = np.concatenate(out_x)[:nn]
    s_all = np.concatenate(out_s)[:nn]
    x_all = np.clip(x_all, sc.x_min, sc.x_max)
    return x_all, s_all


def _compute_neff(sc):
    if sc.neff_override is not None:
        return float(sc.neff_override)
    if sc.v_cutoff is None or sc.p_threshold is None:
        raise ValueError(
            "effective sample size is undefined without `n_eff_override` or both "
            "`v_s_cutoff` and `p_value_threshold`."
        )
    return float((2.0 * (special.erfinv(1.0 - sc.p_threshold) ** 2)) / sc.v_cutoff)


def _ascertainment_statistic_name(ascertainment_mode):
    if ascertainment_mode == "threshold_on_maf":
        return "threshold_on_maf"
    if ascertainment_mode == "none_true_v":
        return "threshold_on_v_true"
    if ascertainment_mode == "noisy_hat_v":
        return "threshold_on_v_hat"
    return ascertainment_mode


def _draw_effects(rng, log10_s, sc):
    s_ud_scaled = np.power(10.0, log10_s) * 2.0 * sc.ne
    if sc.selection_mode == "hd":
        return rng.normal(loc=0.0, scale=np.sqrt(s_ud_scaled), size=s_ud_scaled.shape[0])
    if sc.selection_mode == "1d":
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=s_ud_scaled.shape[0])
        return signs * np.sqrt(s_ud_scaled)
    raise ValueError(f"unsupported effect.selection_mode: {sc.selection_mode}")


def _draw_batch(rng, sc, neff, n_draw):
    if sc.demography_mode != "equilibrium":
        raise ValueError("only frequency.demography_mode='equilibrium' is implemented in this phase")

    x, log10_s = _equilibrium_sample_x(rng, n_draw, sc)
    beta_s_true = _draw_effects(rng, log10_s, sc)
    se = np.sqrt(1.0 / np.maximum(2.0 * x * (1.0 - x) * neff, 1e-20))

    v_s_true = 2.0 * x * (1.0 - x) * (beta_s_true ** 2)
    if sc.ascertainment_mode == "threshold_on_maf":
        maf = np.minimum(x, 1.0 - x)
        if sc.maf_min is None:
            raise ValueError("`maf_min` is required for threshold_on_maf mode")
        keep = maf >= sc.maf_min
        beta_s_obs = beta_s_true
        v_s_hat = v_s_true
        v_s_ascertain = v_s_true
        ascertain_stat_value = maf
    elif sc.ascertainment_mode == "none_true_v":
        keep = v_s_true > sc.v_cutoff
        beta_s_obs = beta_s_true
        v_s_hat = v_s_true
        v_s_ascertain = v_s_true
        ascertain_stat_value = v_s_true
    elif sc.ascertainment_mode == "noisy_hat_v":
        beta_s_hat = rng.normal(loc=beta_s_true, scale=se, size=beta_s_true.shape[0])
        v_s_hat = 2.0 * x * (1.0 - x) * (beta_s_hat ** 2)
        keep = v_s_hat > sc.v_cutoff
        beta_s_obs = beta_s_hat
        v_s_ascertain = v_s_hat
        ascertain_stat_value = v_s_hat
    else:
        raise ValueError(f"unsupported ascertainment.mode: {sc.ascertainment_mode}")

    s_ud = np.power(10.0, log10_s)
    return {
        "log10_s": log10_s[keep],
        "s_ud": s_ud[keep],
        "S_ud": (2.0 * sc.ne * s_ud)[keep],
        "effect_allele_frequency": x[keep],
        "beta_s_true": beta_s_true[keep],
        "beta_s_obs": beta_s_obs[keep],
        "beta": beta_s_obs[keep],
        "standard_error": se[keep],
        "v_s_true": v_s_true[keep],
        "v_s_hat": v_s_hat[keep],
        "v_s_ascertain": v_s_ascertain[keep],
        "ascertain_stat_value": ascertain_stat_value[keep],
        "ascertain_from_hat": np.full(np.sum(keep), sc.ascertainment_mode == "noisy_hat_v", dtype=bool),
    }


def _generate(sc, n_target, seed):
    rng = np.random.default_rng(seed)
    neff = _compute_neff(sc)

    acc: dict[str, list[np.ndarray]] = {
        "log10_s": [],
        "s_ud": [],
        "S_ud": [],
        "effect_allele_frequency": [],
        "beta_s_true": [],
        "standard_error": [],
        "beta_s_obs": [],
        "beta": [],
        "v_s_true": [],
        "v_s_hat": [],
        "v_s_ascertain": [],
        "ascertain_stat_value": [],
        "ascertain_from_hat": [],
    }

    total = 0
    while total < n_target:
        remaining = n_target - total
        draw_n = max(250, min(sc.batch_size, max(remaining * 4, remaining)))
        part = _draw_batch(rng, sc, neff, draw_n)
        take = min(n_target - total, part["beta"].shape[0])
        if take <= 0:
            continue
        for k, arr in part.items():
            acc[k].append(arr[:take])
        total += take

    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def _validate_observed(arrs):
    af = arrs["effect_allele_frequency"]
    beta = arrs["beta"]
    se = arrs["standard_error"]

    if af.size == 0:
        raise ValueError("observed output is empty")
    if np.any(~np.isfinite(af)) or np.any(~np.isfinite(beta)) or np.any(~np.isfinite(se)):
        raise ValueError("observed output contains non-finite values")
    if np.any((af < 0.0) | (af > 1.0)):
        raise ValueError("effect_allele_frequency must be in [0,1]")
    if np.any(se <= 0.0):
        raise ValueError("standard_error must be > 0")


def _write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config_path = Path(args.config)
    sc = _read_scenario(config_path, args.scenario)
    if args.run_id is not None:
        sc.run_id = args.run_id
    if args.n_ascertained is not None:
        sc.n_ascertained = int(args.n_ascertained)
    if args.truth_reference_n is not None:
        sc.truth_reference_n = int(args.truth_reference_n)
    if args.seed is not None:
        sc.seed = int(args.seed)

    truth_seed = sc.seed + 1 if sc.truth_reference_seed is None else sc.truth_reference_seed
    n_truth = max(int(sc.truth_reference_n), int(sc.n_ascertained))

    arrs_observed = _generate(sc, n_target=int(sc.n_ascertained), seed=int(sc.seed))
    _validate_observed(arrs_observed)
    if n_truth == int(sc.n_ascertained) and int(truth_seed) == int(sc.seed):
        arrs_truth = arrs_observed
    else:
        arrs_truth = _generate(sc, n_target=n_truth, seed=int(truth_seed))

    n_obs = arrs_observed["beta"].shape[0]
    n_truth_rows = arrs_truth["beta_s_true"].shape[0]
    row_id_obs = np.arange(n_obs, dtype=int)
    row_id_truth = np.arange(n_truth_rows, dtype=int)

    outdir = Path(args.outdir)
    observed_path = outdir / f"{sc.run_id}.observed.tsv"
    truth_path = outdir / f"{sc.run_id}.truth.tsv"
    meta_path = outdir / f"{sc.run_id}.meta.tsv"
    manifest_path = outdir / f"{sc.run_id}.manifest.json"

    observed_rows = [
        {
            "row_id": int(row_id_obs[i]),
            "effect_allele_frequency": float(arrs_observed["effect_allele_frequency"][i]),
            "beta": float(arrs_observed["beta"][i]),
            "standard_error": float(arrs_observed["standard_error"][i]),
        }
        for i in range(n_obs)
    ]
    _write_tsv(
        observed_path,
        observed_rows,
        ["row_id", "effect_allele_frequency", "beta", "standard_error"],
    )

    truth_rows = [
        {
            "row_id": int(row_id_truth[i]),
            "log10_s": float(arrs_truth["log10_s"][i]),
            "s_ud": float(arrs_truth["s_ud"][i]),
            "S_ud": float(arrs_truth["S_ud"][i]),
            "effect_allele_frequency": float(arrs_truth["effect_allele_frequency"][i]),
            "beta_s_true": float(arrs_truth["beta_s_true"][i]),
            "beta_s_obs": float(arrs_truth["beta_s_obs"][i]),
            "v_s_true": float(arrs_truth["v_s_true"][i]),
            "v_s_hat": float(arrs_truth["v_s_hat"][i]),
            "v_s_ascertain": float(arrs_truth["v_s_ascertain"][i]),
            "ascertain_stat_value": float(arrs_truth["ascertain_stat_value"][i]),
            "ascertain_from_hat": bool(arrs_truth["ascertain_from_hat"][i]),
        }
        for i in range(n_truth_rows)
    ]
    _write_tsv(
        truth_path,
        truth_rows,
        [
            "row_id",
            "log10_s",
            "s_ud",
            "S_ud",
            "effect_allele_frequency",
            "beta_s_true",
            "beta_s_obs",
            "v_s_true",
            "v_s_hat",
            "v_s_ascertain",
            "ascertain_stat_value",
            "ascertain_from_hat",
        ],
    )

    neff = _compute_neff(sc)
    meta_rows = [
        {"key": "run_id", "value": sc.run_id},
        {"key": "n_ascertained", "value": str(sc.n_ascertained)},
        {"key": "n_fit_rows", "value": str(n_obs)},
        {"key": "truth_reference_n", "value": str(n_truth)},
        {"key": "n_truth_rows", "value": str(n_truth_rows)},
        {"key": "seed", "value": str(sc.seed)},
        {"key": "truth_reference_seed", "value": str(truth_seed)},
        {"key": "demography_mode", "value": sc.demography_mode},
        {"key": "selection_mode", "value": sc.selection_mode},
        {"key": "ascertainment_mode", "value": sc.ascertainment_mode},
        {"key": "ascertainment_statistic", "value": _ascertainment_statistic_name(sc.ascertainment_mode)},
        {"key": "maf_min", "value": "" if sc.maf_min is None else f"{sc.maf_min:.12g}"},
        {"key": "frequency_min_x_configured", "value": f"{sc.configured_min_x:.12g}"},
        {"key": "generation_min_x_effective", "value": f"{sc.min_x:.12g}"},
        {"key": "generation_min_x_source", "value": sc.generation_min_x_source},
        {"key": "v_s_cutoff", "value": "" if sc.v_cutoff is None else f"{sc.v_cutoff:.12g}"},
        {"key": "p_value_threshold", "value": "" if sc.p_threshold is None else f"{sc.p_threshold:.12g}"},
        {"key": "neff", "value": f"{neff:.12g}"},
        {"key": "selection_scale", "value": "S_ud = 2Ne*s_ud"},
        {"key": "beta_scale", "value": "beta_s"},
        {"key": "mean_beta_s_true_sq", "value": f"{float(np.mean(arrs_truth['beta_s_true'] ** 2)):.12g}"},
        {"key": "mean_abs_beta_s_true", "value": f"{float(np.mean(np.abs(arrs_truth['beta_s_true']))):.12g}"},
        {"key": "mean_af", "value": f"{float(np.mean(arrs_truth['effect_allele_frequency'])):.12g}"},
        {"key": "mean_se", "value": f"{float(np.mean(arrs_truth['standard_error'])):.12g}"},
        {"key": "mean_v_s_true", "value": f"{float(np.mean(arrs_truth['v_s_true'])):.12g}"},
        {"key": "mean_v_s_hat", "value": f"{float(np.mean(arrs_truth['v_s_hat'])):.12g}"},
    ]
    _write_tsv(meta_path, meta_rows, ["key", "value"])

    manifest = {
        "run_id": sc.run_id,
        "scenario_config": str(config_path),
        "scenario_key": args.scenario,
        "seed": sc.seed,
        "n_ascertained": sc.n_ascertained,
        "truth_reference_n": n_truth,
        "files": {
            "observed": str(observed_path),
            "truth": str(truth_path),
            "meta": str(meta_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote: {observed_path}")
    print(f"wrote: {truth_path}")
    print(f"wrote: {meta_path}")
    print(f"wrote: {manifest_path}")


if __name__ == "__main__":
    main()
