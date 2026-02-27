from __future__ import annotations

# pattern: Imperative Shell
import argparse
import math

from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import polars as pl


class PlotConfig(NamedTuple):
    truth_path: Path | None
    infer_path: Path
    output_dir: Path
    output_prefix: str
    axis_min: float
    axis_max: float
    maf_min: float
    component_mode: Literal["matched", "inferred"]


def _validate_columns(df: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _truth_component_centers(truth_df: pl.DataFrame) -> list[float]:
    centers_df = (
        truth_df.group_by("component")
        .agg(pl.col("sigma2").mean().alias("mean_sigma2"))
        .sort("component")
    )
    if centers_df.height == 0:
        raise ValueError("truth data has no components")
    return [float(value) for value in centers_df.get_column("mean_sigma2").to_list()]


def _build_comparison_dataframe(
    truth_df: pl.DataFrame,
    infer_df: pl.DataFrame,
    *,
    maf_min: float,
) -> tuple[pl.DataFrame, list[tuple[float, str]]]:
    truth_centers = _truth_component_centers(truth_df)
    truth_centers_log = [math.log(max(value, 1e-300)) for value in truth_centers]
    num_components = len(truth_centers)

    maf_values = sorted(float(value) for value in infer_df.select("maf").unique().get_column("maf").to_list())
    maf_values = [maf for maf in maf_values if maf >= maf_min]

    rows: list[dict[str, float | int]] = []
    skipped: list[tuple[float, str]] = []

    for maf in maf_values:
        truth_subset = truth_df.filter(
            (pl.col("effect_allele_frequency") >= maf) & (pl.col("effect_allele_frequency") <= (1.0 - maf))
        )
        n_truth = truth_subset.height
        if n_truth == 0:
            skipped.append((maf, "empty truth subset"))
            continue

        truth_counts = {int(row[0]): int(row[1]) for row in truth_subset.group_by("component").len().iter_rows()}

        infer_group = infer_df.filter(pl.col("maf") == maf).filter(pl.col("var0") > 0.0)
        if infer_group.height == 0:
            skipped.append((maf, "no non-null inferred components"))
            continue

        infer_candidates: list[tuple[float, float]] = []
        assigned = {idx: 0.0 for idx in range(num_components)}
        total_assigned_mass = 0.0

        for row in infer_group.select(["var0", "value"]).iter_rows(named=True):
            var0 = float(row["var0"])
            weight = float(row["value"])
            if not math.isfinite(var0) or var0 <= 0.0:
                continue
            if not math.isfinite(weight) or weight < 0.0:
                continue
            infer_candidates.append((var0, weight))

            distances = [abs(math.log(var0) - center) for center in truth_centers_log]
            component = min(range(num_components), key=lambda idx: distances[idx])
            assigned[component] += weight
            total_assigned_mass += weight

        if not infer_candidates:
            skipped.append((maf, "no valid inferred candidates after finite filtering"))
            continue

        if total_assigned_mass <= 0.0 or not math.isfinite(total_assigned_mass):
            skipped.append((maf, f"invalid inferred assigned mass={total_assigned_mass}"))
            continue

        for component in range(num_components):
            simulated_proportion = truth_counts.get(component, 0) / n_truth
            inferred_raw = assigned[component]
            inferred_proportion = inferred_raw / total_assigned_mass
            closest_var0, closest_weight = min(
                infer_candidates,
                key=lambda pair: abs(math.log(pair[0]) - truth_centers_log[component]),
            )
            closest_inferred_proportion = closest_weight / total_assigned_mass
            rows.append(
                {
                    "maf": float(maf),
                    "component": int(component),
                    "n_truth_rows": int(n_truth),
                    "simulated_proportion": float(simulated_proportion),
                    "inferred_proportion": float(inferred_proportion),
                    "inferred_raw_proportion": float(inferred_raw),
                    "inferred_assigned_mass": float(total_assigned_mass),
                    "true_component_sigma2": float(truth_centers[component]),
                    "closest_inferred_var0": float(closest_var0),
                    "closest_inferred_raw_proportion": float(closest_weight),
                    "closest_inferred_proportion": float(closest_inferred_proportion),
                }
            )

    if not rows:
        raise ValueError("no thresholds produced component-proportion comparisons")

    return pl.DataFrame(rows).sort(["component", "maf"]), skipped


def _build_inferred_component_dataframe(
    infer_df: pl.DataFrame,
    *,
    maf_min: float,
) -> tuple[pl.DataFrame, list[tuple[float, str]]]:
    filtered = infer_df.filter(pl.col("maf") >= maf_min).filter(pl.col("var0") > 0.0)
    if filtered.height == 0:
        raise ValueError("no inferred non-null components available after filtering")

    var0_values = sorted(float(value) for value in filtered.select("var0").unique().get_column("var0").to_list())
    component_map = {var0: idx for idx, var0 in enumerate(var0_values)}

    maf_values = sorted(float(value) for value in filtered.select("maf").unique().get_column("maf").to_list())
    rows: list[dict[str, float | int]] = []
    skipped: list[tuple[float, str]] = []

    for maf in maf_values:
        group = filtered.filter(pl.col("maf") == maf)
        selected_mass = 0.0
        for value in group.get_column("value").to_list():
            weight = float(value)
            if math.isfinite(weight) and weight >= 0.0:
                selected_mass += weight

        if selected_mass <= 0.0 or not math.isfinite(selected_mass):
            skipped.append((maf, f"invalid inferred selected mass={selected_mass}"))
            continue

        for row in group.select(["var0", "value"]).iter_rows(named=True):
            var0 = float(row["var0"])
            weight = float(row["value"])
            if not math.isfinite(var0) or var0 <= 0.0:
                continue
            if not math.isfinite(weight) or weight < 0.0:
                continue
            rows.append(
                {
                    "maf": float(maf),
                    "component": int(component_map[var0]),
                    "var0": float(var0),
                    "inferred_raw_proportion": float(weight),
                    "inferred_selected_mass": float(selected_mass),
                    "inferred_proportion": float(weight / selected_mass),
                }
            )

    if not rows:
        raise ValueError("no thresholds produced inferred component-proportion rows")

    return pl.DataFrame(rows).sort(["component", "maf"]), skipped


def _render_vs_maf_plot(compare_df: pl.DataFrame, *, axis_min: float, axis_max: float, output_path: Path) -> None:
    components = sorted(
        int(value) for value in compare_df.select("component").unique().get_column("component").to_list()
    )

    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(8, 3.5 * len(components)),
        sharex=True,
        constrained_layout=True,
    )
    if len(components) == 1:
        axes = [axes]

    for ax, component in zip(axes, components, strict=False):
        comp_df = compare_df.filter(pl.col("component") == component).sort("maf")
        true_sigma2 = float(comp_df.get_column("true_component_sigma2")[0])
        median_closest_var0 = float(comp_df.select(pl.col("closest_inferred_var0").median())[0, 0])
        ax.semilogx(
            comp_df.get_column("maf").to_list(),
            comp_df.get_column("simulated_proportion").to_list(),
            "o-",
            label="simulated",
        )
        ax.semilogx(
            comp_df.get_column("maf").to_list(),
            comp_df.get_column("inferred_proportion").to_list(),
            "s--",
            label="inferred (assigned, renorm)",
        )
        ax.semilogx(
            comp_df.get_column("maf").to_list(),
            comp_df.get_column("closest_inferred_proportion").to_list(),
            "x-.",
            label="closest inferred component",
        )
        ax.set_ylabel("Proportion")
        ax.set_ylim(axis_min, axis_max)
        ax.set_title(
            f"Component {component} (true σ²={true_sigma2:.4g}, closest inferred var0~{median_closest_var0:.4g})"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("MAF threshold")
    fig.suptitle("Simulated vs inferred component proportions across MAF thresholds", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_inferred_components_vs_maf_plot(
    infer_component_df: pl.DataFrame,
    *,
    axis_min: float,
    axis_max: float,
    output_path: Path,
) -> None:
    components = sorted(
        int(value) for value in infer_component_df.select("component").unique().get_column("component").to_list()
    )

    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(8, 3.0 * len(components)),
        sharex=True,
        constrained_layout=True,
    )
    if len(components) == 1:
        axes = [axes]

    for ax, component in zip(axes, components, strict=False):
        comp_df = infer_component_df.filter(pl.col("component") == component).sort("maf")
        var0_value = float(comp_df.get_column("var0")[0])
        ax.semilogx(
            comp_df.get_column("maf").to_list(),
            comp_df.get_column("inferred_proportion").to_list(),
            "o-",
            label=f"inferred component {component}",
        )
        ax.set_ylabel("Proportion")
        ax.set_ylim(axis_min, axis_max)
        ax.set_title(f"Inferred component {component} (var0={var0_value:.4g})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("MAF threshold")
    fig.suptitle("Inferred component proportions across MAF thresholds", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_scatter_plot(compare_df: pl.DataFrame, *, axis_min: float, axis_max: float, output_path: Path) -> None:
    components = sorted(
        int(value) for value in compare_df.select("component").unique().get_column("component").to_list()
    )

    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=(4.8 * len(components), 4.8),
        constrained_layout=True,
    )
    if len(components) == 1:
        axes = [axes]

    scatter = None
    for ax, component in zip(axes, components, strict=False):
        comp_df = compare_df.filter(pl.col("component") == component)
        true_sigma2 = float(comp_df.get_column("true_component_sigma2")[0])
        median_closest_var0 = float(comp_df.select(pl.col("closest_inferred_var0").median())[0, 0])
        scatter = ax.scatter(
            comp_df.get_column("simulated_proportion").to_list(),
            comp_df.get_column("inferred_proportion").to_list(),
            c=comp_df.get_column("maf").to_list(),
            cmap="viridis",
            s=50,
        )
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "k:", linewidth=1)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_xlabel("Simulated proportion")
        ax.set_ylabel("Inferred proportion")
        ax.set_title(
            f"Component {component}\ntrue σ²={true_sigma2:.4g}, closest inferred var0~{median_closest_var0:.4g}"
        )
        ax.grid(alpha=0.25)

    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.9)
        colorbar.set_label("MAF threshold")

    fig.suptitle("Inferred vs simulated component proportions (per threshold)", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_summary(compare_df: pl.DataFrame) -> pl.DataFrame:
    return (
        compare_df.with_columns((pl.col("inferred_proportion") - pl.col("simulated_proportion")).abs().alias("abs_err"))
        .group_by("component")
        .agg(
            [
                pl.col("abs_err").mean().alias("mean_abs_error"),
                pl.col("abs_err").max().alias("max_abs_error"),
                pl.col("inferred_assigned_mass").mean().alias("mean_inferred_assigned_mass"),
                pl.col("true_component_sigma2").mean().alias("true_component_sigma2"),
                pl.col("closest_inferred_var0").mean().alias("mean_closest_inferred_var0"),
            ]
        )
        .sort("component")
    )


def _build_inferred_summary(infer_component_df: pl.DataFrame) -> pl.DataFrame:
    return (
        infer_component_df.group_by("component")
        .agg(
            [
                pl.col("var0").mean().alias("var0"),
                pl.col("inferred_proportion").mean().alias("mean_inferred_proportion"),
                pl.col("inferred_proportion").max().alias("max_inferred_proportion"),
            ]
        )
        .sort("component")
    )


def run_plotting(config: PlotConfig) -> dict[str, Path]:
    if not config.infer_path.exists():
        raise FileNotFoundError(f"infer file does not exist: {config.infer_path}")
    if config.axis_min >= config.axis_max:
        raise ValueError("axis_min must be less than axis_max")
    if config.maf_min < 0.0 or config.maf_min >= 1.0:
        raise ValueError("maf_min must satisfy 0 <= maf_min < 1")

    infer_df = pl.read_csv(config.infer_path, separator="\t")

    _validate_columns(
        infer_df,
        required={"maf", "var0", "value"},
        label="infer file",
    )

    infer_df = infer_df.with_columns(
        pl.col("maf").cast(pl.Float64),
        pl.col("var0").cast(pl.Float64),
        pl.col("value").cast(pl.Float64),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    compare_path = config.output_dir / f"{config.output_prefix}.tsv"
    summary_path = config.output_dir / f"{config.output_prefix}_summary.tsv"
    vs_maf_path = config.output_dir / f"{config.output_prefix}_vs_maf.png"

    if config.component_mode == "inferred":
        infer_component_df, skipped = _build_inferred_component_dataframe(infer_df, maf_min=config.maf_min)
        summary_df = _build_inferred_summary(infer_component_df)
        infer_component_df.write_csv(compare_path, separator="\t")
        summary_df.write_csv(summary_path, separator="\t")
        _render_inferred_components_vs_maf_plot(
            infer_component_df,
            axis_min=config.axis_min,
            axis_max=config.axis_max,
            output_path=vs_maf_path,
        )

        if skipped:
            print(f"skipped_thresholds={len(skipped)}")
            for maf, reason in skipped:
                print(f"  maf={maf:.6g} reason={reason}")

        return {
            "compare": compare_path,
            "summary": summary_path,
            "vs_maf_plot": vs_maf_path,
        }

    if config.truth_path is None:
        raise ValueError("--truth is required when --component-mode is 'matched'")
    if not config.truth_path.exists():
        raise FileNotFoundError(f"truth file does not exist: {config.truth_path}")

    truth_df = pl.read_csv(config.truth_path, separator="\t")
    _validate_columns(
        truth_df,
        required={"component", "sigma2", "effect_allele_frequency"},
        label="truth file",
    )
    truth_df = truth_df.with_columns(
        pl.col("component").cast(pl.Int64),
        pl.col("sigma2").cast(pl.Float64),
        pl.col("effect_allele_frequency").cast(pl.Float64),
    )
    compare_df, skipped = _build_comparison_dataframe(truth_df, infer_df, maf_min=config.maf_min)
    summary_df = _build_summary(compare_df)
    scatter_path = config.output_dir / f"{config.output_prefix}_scatter.png"

    compare_df.write_csv(compare_path, separator="\t")
    summary_df.write_csv(summary_path, separator="\t")
    _render_vs_maf_plot(compare_df, axis_min=config.axis_min, axis_max=config.axis_max, output_path=vs_maf_path)
    _render_scatter_plot(compare_df, axis_min=config.axis_min, axis_max=config.axis_max, output_path=scatter_path)

    if skipped:
        print(f"skipped_thresholds={len(skipped)}")
        for maf, reason in skipped:
            print(f"  maf={maf:.6g} reason={reason}")

    return {
        "compare": compare_path,
        "summary": summary_path,
        "vs_maf_plot": vs_maf_path,
        "scatter_plot": scatter_path,
    }


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    parser = argparse.ArgumentParser(description="Plot inferred vs simulated component proportions.")
    parser.add_argument(
        "--truth",
        type=Path,
        required=False,
        help="Path to simulation truth TSV (e.g., demo.truth.tsv).",
    )
    parser.add_argument("--infer", type=Path, required=True, help="Path to inference TSV (e.g., demo.infer.tsv).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/sim_compare_plots"),
        help="Directory for output TSV/PNG artifacts.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="component_proportions",
        help="Prefix for output artifacts.",
    )
    parser.add_argument(
        "--axis-min",
        type=float,
        default=0.0,
        help="Lower bound for proportion axes.",
    )
    parser.add_argument(
        "--axis-max",
        type=float,
        default=1.0,
        help="Upper bound for proportion axes.",
    )
    parser.add_argument(
        "--component-mode",
        choices=("matched", "inferred"),
        default="matched",
        help="`matched` compares to truth components; `inferred` plots all inferred non-null components.",
    )
    parser.add_argument(
        "--maf-min",
        type=float,
        default=1e-3,
        help="Ignore thresholds below this value.",
    )

    args = parser.parse_args(argv)
    return PlotConfig(
        truth_path=args.truth,
        infer_path=args.infer,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        axis_min=args.axis_min,
        axis_max=args.axis_max,
        maf_min=args.maf_min,
        component_mode=args.component_mode,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    outputs = run_plotting(config)
    for label, path in outputs.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
