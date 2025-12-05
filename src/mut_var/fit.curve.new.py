#! /usr/bin/env python
import argparse as ap
import sys


import jax.numpy as jnp
import polars as pl
import optimistix as optx
import matplotlib.pyplot as plt  # <-- added


def objective(coef, args):
    maf, value = args
    return curve(maf, coef) - value


def curve(maf, coef):
    left_asym, right_asym, rate = coef
    return left_asym + (right_asym - left_asym) / (1.0 + jnp.power(maf, rate))


def fit_curve(df_s):
    df_s = df_s.sort("maf")
    maf = jnp.asarray(df_s["maf"].to_numpy())
    value = jnp.asarray(df_s["value"].to_numpy())

    solver = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-3)
    result = optx.least_squares(
        objective,
        solver,
        y0=jnp.ones(3),
        args=(maf, value),
        max_steps=1000,
    )

    return result.value


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("data")
    argp.add_argument("--lowest", type=float, default=1e-10)
    argp.add_argument("--highest", type=float, default=1e-2)
    argp.add_argument("--num-breaks", type=int, default=10)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)
    df_r = pl.read_csv(args.data, separator="\t")

    for variance, df_sub in df_r.group_by("var0"):
        coef = fit_curve(df_sub)

        # Plot for each variance using the stored df_sub (no filtering)
        maf = jnp.asarray(df_sub["maf"].to_numpy())
        value = jnp.asarray(df_sub["value"].to_numpy())

        maf_space = jnp.linspace(float(maf.min()), float(maf.max()), 200)
        fitted_values = curve(maf_space, coef)

        plt.figure(figsize=(6, 4))
        plt.semilogx(maf, value, 'o', label="data")
        plt.semilogx(maf_space, fitted_values, '-', label="fit")
        plt.xlabel("MAF")
        plt.ylabel("Value")
        plt.title(f"var0 = {variance[0]}")
        plt.legend()
        plt.tight_layout()

        # Write to file
        out_name = f"{args.data}_{variance[0]:.6g}.png" if isinstance(variance, float) else f"{args.data}_{variance[0]}.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
