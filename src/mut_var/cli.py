import argparse as ap
import logging
import sys

from typing import NamedTuple

import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rdm
import numpy as np
import polars as pl

from jax.scipy.special import xlogy, logsumexp
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike


def get_logger(name, path=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)

        log_format = "[%(asctime)s - %(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console.setFormatter(formatter)

        if path is not None:
            disk_log_stream = open("{}.log".format(path), "w")
            disk_handler = logging.StreamHandler(disk_log_stream)
            logger.addHandler(disk_handler)
            disk_handler.setFormatter(formatter)

    return logger


def _logpdf(beta_hat, s2, mean, var_k):
    return norm.logpdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


logpdf = jax.vmap(_logpdf, (None, None, 0, 0), 1)


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


pdf = jax.vmap(_pdf, (None, None, 0, 0), 1)


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


def baseline_objective(
    param: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
):
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))
    pi = param.pi
    log_likelihood = jnp.sum(
        jnp.log(
            pdf(beta_hat, s2, param.mu_k, param.var_k) @ pi[1:]
            + _pdf(beta_hat, s2, 0.0, 0.0) * pi[0]
        )
    )

    return log_likelihood - log_penalty

def baseline_objective_lse(
    param: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
):
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))
    pi = param.pi
    log_likes = jnp.concatenate((
        _logpdf(beta_hat, s2, 0.0, 0.0)[:, jnp.newaxis],
        logpdf(beta_hat, s2, param.mu_k, param.var_k),
        ), axis=1)
    lse = logsumexp(log_likes, axis=1, b=pi)
    log_likelihood = jnp.sum(lse)


    return log_likelihood - log_penalty


def penalized_objective(
    param: Params,
    likelihoods: ArrayLike,
    alpha: ArrayLike,
    baseline_param: Params,
    penalty: ArrayLike,
):

    # marginal likelihood
    obj_term = jnp.sum(jnp.log(likelihoods @ param.pi))

    # log penalty on mixture weights
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))

    # penalty on neighboring model's mixture weights
    p1 = jnp.sum(
        nn.relu(
            baseline_param.pi[1:] * param.pi[:-1]
            - baseline_param.pi[:-1] * param.pi[1:]
        )
    )
    rel_point_mass_dist = nn.relu(param.pi[0] / baseline_param.pi[0] - 1)

    penalty_term = penalty * (p1 + rel_point_mass_dist)

    return obj_term - log_penalty - penalty_term


def _fix_var(var):
    eps = jnp.finfo(float).eps
    inf = ~jnp.isfinite(var)
    zs = var == 0.0
    return jnp.where(jnp.logical_or(inf, zs), eps, var)


def _exponential_map_normal(
    mu0: Array, v0: Array, mu_direction: Array, v_direction: Array, step_size: float
) -> tuple[Array, Array]:
    """
    Compute the closed-form geodesic in variance coordinates (mu, v)
    for the normal family with Fisher metric
        ds^2 = (1/v) dmu^2 + (1/(2*v^2)) dv^2,
    where v = sigma^2.

    **Arguments:**

    - `mu0`: initial mu-coordinate.
    - `v0`: initial variance (v0 > 0).
    - `direction`: a 2D vector (d_mu, d_v) representing the initial tangent
        direction
    - `step_size`: the step-size parameter (can be scalar or an array of values).

    **Returns:**
      (mu, v)   : the coordinates along the geodesic using step_size.
    """
    # Recover the angle theta from the tangent vector.
    std_dev = jnp.sqrt(v0)
    theta = jnp.arctan2(v_direction / jnp.sqrt(2.0), mu_direction / std_dev)

    a = step_size / jnp.sqrt(2.0)
    tanh_a = jnp.tanh(a)
    denom = 1.0 - jnp.sin(theta) * tanh_a

    mu_step = jnp.sqrt(2.0) * std_dev * jnp.cos(theta) * tanh_a / denom
    # if the step is nan, its due to precision, and we wouldn't move far anyway
    mu = jnp.where(jnp.isnan(mu_step), mu0, mu0 + mu_step)

    denom_sq = jnp.square(jnp.cosh(a) * denom)
    # shouldnt happen, but if variance is smaller than machine precision, cap it there
    v = _fix_var(v0 / denom_sq)

    return mu, v


def _exponential_map_simplex(
    pi: Array,
    direction: Array,
    step_size: float,
) -> Array:
    s = jnp.sqrt(jnp.sum(direction**2) / pi)
    c = jnp.cos(0.5 * step_size * s)
    s2 = jnp.sin(0.5 * step_size * s)

    phi = jnp.sqrt(pi)
    step = (direction / (s * phi)) * s2
    phi_new = phi * c + step
    pi_new = phi_new**2

    return pi_new / jnp.sum(pi_new)  # numerical safety


def _reimannian_step(
    params: Params,
    direction: Params,
    step_size: float,
) -> Params:
    tangent_pi = params.pi * (direction.pi - (direction.pi @ params.pi))
    pi = _exponential_map_simplex(params.pi, tangent_pi, step_size)

    # geodesics on the half-plane
    tangent_var_k = 2 * direction.var_k * params.var_k**2
    tangent_mu_k = direction.mu_k * params.var_k
    mu_k, var_k = _exponential_map_normal(
        params.mu_k, params.var_k, tangent_mu_k, tangent_var_k, step_size
    )

    return Params(pi, mu_k, var_k)


def _filter_df(
    data: pl.DataFrame,
    maf_threshold: float,
    af_name: str,
    beta_name: str,
    se_name: str,
):
    subset = data.filter(
        (pl.col(af_name).is_between(maf_threshold, 1 - maf_threshold, closed="both")) &
        (pl.col(se_name) > 0)
    )
    beta_hat = jnp.asarray(subset[beta_name].to_numpy())
    std_err = jnp.asarray(subset[se_name].to_numpy())

    return beta_hat, std_err


def fit_baseline_mixture(
    data: pl.DataFrame,
    num_clusters: int,
    maf_threshold: float,
    key: rdm.PRNGKey,
    batch_size: int = 10_000,
    max_iter: int = 100,
    tol: float = 1e-3,
    step_size: float = 0.999,
    af_name: str = "effect_allele_frequency",
    beta_name: str = "beta",
    se_name: str = "standard_error",
):
    beta_hat, std_err = _filter_df(data, maf_threshold, af_name, beta_name, se_name)
    s2 = std_err**2

    alpha = jnp.array([10.0] + (num_clusters - 1) * [1.0])
    min_val = jnp.min(std_err) / 10
    max_val = jnp.max(beta_hat**2 - s2)
    if max_val < 0.0:
        max_val = 8 * min_val
    else:
        max_val = 2 * jnp.sqrt(max_val)

    params = Params(
        pi=rdm.dirichlet(key, alpha),
        mu_k=jnp.zeros(num_clusters - 1),
        var_k=jnp.exp(
            jnp.linspace(jnp.log(min_val), jnp.log(max_val), num_clusters - 1)
        )
        ** 2,
    )

    vg_f = jax.jit(jax.value_and_grad(baseline_objective))
    obj = jax.jit(baseline_objective)
    ologlike = -1e10
    init_ss = step_size
    nobs = len(beta_hat)
    using_sgd = batch_size < nobs
    for epoch in range(max_iter):
        step_size = jnp.power(init_ss, epoch)
        if using_sgd:
            key, skey = rdm.split(key)
            idxs = rdm.choice(skey, nobs, shape=(batch_size,), replace=False)
            _, direction = vg_f(params, beta_hat[idxs], s2[idxs], alpha)
            direction = jax.tree.map(lambda x: (nobs / batch_size) * x, direction)
            _params = _reimannian_step(params, direction, step_size)
            nloglike = obj(_params, beta_hat[idxs], s2[idxs], alpha)
        else:
            _, direction = vg_f(params, beta_hat, s2, alpha)
            _params = _reimannian_step(params, direction, step_size)
            nloglike = obj(_params, beta_hat, s2, alpha)
        """
        for inner in range(20):
            nloglike = obj(_params, beta_hat, s2, alpha)
            diff = nloglike - ologlike
            if diff < 0 or jnp.isnan(nloglike) or jnp.isinf(nloglike):
                step_size *= 0.5
            else:
                break
        if inner < 19:  # 19 is 20 steps
            params = _params
            ologlike = nloglike
        """
        print(f"LL[{epoch}] = {nloglike} @ step-size = {step_size}")
        #if jnp.fabs(diff) < tol or jnp.isnan(diff) or inner == 19:
        #    break

    return params


def fit_mixture(
    data: pl.DataFrame,
    num_clusters: int,
    maf_threshold: float,
    init: Params,
    key: rdm.PRNGKey,
    penalty: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-3,
    step_size: float = 0.01,
    af_name: str = "effect_allele_frequency",
    beta_name: str = "beta",
    se_name: str = "standard_error",
):
    beta_hat, std_err = _filter_df(data, maf_threshold, af_name, beta_name, se_name)
    s2 = std_err**2

    alpha = jnp.array([10.0] + (num_clusters - 1) * [1.0])
    mu_k = jnp.pad(init.mu_k, (1, 0))
    var_k = jnp.pad(init.var_k, (1, 0))

    import copy

    params = copy.deepcopy(init)
    likelihoods = pdf(beta_hat, s2, mu_k, var_k)

    vg_f = jax.jit(jax.value_and_grad(penalized_objective))
    obj = jax.jit(penalized_objective)
    ologlike = -1e10
    start = step_size
    for epoch in range(max_iter):
        nloglike, direction = vg_f(params, likelihoods, alpha, init, penalty)
        step_size = start
        for inner in range(20):
            tangent_pi = params.pi * (direction.pi - (direction.pi @ params.pi))
            pi = _exponential_map_simplex(params.pi, tangent_pi, step_size)
            _params = Params(pi, params.mu_k, params.var_k)
            nloglike = obj(_params, likelihoods, alpha, init, penalty)
            diff = nloglike - ologlike
            if diff < 0 or jnp.isnan(nloglike) or jnp.isinf(nloglike):
                step_size *= 0.5
            else:
                break
        if inner < 19:  # 19 is 20 steps
            params = _params
            ologlike = nloglike
        print(f"LL[{epoch}] = {nloglike} @ step-size = {step_size}")
        if jnp.fabs(diff) < tol or jnp.isnan(diff) or inner == 19:
            break

    return Params(pi, init.mu_k, init.var_k)


def _main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("sumstats")
    argp.add_argument("-t", "--maf-threshold", type=float, default=0.01)
    argp.add_argument("-k", "--num-clusters", type=int, default=30)
    argp.add_argument("-m", "--max-iter", type=int, default=100)
    argp.add_argument("-r", "--step-size", type=float, default=0.01)
    argp.add_argument("-s", "--seed", type=int, default=0)
    argp.add_argument("-f", "--filter", type=float, default=1e-8)
    argp.add_argument("--lowest", type=float, default=1e-5)
    argp.add_argument("--highest", type=float, default=1e-2)
    argp.add_argument("--num_breaks", type=int, default=10)
    argp.add_argument("--penalty", type=float, default=1.0)
    argp.add_argument("--af-col", type=str, default="effect_allele_frequency")
    argp.add_argument("--beta-col", type=str, default="beta")
    argp.add_argument("--se-col", type=str, default="standard_error")
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    jax.config.update("jax_enable_x64", True)
    key = rdm.PRNGKey(args.seed)
    key, baseline_key, mixture_key = rdm.split(key, 3)

    # get data and put into groups based
    df_d = pl.read_csv(args.sumstats, separator="\t")

    baseline = fit_baseline_mixture(
        df_d,
        args.num_clusters,
        0.00,
        baseline_key,
        max_iter=args.max_iter,
        step_size=args.step_size,
        af_name=args.af_col,
        beta_name=args.beta_col,
        se_name=args.se_col,
    )
    keep = baseline.pi > args.filter
    keep = keep.at[0].set(True)
    filtered = Params(
        pi=baseline.pi[keep] / sum(baseline.pi[keep]),
        mu_k=baseline.mu_k[keep[1:]],  # mean is 0 for 0th component
        var_k=baseline.var_k[keep[1:]],  # no variance for 0th component
    )
    models = [filtered]

    maf_space = jnp.exp(
        jnp.linspace(jnp.log(args.lowest), jnp.log(args.highest), args.num_breaks)
    )
    for maf in maf_space:
        print(f"Fitting mixture @ AF {maf}")
        model = fit_mixture(
            df_d,
            models[-1].pi.shape[0],
            maf,
            models[-1],
            mixture_key,
            penalty=args.penalty,
            max_iter=args.max_iter,
            step_size=args.step_size,
            af_name=args.af_col,
            beta_name=args.beta_col,
            se_name=args.se_col,
        )
        models.append(model)

    min_af = df_d[args.af_col].min()
    max_af = df_d[args.af_col].max()
    maf = np.array(min(min_af, 1 - max_af))
    out = dict()
    maf_space = np.concatenate((maf[np.newaxis], np.asarray(maf_space)))
    names = [f"pi{idx}" for idx in range(len(models))]
    for idx, model in enumerate(models):
        if idx == 0:
            out["mu0"] = np.asarray(jnp.pad(model.mu_k, (1, 0)))
            out["var0"] = np.asarray(jnp.pad(model.var_k, (1, 0)))
        out[f"pi{idx}"] = np.asarray(model.pi)

    df_b = pl.DataFrame(out)
    df_maf = pl.DataFrame({"name": names, "maf": maf_space})

    df_long = df_b.melt(
        id_vars=["mu0", "var0"],
        value_vars=names,
        variable_name="name",
        value_name="value",
    )
    df_joined = df_long.join(df_maf, on="name", how="left")
    df_joined = df_joined.select(["mu0", "var0", "maf", "name", "value"])
    df_joined.write_csv(args.output, separator="\t")

    return 0


def run_cli():
    return _main(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

