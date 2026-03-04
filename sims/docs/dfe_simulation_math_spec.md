# DFE-based simulation model for stabilizing selection inference

## Scope
This document defines the simulation used to generate GWAS-like summary statistics for stabilizing-selection analyses.

## Full model overview
For each locus $j$:
$$
s_{ud,j} \sim p_{\mathrm{DFE}}(s_{ud}),
\qquad
X_j \sim p_X(x\mid s_{ud,j}),
\qquad
\beta_{s,j} \sim p_{\beta}(\beta_s\mid s_{ud,j}),
$$
$$
\mathrm{SE}_j=\left(2X_j(1-X_j)n_{\mathrm{eff}}\right)^{-1/2},
\qquad
\hat\beta_{s,j}\mid \beta_{s,j},\mathrm{SE}_j \sim N\!\left(\beta_{s,j},\mathrm{SE}_j^2\right).
$$

Ascertainment indicator (default MAF mode):
$$
\mathbb{1}_j=\mathbb{1}\!\left\{\min(X_j,1-X_j)\ge m^*\right\}.
$$

Observed tuple exported to `mutvar infer`:
$$
\left(\texttt{effect\_allele\_frequency},\texttt{beta},\texttt{standard\_error}\right)_j
=
\left(X_j,\hat\beta_{s,j},\mathrm{SE}_j\right).
$$

## Core effect and scaling relationships
Notation:

- $s_{ud,j}$: underdominant selection coefficient,
- $S_{ud,j}=2N_e s_{ud,j}$: scaled underdominant coefficient,
- $X_j\in(0,1)$: effect-allele frequency,
- $\beta_{s,j}$: latent effect on the $\beta_s$ axis,
- $\hat\beta_{s,j}$: observed effect estimate,
- $\mathrm{SE}_j$: standard error,
- $n_{\mathrm{eff}}$: effective GWAS sample size controlling observation noise.

All beta quantities in this document are on the selection-scaled axis $\beta_s$ (not raw trait-scale effect units). In the one-dimensional limit, $\beta_s^2=S_{ud}$, which fixes the scale of both $\beta_s$ and $\hat\beta_s$.

Effect model under stabilizing-selection limits:

### High-dimensional (pleiotropic)
$$
\beta_{s,j}\mid s_{ud,j} \sim N\!\left(0,\;2N_es_{ud,j}\right)=N(0,S_{ud,j}).
$$

### One-dimensional (single-trait)
$$
\beta_{s,j}\mid s_{ud,j}
\in
\left\{-\sqrt{2N_es_{ud,j}},\,+\sqrt{2N_es_{ud,j}}\right\}
=
\left\{-\sqrt{S_{ud,j}},\,+\sqrt{S_{ud,j}}\right\},
$$
with equal sign probability.

Observation model:
$$
\mathrm{SE}_j = \left(2X_j(1-X_j)n_{\mathrm{eff}}\right)^{-1/2},
\qquad
\hat\beta_{s,j}\mid \beta_{s,j},\mathrm{SE}_j \sim N(\beta_{s,j},\mathrm{SE}_j^2).
$$

## Ascertainment
Default discovery mode is a MAF threshold:
$$
\min(X_j,1-X_j)\ge m^*.
$$

Here $v_s^*$ denotes the fixed variance-contribution cutoff used by variance-threshold ascertainment.

Optional alternative ascertainment modes use variance contribution thresholds with
$$
v_{s,j}=2X_j(1-X_j)\beta_{s,j}^2,
\qquad
\hat v_{s,j}=2X_j(1-X_j)\hat\beta_{s,j}^2,
$$
and keep loci by either $v_{s,j}>v_s^*$ (truth-side) or $\hat v_{s,j}>v_s^*$ (noisy-side).

If variance-threshold ascertainment is used with target $p^*$, the implied effective GWAS sample size is:
$$
n_{\mathrm{eff}}=\frac{2\,\operatorname{erf}^{-1}(1-p^*)^2}{v_s^*}.
$$

## Mutvar-facing observed output
Observed output is written for direct input to `mutvar infer` with exactly:

- `effect_allele_frequency`,
- `beta`,
- `standard_error`.

At each locus $j$, the observed columns map to latent quantities as:

- `effect_allele_frequency` $= X_j$,
- `beta` $= \hat\beta_{s,j}$,
- `standard_error` $= \mathrm{SE}_j$.

So the exported `beta` column is on the selection-scaled axis $\beta_s$.
The exported `standard_error` is on that same $\beta_s$ axis.

## Evaluation outputs
Primary endpoint is recovery of the truth-side distribution of $|\beta_s|$.

Reported quantities:

- absolute error in mean and variance of $\beta_s$,
- quantile and tail-mass error for $|\beta_s|$,
- KS and Wasserstein distances for $|\beta_s|$.

AF-conditioned summaries are optional diagnostics.

## Technical generation details
DFE sampling is performed on
$$
\ell=\log_{10}(s_{ud}),
$$
using an SSD table $\{(\ell_i,f_i)\}_{i=1}^m$ with interpolation-defined density
$$
p_\ell(\ell)\propto \operatorname{interp}(\ell;\{\ell_i,f_i\}).
$$

Allele frequencies are sampled from an underdominant SFS conditional on $s_{ud}$. In this implementation, the SFS kernel is parameterized with $2S_{ud}$, where $S_{ud}=2N_es_{ud}$:
$$
\log \tau(x\mid S_{ud})
=
\log\!\left(\frac{\theta}{x(1-x)}\right)
-(2S_{ud})x(1-x)
+\text{erf correction}(x,2S_{ud}).
$$
