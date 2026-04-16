# Model Symbol Table

| Symbol | Meaning | Source / Notes |
| --- | --- | --- |
| `mu_k` | Mixture component means | Fixed at log-spaced grid initialization in inference. |
| `var_k` | Mixture component variances | Fixed at log-spaced grid initialization in inference. |
| `pi` | Mixture weights on the probability simplex | Sole optimization variable in the pi-only refactor. |
| `L` | Likelihood matrix | Precomputed once from fixed `mu_k`, `var_k`, and input arrays. |
| `af` | Effect allele frequency array | Ingress field in `InferenceArrays`. |
| `beta_hat` | Effect-size estimate array | Ingress field in `InferenceArrays`. |
| `s2` | Squared standard-error array | Ingress field in `InferenceArrays`; derived from `standard_error^2`. |
| `InferenceArrays` | Validated numeric ingress payload | Lives in `mut_var.io` after Phase 1. |
| `InferenceConfig` | Pipeline inference configuration | Lives in `mut_var.types` after Phase 1. |
| `FitState` | Precomputed numerics state | Holds the likelihood matrix and initial parameters. |
| `Params` | Output parameter bundle | Carries fitted `pi` plus fixed `mu_k`/`var_k`. |
| `RESULTS` | Numerics status enum | Success/failure signal for numerics entrypoints. |
| `Solution` | Result wrapper | Wraps `value`, `result`, and optional `stats` / `state`. |
