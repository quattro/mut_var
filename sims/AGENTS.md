# sims agent rules

## Scope and boundaries
- Only create or edit files under `sims/`.
- Do not modify files outside `sims/` unless the user explicitly asks.
- Keep method/inference code in `src/` read-only.
- Do not run extra workflows beyond what the user requested.

## Output locations
- Keep generated outputs under:
  - `sims/results/`
  - `sims/plots/`
  - `sims/docs/docs_pdf/` (for rendered docs PDFs)

## Plotting rules (required)
- Use PDF outputs only for diagnostic/simulation plots unless the user explicitly requests other formats.
- Keep panel sizes small: no panel dimension may exceed `8` inches.
- Use relatively bold lines and large, clear axis labels/ticks.
- Do not use filler titles; titles must be concise and informative.
- Use bespoke, colorblind-friendly palettes; do not use default Matplotlib color cycles.
- Legends must never overlap plotted curves or points.
  - Place legends outside the axes when needed to guarantee no overlap.
- For symmetric beta distributions, do not show redundant negative-side plots.
  - Prefer positive-side summaries (for example `|beta|`).
- For beta-distribution comparisons across MAF cutoffs, include a log-scale view (for example `log10(|beta|)`) to improve separability.
- For DFE pre-beta plots, label the x-axis clearly as log-S (for example `log10(S_ud)`), not abstract symbols that reduce readability.

## Workflow discipline
- If the user asks to run sanity checks only, do not run extra grid/cutoff workflows unless explicitly requested.
- Before running inference-oriented tests, simulation-only diagnostics may be generated if requested.

## Language rule
- Banned term: the s-word.
- Do not use this term in chat responses, scripts/messages, plot titles/labels, docs, or logs.
- Preferred replacements: `MAF-threshold runs`, `MAF-cutoff set`, `MAF configuration set`.

## Math notation rules
- In equations, do not place a probability density/mass function expression on the right-hand side of a `cases` branch for a random variable value.
- If zero is part of the same conditional law, write a single distribution statement such as
  - `\beta_{s,j}\mid s_{ud,j} \sim p_\beta(\beta_s\mid s_{ud,j})`
  - plus an explicit atom statement `\Pr(\beta_s=0\mid s_{ud})=p_0`.
- Use mixture decompositions only when mathematically explicit and internally consistent (atom + continuous part), with no duplicated zero specification across separate definitions.

## User math primacy protocol (required)
- User-stated mathematical intent is authoritative and takes precedence over stylistic rewrites.
- If user intent and existing prose/equations conflict, rewrite equations to match user intent first, then align surrounding prose.
- Do not introduce alternative mathematical formulations unless the user asks for alternatives.
- Before finalizing any math-doc edit, run this consistency check:
  1. Random-variable statements (`\sim`) map to valid distributions, not sampled values/densities mixed incorrectly.
  2. Point masses/atoms are represented exactly once in the model definition unless explicitly decomposed by request.
  3. Symbols and prose agree on what belongs to the full law versus a component law.
  4. No sentence implies a separation that the user explicitly rejected.

## Failure-prevention addendum (required)
- Lock user intent before editing: restate in one sentence what must appear, what must not appear, and where it belongs.
- Respect section scope: keep model-overview equations generic unless the user explicitly asks for model-specific parameterization details there.
- Perform a final contradiction scan: remove any sentence that reintroduces a framing the user rejected.

## Math-spec placement protocol (required)
- Treat the methods math spec as layered by semantic role, and place changes only in the matching layer.
- Layer 1 — `Model overview`: only the canonical generative law and observation law used across scenarios.
- Layer 2 — `Core effect and scaling relationships`: invariant definitions/identities and symbol mappings.
- Layer 3 — scenario or family-specific parameterizations: place under a dedicated subsection (for example practical DFE parameterization), not in the overview, unless the user explicitly elevates it to core model status.
- Layer 4 — `Technical generation details`: numerical/discretization/interpolation mechanics only.
- Placement decision rule before editing:
  1. Ask: “Does this change alter the canonical generative law for all runs?”
  2. If yes, update Layer 1 and harmonize downstream layers.
  3. If no, keep Layer 1 unchanged and add/update the appropriate lower-layer subsection.
  4. Do not move a lower-layer detail upward without explicit user instruction.
