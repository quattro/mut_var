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
