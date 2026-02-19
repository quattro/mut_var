# MutVar Whole Project Hardening Design Plan

## Scope Summary

- Boundary validation and deterministic CLI failures
- Modular numerics/adapters architecture with canonical pipeline entrypoints
- Curve workflow split into fit-only numerics and optional plotting adapter
- Performance hardening with cache, profiling, and benchmark gate
- CI and regression hardening for contracts and stability

## Algorithm Scope Constraint

This plan excludes wholesale objective/model redesign. Algorithm adjustments must remain targeted and
justified by regression and benchmark evidence.

## Human Sign-Off Links

- Benchmark representativeness: `docs/reviews/benchmark-representativeness.md`
- Algorithm scope sign-off: `docs/reviews/algorithm-scope-signoff.md`
