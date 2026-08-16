# Causal 32-Bar RMS-Normalized Close-Return Surprise

## Implemented contract

Base column 37 is `clamp(r_t / sqrt(mean(r_j^2)), -10, 10)`, where
`r_t = log(close_t / close_(t-1))` and the reference contains at most 32 prior
valid close-to-close returns. This is RMS magnitude, deliberately without mean
subtraction. Invalid/non-finite/non-positive closes, bootstrap, no predecessor
returns, or a non-finite/non-positive denominator yield finite `0`.

## Causality

`CausalReturnSurprise32::AddCompletedClose` computes the result from its
existing return queue and rolling sum of squares before appending the current
valid return. Therefore changing `r_t` can change its numerator but cannot
change its denominator. After append, that return may influence future rows.

## Compatibility and provenance

The append-only layout is now 38 base columns and current `n_in=42`.
Historical projections remain `36 -> 32`, `38 -> 34`, `40 -> 36`, and
`41 -> 37`; current is `42 -> 38`. Recommendation semantic configuration
defaults to v9 to distinguish this input contract. Explicit v3-v8 parsing and
canonical reconstruction remain unchanged; v9 has the same persisted
configuration fields as v8 with a new canonical version prefix.

## Verification

- Focused causal-return-surprise helper/Tensor test covers bootstrap, validity,
  RMS formula, clamps, rolling eviction, causality, future dependency, and
  Tensor/model-vector placement.
- Model-input compatibility and feature-vector parity tests cover historical
  projections and current 42-width assembly.
- The requested isolated Release build was attempted with
  `DerivedData/ExpertAdvisor-next-feature`; its provenance phase correctly
  rejected the intentionally uncommitted working tree before compilation with
  `Release provenance requires a clean source tree`. No supported dirty-tree
  override exists, and no commit was created.
