# Layout-9 causal Fibonacci structural-feature paired-experiment evaluation precommit

## Status and scope

This document freezes the evaluation contract before outcomes for the first
layout-9 causal Fibonacci structural-feature seed-43 pilot pair. It does not
report or inspect a result, establish a numerical acceptance threshold, create
a composite score, or declare a winner rule.

The scientific question is:

> Does adding the frozen 23-feature causal Fibonacci structural representation
> change or improve out-of-sample model behavior relative to an otherwise
> identical model in which exactly those 23 features are zero-ablated?

This one fresh-initialization seed is pilot evidence only; it cannot by itself
support a general efficacy claim.

## Frozen pair identity

| Role | Experiment | Treatment |
|---|---:|---|
| CONTROL | 667 | Fibonacci enabled; `feature_ablation_mask` must be empty. |
| ABLATION | 666 | Fibonacci removed; `feature_ablation_mask` must equal the complete canonical 23-feature mask below. |

The pair-level constants are `symbol=cadchfrmp`, `prediction_horizon=4`,
`target_epochs=20`, `fresh_initialization_seed=43`,
`model_input_semantic_layout_version=9`, `model_input_width=103`, and
`training_objective_id=legacy_first_hit_weighted_ce_v1`.

The expected ablation mask is the concrete canonical persisted list in
`FeatureAblation.hpp`, not an alias:

```text
fib_recent_price_scale_valid,fib_up_recent_union_count_log,fib_up_recent_h1_count_log,fib_up_recent_h2_count_log,fib_up_recent_h1_h2_both_count_log,fib_up_recent_h1_youngest_age_20,fib_up_recent_h2_youngest_age_20,fib_up_recent_median_1272_signed_atr,fib_up_recent_median_1618_signed_atr,fib_up_recent_median_pullback_0382_signed_atr,fib_up_recent_median_pullback_0500_signed_atr,fib_up_recent_median_pullback_0618_signed_atr,fib_down_recent_union_count_log,fib_down_recent_h1_count_log,fib_down_recent_h2_count_log,fib_down_recent_h1_h2_both_count_log,fib_down_recent_h1_youngest_age_20,fib_down_recent_h2_youngest_age_20,fib_down_recent_median_1272_signed_atr,fib_down_recent_median_1618_signed_atr,fib_down_recent_median_pullback_0382_signed_atr,fib_down_recent_median_pullback_0500_signed_atr,fib_down_recent_median_pullback_0618_signed_atr
```

All reported deltas retain the comparison tool convention
`control_minus_ablation`: a positive delta means the Fibonacci-enabled control
is numerically higher for that metric. Higher is not automatically desirable;
each metric's meaning must be interpreted separately.

## Validity and readiness gate

Interpret no outcome metric until the existing feature-ablation comparison
machinery accepts the pair's identity and final evidence. In particular, it
must establish the empty control mask, exact parsed-and-canonicalized ablation
mask, otherwise matched scientific configuration, final-model input identity,
exact-final classification and profitability provenance, and matched scientific
execution provenance.

The pre-training intervention identity record is:

| Field | Frozen value |
|---|---|
| Expected ablation mask | The canonical 23-feature list above |
| `ablation_identity_hash` | `fnv1a64:a3f595680caadb2e` |
| Recorded `evaluation_identity_hash` expectation | `fnv1a64:a22abf3a2055567b` |

These hashes are provenance and identity evidence, never performance evidence.
The evaluator's `evaluation_identity_hash` includes final inference and
profitability observation identifiers plus disposition, so its emitted value
must be verified only from the completed comparison output; the recorded value
above is not a substitute for that gate.

Use the evaluator's existing terminology. A running pair is expected to be
`comparable_incomplete` and `not_ready`; only `comparable_complete` with
`pair_validity_state=valid` and `readiness_state=ready` supports the full
predeclared comparison. `invalid_ablation_pair`, `incompatible_configuration`,
`missing_final_inference`, `ambiguous_final_inference`, and
`profitability_evidence_unavailable` must be reported as such, not relabeled as
an efficacy result. Layout-9 Fibonacci evidence uses the existing generic
`generic_feature_ablation_evidence` classification.

## Predeclared final out-of-sample observations

The evaluator uses exact `final` inference and analysis evidence. The primary
model-performance observations are `leader_score`, `inference_accuracy`, and
`accept_accuracy`. They are reported arm-by-arm and as `control_minus_ablation`
deltas without post-hoc weighting or a forced overall conclusion.

The behavioral and coverage diagnostics are `accept_rate`,
`down_proportion`, `neutral_proportion`, and `up_proportion`. The output also
retains `prediction_count`, `predicted_down_count`, `predicted_neutral_count`,
`predicted_up_count`, and `accepted_prediction_count` to make those proportions
and coverage auditable. A change in these diagnostics is behavioral change,
not automatically performance improvement.

## Predeclared final profitability observations

When exact-final profitability evidence is available, report:

- `actionable_count`;
- `aggregate_terminal_horizon_log_return_sum`;
- `average_terminal_horizon_log_return_per_actionable_prediction`.

The persisted observation definition makes an actionable prediction a predicted
Up or Down with finite positive terminal prices. Its return is directional
terminal-horizon log return at the configured horizon; the per-actionable
average is absent when `actionable_count=0`. The authoritative persisted
evidence also contains winning and losing actionable counts, but winning,
losing, and actionable percentages are not predeclared pair-comparison
endpoints and will not be promoted selectively after results are known.

These observations are not deployable trading-profitability evidence by
themselves. They do not model transaction costs, slippage, position sizing,
leverage, overlap/capital constraints, or any other assumptions absent from the
persisted observation definition.

## Precommitted interpretation rules

- No metric threshold, numeric score, composite metric, weighted ranking, or
  winner rule may be invented after seed-43 outcomes are seen.
- No feature may be added, removed, renamed, or selectively excluded from the
  frozen intervention after outcomes; neither pair member may be replaced
  because another run appears preferable.
- Report every predeclared primary metric and diagnostic. Do not promote a
  favorable metric while silently omitting an unfavorable one.
- Distinguish performance improvement from behavioral or coverage change.
  When metrics disagree, report the disagreement rather than collapse it into
  an unsupported overall win/loss declaration.
- One seed is pilot evidence only. An interesting seed-43 result motivates
  replication, not tuning around seed 43.

## Replication boundary

Any future replication must preserve this exact scientific pair identity,
including control/ablation roles and the complete canonical mask, and change
only `fresh_initialization_seed`. Changing any other scientific setting or the
intervention declares a new study. This precommit creates no replication.
