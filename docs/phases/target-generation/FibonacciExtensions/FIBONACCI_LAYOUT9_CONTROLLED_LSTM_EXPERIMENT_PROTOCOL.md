# Causal Fibonacci layout-9 controlled LSTM experiment design

**Protocol identifier:** `causal-fibonacci-layout9-controlled-lstm-v1`

**Status:** **BLOCKED — NOT FROZEN BEFORE EXPERIMENT MATERIALIZATION**

**Scope:** outcome-blind design for the next scientific stage following the
completed `causal-fibonacci-layout9-incremental-information-v2` screen. This
document neither authorizes nor materializes an experiment.

## 1. Relationship to the completed screen

The completed incremental-information screen remains unchanged:

- protocol: `causal-fibonacci-layout9-incremental-information-v2`;
- protocol SHA-256:
  `9f39886dc46ede11e2322fd8c0afcef6b09505132bfdc243c85ae8b790352c53`;
- final classification: `ROBUST_INCREMENTAL_SCREENING_SIGNAL`.

That classification is a reason to propose this controlled LSTM experiment;
it is not an LSTM result, profitability result, production-readiness result,
or authorization to change a live model. The screen's `confirmation_2025`
holdout was consumed once and is permanently closed as unseen confirmation
data for that screen. This experiment must not describe, select, or interpret
any 2025 interval as a new unseen confirmation of the completed screen.

## 2. Scientific question and controlled intervention

The planned question is:

> Does providing the complete causal 23-column Fibonacci family to the actual
> layout-9 LSTM improve its predictive and/or existing downstream
> terminal-horizon directional-log-return observations, compared with an
> otherwise identical LSTM in which exactly that family is zero-ablated?

Each replication cell contains exactly these arms:

| Role | Name | `feature_ablation_mask` |
| --- | --- | --- |
| control | Fibonacci-complete | empty |
| ablation | Fibonacci-family-ablated | the exact canonical 23-name list below |

The comparison direction is always `control_minus_ablation`. A positive
delta means the complete-family control has a numerically higher value. That
is favorable for `leader_score`, `inference_accuracy`, `accept_accuracy`, and
the two logged return observations, but not automatically for coverage or
class-composition diagnostics.

`FeatureAblationMask` is applied only after the model-input Tensor-prefix
projection. It sets selected projected Tensor values to zero; it does not
delete columns. Therefore both arms retain physical layout 9 and width 103,
and ablation cannot change feature-row eligibility, target construction,
economic-calendar availability, feature warmup, or the sample population.

## 3. Exact frozen intervention identity

The complete family is the concrete, ordered
`FeatureAblation::kCausalFibonacciStructuralAblationMaskText` identity. Its
canonical text and existing identity hash are:

```text
fib_recent_price_scale_valid,fib_up_recent_union_count_log,fib_up_recent_h1_count_log,fib_up_recent_h2_count_log,fib_up_recent_h1_h2_both_count_log,fib_up_recent_h1_youngest_age_20,fib_up_recent_h2_youngest_age_20,fib_up_recent_median_1272_signed_atr,fib_up_recent_median_1618_signed_atr,fib_up_recent_median_pullback_0382_signed_atr,fib_up_recent_median_pullback_0500_signed_atr,fib_up_recent_median_pullback_0618_signed_atr,fib_down_recent_union_count_log,fib_down_recent_h1_count_log,fib_down_recent_h2_count_log,fib_down_recent_h1_h2_both_count_log,fib_down_recent_h1_youngest_age_20,fib_down_recent_h2_youngest_age_20,fib_down_recent_median_1272_signed_atr,fib_down_recent_median_1618_signed_atr,fib_down_recent_median_pullback_0382_signed_atr,fib_down_recent_median_pullback_0500_signed_atr,fib_down_recent_median_pullback_0618_signed_atr
```

```text
fnv1a64:a3f595680caadb2e
```

The names are persisted in canonical Tensor-column order. For layout 9 they
resolve to Tensor columns 76--98. Layout 9 has 99 Tensor columns and its
model-input contract is 103 wide (the Tensor columns plus four model-only
return inputs); `model_input_semantic_layout_version=9` and
`model_input_width=103` are mandatory for *both* arms. The ablation mechanism
has all-or-nothing support for this exact concrete list, and rejects a feature
that is absent from the persisted model-input layout.

No individual Fibonacci column, subset, alias, ratio, or regime may be
selected after results are known.

## 4. Specified experimental matrix pending provenance freeze

This is intentionally the full screening universe rather than a
post-confirmation subset. The order is canonical lexical order, not an effect
size ranking.

| Dimension | Specified value pending freeze |
| --- | --- |
| symbols | `audcadrmp`, `audusdrmp`, `eurusdrmp`, `gbpusdrmp`, `usdcadrmp`, `usdjpyrmp` |
| horizons | H4 and H6 |
| fresh initialization seeds | `43`, `47`, `53`, `59`, in that order |
| cells | 6 symbols × 2 horizons × 4 seeds = 48 matched pairs (96 arms) |
| training range | `[2010-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` |
| final inference range | `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)` |
| target epoch budget | exactly 20 epochs per arm |
| fresh start | `resume_model_id=NULL`; `resume_expand_input_width=false` |
| prediction threshold | `c_next_threshold=0.0008` |
| Tensor feature warmup | `full_history_warmup` |
| Donchian contract | `enabled`, lookback 20 |
| base learning rate | `1e-3 / 3` |
| core LR multiplier | `120` |
| head weight LR multiplier | `25` |
| head bias LR multiplier | `2.5`, verified from final `train_config_meta` |
| batch size | 256 |
| window size | 64 |
| hidden size / layers | 64 / 1 |
| normalization version | 1 |
| target / labels | existing up-neutral-down first-hit target, classes Down=0, Neutral=1, Up=2 |
| training objective | `legacy_first_hit_weighted_ce_v1`, canonical objective hash `fnv1a64:65818f2e1fa1a324`; auxiliary loss disabled |
| checkpoint interval | 20; no checkpoint inference or checkpoint policy |
| continuation policy | disabled; no continuation experiment is part of this protocol |
| final evidence | exact `final` inference, profitability observation, and analysis only; checkpoint evidence is not an endpoint |

The initial 20-epoch budget is fixed for every arm. Existing continuation
machinery makes per-experiment decisions from performance evidence, so it is
not a safe paired continuation mechanism for this feature-family study. No
extension to 60 or 80 epochs is permitted by this protocol, even if either
arm appears attractive. A future paired-continuation study would require a
new protocol and a symmetric, outcome-blind materialization mechanism.

The proposed inference range ends before `2025-01-01`. It is intentionally
not a re-use of the completed screen's 2025 confirmation as a new LSTM
holdout. The 2022--2024 observations are a new, explicitly named LSTM
evaluation range, not an unseen confirmation claim for the completed screen.

## 5. Pairing, materialization, and execution identity

For a fixed symbol, horizon, and seed, control and ablation must have the same
fresh initialization seed. No run from another seed may replace a failed or
missing arm, and unmatched arms must not enter a paired result.

The existing feature-ablation comparator must be invoked in its new explicit
mode with the expected complete 23-name mask:

```text
--compare-feature-ablation-pair=CONTROL_ID:ABLATION_ID
--expected-ablation-mask=<the canonical 23-name text in section 3>
```

The replication comparator likewise requires the same explicit mask. It
validates empty control mask, exact canonical ablation mask, matched
scientific configuration, final model identity, exact-final inference and
profitability evidence, and scientific execution provenance. It renders no
winner, ranking, recommendation, or database write.

Each completed pair must have matching training execution provenance and
matching final-inference execution provenance: semantic layout, executable
SHA-256, and runtime identity. The source commit and canonical manifest path
remain audit locators. A mismatch is invalid evidence, not a treatment effect.

The ablated arm requires the registered training capability
`train_feature_ablation_v1`. Before any future materialization, the
operational semantic-worker registry must prove that the control and ablated
layout-9/103 train selections resolve to the same immutable executable
SHA-256 and runtime identity; the final inference selections must satisfy the
same equality. The ordinary empty-mask control must not silently route to a
different binary merely because it does not itself request the ablation
capability.

The 2026-09-28 provenance-only qualification established that this condition
is **not currently met**. The checked operational registry
`Builds/SemanticWorkers/registry.json` (SHA-256
`2d874267cd24cb1da0f9c32ab7357cfe676cb9336337ac30ca2729c92cf86ac3`)
contains exactly one layout-9/103 TRAIN candidate:

| Field | Registered value |
| --- | --- |
| source commit | `273059036f0c1432082b32cfddbb6d9dc2285d16` (`Add causal Fibonacci feature ablation support`) |
| executable SHA-256 | `4657767524a8effc9e2610ae8fd2bd8cf7ad0b3281015d890731ccd566f15495` |
| runtime identity | `6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e` |
| capabilities | `train`, `infer`, `analyze` |
| required but absent capability | `train_feature_ablation_v1` |

The executable file's SHA-256 was independently recomputed and matches the
manifest. This does **not** qualify it for treatment execution: the scheduler
contract deliberately never infers feature-ablation capability from a source
commit, layout, or executable name. Consequently the complete arm resolves
to this candidate, while the nonempty exact-23 mask is rejected before worker
reservation with `semantic_worker_capability_incompatible`.

The existing selector has a useful general property once a common qualified
candidate is published: a capability-qualified TRAIN candidate with no
unrelated excess capabilities is selected for an otherwise-identical
empty-mask control as well as for the masked treatment. The disposable
semantic-worker-registry regression proves that rule and also proves
fail-closed rejection when the capability is absent. It is not a substitute
for an actual registered layout-9 qualified candidate. No registry entry,
binary, or historical worker artifact was changed in this qualification.

## 6. Required immutable calendar provenance — resolved

Both arms of every pair must bind the same non-null
`economic_calendar_snapshot_id` and matching immutable
`economic_calendar_snapshot_hash`. The snapshot must cover the proposed
train/inference ranges and be recorded before either arm is materialized.

Every arm is frozen to this exact immutable economic-calendar identity:

| Field | Frozen value |
| --- | --- |
| `economic_calendar_snapshot_id` | `1` |
| `economic_calendar_snapshot_hash` | `fnv1a64:67610f94f5c8e7cc` |
| state / hash contract | `finalized` / `1` |
| canonical event / consensus counts | `2601` / `1521` |
| release-actual / first-release rows | `1230` / `2601` |
| first-release provenance states | `1230` `proven_first_release`; `1371` `provenance_unavailable`; `0` `ambiguous` |
| event timestamp range | 2010-01-07 through 2026-08-20 UTC-offset timestamps |
| proven first-release availability range | 2010-01-08 through 2026-08-14 UTC-offset timestamps |

This identity was verified by a read-only query against the authoritative
database. The persisted child-table counts equal the finalized header counts;
the range covers both this protocol's `[2010-01-01, 2022-01-01)` training
range and `[2022-01-01, 2025-01-01)` final-inference range. The schema's
snapshot trigger makes finalized header rows immutable; child content is
immutable; and the experiment/model binding triggers require the exact
finalized ID/hash pair. `EconomicEventRepository::InspectEconomicCalendarSnapshot`
also recomputes and requires the stored canonical report/hash before runtime
use. The snapshot contains the model-facing point-in-time economic-event,
consensus, actual-release, and first-release-provenance semantics used by
layout 9, rather than a live calendar lookup.

Materialization, when separately authorized and only after section 5's worker
blocker is resolved, must persist this exact non-null ID/hash on every arm.
No queue-time snapshot creation or reuse decision is authorized for this
protocol.

## 7. Evaluation endpoints

All endpoints use complete final evidence only. They are retained per
symbol/horizon/seed pair; raw rows and unmatched arms are never pooled.

Predictive co-primary observations are, without weighting or composite score:

1. final `leader_score` from `experiment_analysis_result`;
2. final `inference_accuracy` / `infer_accuracy`;
3. final `accept_accuracy`.

Behavioral and coverage diagnostics are `accept_rate`, predicted
Down/Neutral/Up proportions and counts, accepted prediction count, and total
prediction count. They explain a change in model behavior but are not a
selective substitute for a predictive endpoint.

Profitability is secondary and corroborative, not a training objective. From
the exact-final `inference_profitability_observation`, report:

1. `actionable_count`;
2. `aggregate_terminal_horizon_log_return_sum`;
3. `average_terminal_horizon_log_return_per_actionable_prediction`.

The last value is `NULL` when `actionable_count=0`; it must remain unavailable
and cannot be replaced with zero or a favorable proxy. These observations are
the repository's terminal-horizon directional log-return observations, not
portfolio P&L: they omit transaction costs, slippage, sizing, leverage,
capital constraints, and overlap modeling. No minimum actionable-count gate
is introduced by this protocol.

## 8. Outcome-blind interpretation states

First establish integrity. Any missing/ambiguous final evidence, failed arm,
non-identical paired seed/configuration/snapshot, execution-provenance
mismatch, or row-population mismatch yields `INCOMPLETE_OR_INVALID`; no
efficacy classification is made and no replacement run is permitted.

For complete valid evidence, summarize every pair and, separately for each
symbol/horizon, its four seed deltas. Use unweighted descriptive summaries
(count, sign count, mean, median, minimum, and maximum). Do not calculate
p-values or infer IID independence.

The predictive states are:

- `REPLICATED_LSTM_FIBONACCI_BENEFIT`: at **each** horizon, at least four of
  the six symbols have at least three of four seed pairs with strictly positive
  deltas on **all three** predictive co-primary observations. The state cannot
  be supplied by a single symbol, seed, horizon, or metric.
- `MIXED_LSTM_FIBONACCI_EVIDENCE`: complete valid evidence contains both
  favorable and unfavorable/zero predictive behavior, or favorable behavior
  that does not meet the replicated-benefit rule.
- `NO_REPLICATED_LSTM_FIBONACCI_BENEFIT`: complete valid evidence does not
  meet either of the preceding conditions.

Profitability receives a separate, secondary statement. It is
`REPLICATED_PROFITABILITY_CORROBORATION` only if, at each horizon, at least
four symbols have at least three of four strictly positive deltas for both
aggregate and per-actionable return and all relevant per-actionable values are
available. Otherwise it is `NO_REPLICATED_PROFITABILITY_CORROBORATION` or
`PROFITABILITY_EVIDENCE_UNAVAILABLE`, as applicable. Profitability cannot
upgrade a mixed/no predictive state, and a predictive state cannot itself be
called profitable.

None of these states proves a causal economic mechanism, production readiness,
or a trading strategy.

## 9. Adversarial design audit

- **2025-selection leakage:** the matrix is the entire six-symbol/H4/H6
  screen universe in lexical order; no symbol or horizon is chosen by a 2025
  effect magnitude. The LSTM inference interval excludes 2025.
- **Treatment isolation:** the only intended configured difference is empty
  versus the exact 23-column canonical mask. Width/layout remain 103/9 and
  masking follows projection, preserving population and target semantics.
- **Seed and stopping leakage:** every cell has paired predeclared seeds;
  there is no result-driven continuation, extension, replacement, or arm
  substitution.
- **Worker-routing confounding:** the existing comparator fails a mismatch,
  and the present registry fails closed for treatment because no layout-9
  candidate declares `train_feature_ablation_v1`. Identical selected
  bytes/runtime remain a pre-materialization requirement; this unresolved
  condition keeps the protocol blocked.
- **Failure handling:** incomplete or failed pairs are explicit invalid or
  incomplete evidence, never silently omitted.
- **Metric discretion:** all predictive, behavioral, and profitability fields
  and their roles are named before results; no composite score, p-value, or
  post-hoc threshold is authorized.
- **Profitability overstatement:** persisted returns are accurately named
  terminal-horizon directional log-return observations and remain secondary.

## 10. Blockers and required next action

This design is **BLOCKED** and must not be materialized. The
economic-calendar blocker is resolved by the exact section 6 snapshot
identity. The remaining blocker is operational worker qualification: no
registered layout-9/103 TRAIN artifact declares
`train_feature_ablation_v1`, so the masked treatment cannot be admitted and
no same-byte complete/treatment pair can be proven.

No scientist or operator may resolve this blocker by looking at outcomes. A
future authorized operational qualification must publish a new immutable
layout-9/103 TRAIN registry entry that explicitly declares
`train_feature_ablation_v1`, verify that both masks deterministically select
the same executable SHA-256 and runtime identity, and bind that identity here
before the 96-arm matrix can be materialized. It must not mutate the historical
artifact above or infer the missing capability retrospectively. Until then,
this document is a design and blocker record, not execution authority.
