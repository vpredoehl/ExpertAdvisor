---
title: "LSTM Campaign Manager Phase 3 Profitability Integration Design Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3_ProfitabilityIntegration_Design_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3 Profitability Integration Design Inspection

# 1. Executive Summary

The current repository calculates trading-return diagnostics during both final and checkpoint inference, but it does not persist them. Consequently, there is no authoritative persisted profitability evidence that the modern recommendation/campaign pipeline can safely consume.

The safest next implementation is **Option 1: behavior-neutral Phase 3A profitability plumbing first**, with strict conditions:

- Persist an immutable, explicitly scoped inference profitability observation.
- Load only `inference_scope='final'` observations into modern recommendations.
- Snapshot exact profitability provenance in recommendations and evaluations.
- Expose it diagnostically.
- Do not add a profitability score component, even with zero weight.
- Do not populate the existing campaign-admission `profitabilityMetric` field yet.
- Leave `minimumProfitability` inactive until a separate policy-activation phase.
- Do not modify checkpoint policy, continuation logic, legacy meta-analysis, conversion, materialization, or Campaign Manager dispatch.

Two existing defects are confirmed but do not block observation-only Phase 3A:

1. Broad ranking scopes can combine evaluations from different scoring policies and multiple evaluation runs.
2. Final analysis rows are mutable under stable `analysis_id`, while recommendation staleness validation compares identity rather than current metric content.

Both must be fixed **after Phase 3A but before profitability becomes decision-bearing**, whether through a nonzero score weight or `minimumProfitability`.

The repository does not define a suitable canonical cross-symbol/cross-horizon profitability metric. Existing inference code computes gross terminal-horizon log-return statistics without transaction costs, spread, slippage, capital allocation, overlap management, or risk normalization. Those primitives can be persisted in Phase 3A, but none should yet be called the canonical campaign profitability metric.

# 2. Phase 2 Finding Reverification

| # | Classification | Reverification |
|---|---|---|
| 1 | **CONFIRMED** | Modern candidates are single-parameter mutations of source experiment configurations. `GenerateRecommendationCandidates` constructs configurations and metadata only; no candidate model or inference exists until later conversion creates an experiment. See [ExperimentRecommendationCandidateGenerator.cpp:342](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:342>) and the eventual paused-experiment insertion in [ExperimentRecommendationConversionExecutionRepository.cpp:197](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationConversionExecutionRepository.cpp:197>). |
| 2 | **CONFIRMED** | Scoring combines source final-analysis evidence with structural/mutation heuristics. It is explicitly described as deterministic advisory prioritization, not predicted profitability, in [ExperimentRecommendationScoringService.cpp:422](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoringService.cpp:422>). |
| 3 | **CONFIRMED** | Recommendation source loading joins the experiment’s `last_model_id` to final analysis only: `COALESCE(a.analysis_scope,'final')='final'`. Evaluation reload uses the same boundary. See [ExperimentRecommendationRepository.cpp:421](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:421>) and [ExperimentRecommendationEvaluationRepository.cpp:293](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:293>). |
| 4 | **CONFIRMED, with precision correction** | The reported default weights are exact. Positive contributions are divided by total positive weight, penalties are divided by the same value, and the result is clamped. Under defaults, positive weight is exactly 1.0, so the shorthand formula is numerically accurate. See [ExperimentRecommendationScoring.hpp:13](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.hpp:13>) and [ExperimentRecommendationScoring.cpp:321](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:321>). |
| 5 | **CONFIRMED** | `source_evidence_count` is `pred_down_count + pred_neutral_count + pred_up_count`. These are prediction/confusion counts, not actionable/trade counts. See [ExperimentRecommendationRepository.cpp:442](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:442>). |
| 6 | **CONFIRMED** | Campaign policy has `minimumProfitability`; candidates have optional `profitabilityMetric` and identity. Normal repository loading does not populate them. See [ExperimentRecommendationCampaignPlanning.hpp:29](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.hpp:29>) and [ExperimentRecommendationCampaignPlanningRepository.cpp:143](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:143>). |
| 7 | **CONFIRMED** | When `minimumProfitability` is unset, no profitability check occurs. See [ExperimentRecommendationCampaignPlanning.cpp:1008](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:1008>). |
| 8 | **CONFIRMED** | When configured and the candidate metric is absent, admission fails closed as `profitability_metric_unavailable`. The behavior is tested in [ExperimentRecommendationCampaignPlanningTests.cpp:335](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationCampaignPlanningTests.cpp:335>). |
| 9 | **CONFIRMED** | Broad ranking scopes load all matching evaluation results without restricting scoring/evaluation policy or selecting one evaluation per recommendation. The core sorter then compares `finalScore` without homogeneity validation. See [ExperimentRecommendationRankingRepository.cpp:353](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingRepository.cpp:353>) and [ExperimentRecommendationRanking.cpp:406](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:406>). |
| 10 | **CONFIRMED** | Final analysis upserts on `(experiment_id,model_id)` and updates metric values while retaining `analysis_id`. Recommendation validation compares model/analysis IDs and current status/scope, not current metric values. See [ExperimentScheduler.cpp:11461](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11461>) and [ExperimentRecommendationEvaluation.cpp:346](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:346>). |
| 11 | **CONFIRMED, with added nuance** | Analysis starts from log parsing, then optionally overlays accuracy/acceptance from `inference_eval_result`. The structured query does not retain `inference_eval_result.id`; confusion counts may still be log-derived even when accuracy is structured. See [ExperimentScheduler.cpp:9766](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9766>) and [ExperimentScheduler.cpp:9997](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9997>). |
| 12 | **CONFIRMED** | `ExperimentMetaAnalyzer` has a separate leader-neighborhood/direct-insert path. `InsertMetaRecommendationExperiment` inserts directly into `experiment`; it does not use modern recommendation/evaluation/ranking/campaign tables. See [ExperimentMetaAnalyzer.cpp:1214](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:1214>) and [ExperimentMetaAnalyzer.cpp:1510](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:1510>). |
| 13 | **CONFIRMED** | Modern recommendation scoring contains no history/trend model. Scheduler continuation has explicit patience-window trend logic over leader score or inference accuracy. See [ContinuationPolicy.hpp:18](</Volumes/Developer SSD/ExpertAdvisor/Sources/ContinuationPolicy.hpp:18>) and [ExperimentScheduler.cpp:12544](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12544>). |

The exact executable default score is:

```text
positive =
    0.25 * clamp(leader_score)
  + 0.25 * inference_accuracy
  + 0.15 * evidence_strength
  + 0.10 * neutral_balance
  + 0.15 * structural_proximity
  + 0.05 * parameter_preference
  + 0.05 * reciprocal_source_rank

penalty =
    0.05 * normalized_horizon_change
  + 0.10 * normalized_mutation_distance

final = clamp((positive - penalty) / total_positive_weight,
              score_floor, score_ceiling)
```

The current component set is exactly nine components, enforced by [034_experiment_recommendation_evaluation.sql:143](</Volumes/Developer SSD/ExpertAdvisor/Database/Migrations/034_experiment_recommendation_evaluation.sql:143>).

# 3. Current Profitability Producer and Persistence State

## Current calculation

The only profitability-like implementation is in inference:

- `PredictionStats` owns actionable/trade counts, win/loss counts, gross positive/negative log returns, and aggregate log return: [main.cpp:2460](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:2460>).
- `ProcessBatchPredict` calculates per-window directional returns: [main.cpp:2763](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:2763>).
- `PrintEvalTradingMetrics` derives average log return, win rate, and profit factor: [main.cpp:2513](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:2513>).
- `RunInferenceEvaluation` aggregates these statistics: [main.cpp:6483](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6483>).

For a predicted up class:

```text
realized_log_return = log(target_close / close_t)
```

For a predicted down class, the sign is reversed. Predicted neutral increments `flatCount` and does not create a trade.

Important semantic limitations:

- The label is based on look-ahead first-hit high/low behavior.
- The return calculation exits at the terminal horizon close, not at the first-hit event.
- Successive inference windows can overlap, so “trades” are not an executable non-overlapping portfolio.
- Argmax direction is used; there is no separate confidence/action threshold.
- There is no spread, commission, slippage, borrow cost, latency, or transaction-cost deduction.
- There is no capital, risk, leverage, notional, volatility, or position-sizing model.
- A zero return is neither a win nor a loss.
- Profit factor returns `0.0` when no negative gross return exists, which is a diagnostic convention rather than a generally accepted profitability definition.

## Final versus checkpoint

Final and checkpoint inference use the same `RunInferenceEvaluation` calculation. Scope is assigned by `ResolveSchedulerInferencePersistenceContext`:

- `schedulerCheckpointEvalId` present → `checkpoint`
- otherwise scheduler inference → `final`

See [main.cpp:6061](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6061>) and the shared persistence branch at [main.cpp:7871](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:7871>).

The database distinguishes scopes through `inference_scope`, `checkpoint_eval_id`, `parent_experiment_id`, and `checkpoint_epoch`, with final rows requiring the latter three to be `NULL`: [018_checkpoint_inference_result_scope.sql:1](</Volumes/Developer SSD/ExpertAdvisor/Database/Migrations/018_checkpoint_inference_result_scope.sql:1>).

## Existing identities

- Inference row identity: `inference_eval_result.id`.
- Logical inference identity: model, symbol, horizon, threshold, window size, label-rule ID, target type, `from_date`, and `to_date`, represented in `InferenceIdentity`: [main.cpp:6006](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6006>).
- Model identity: `model.model_id`; `model.experiment_id` provides experiment lineage and is immutable once linked under [070_model_experiment_lineage_immutable.sql:5](</Volumes/Developer SSD/ExpertAdvisor/Database/Migrations/070_model_experiment_lineage_immutable.sql:5>).
- Experiment identity: `experiment.experiment_id`.
- Checkpoint inference identity: `checkpoint_eval_id`, parent experiment, checkpoint model, and epoch.
- Final inference currently has no direct `experiment_id` column; it is resolved through model lineage.
- Interval provenance is persisted as text `from_date`/`to_date`.

## Persistence state

`inference_eval_result` persists accuracy, acceptance, predicted class fractions, scope, and interval identity, but no trade or profitability fields: [004_inference_eval_result.sql:1](</Volumes/Developer SSD/ExpertAdvisor/Database/Migrations/004_inference_eval_result.sql:1>).

`InferenceEvaluationResult` carries only completed epochs, accuracy, and acceptance. Aggregated trading statistics are discarded after optional output: [main.cpp:6216](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6216>).

Final inference rows are mutable upserts under the same `id`: [main.cpp:6311](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6311>). Therefore `inference_eval_result.id` alone is not an immutable inference-result revision.

`--eval-trading` controls printing, not the underlying classification-loop calculation. It defaults false: [main.cpp:3812](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:3812>).

## Live persisted state

A read-only transaction against database `LSTM` found:

- 315 completed final inference rows.
- 502 completed checkpoint inference rows.
- No profitability-, trade-, or return-like columns in the inference, analysis, recommendation, evaluation-result, or ranking-member tables.

Therefore the Campaign Manager must wait for a producer-owned, immutable or content-versioned, final-scope profitability observation. It must not parse inference logs or reconstruct profitability from prediction fractions.

# 4. Proposed FINAL Profitability Evidence Contract

The recommended authoritative record is an append-only table such as `inference_profitability_observation`. The name is illustrative; no schema change was made.

It should reference an inference row but also retain a content identity because `inference_eval_result.id` is currently mutable.

| Field | Classification | Reason |
|---|---|---|
| `profitability_observation_id` | **REQUIRED FOR CORRECTNESS** | Durable immutable identity consumed by recommendations. |
| `experiment_id` | **REQUIRED FOR CORRECTNESS** | Proves the final experiment source without relying on an unversioned join through a mutable inference row. |
| `model_id` | **REQUIRED FOR CORRECTNESS** | Binds evidence to the exact final model. |
| `inference_eval_result_id` | **REQUIRED FOR CORRECTNESS** | Answers which structured inference result produced the observation. |
| Inference-result content/revision hash | **REQUIRED FOR CORRECTNESS** | `inference_eval_result.id` can survive an upsert; the content identity detects revision drift. |
| Explicit `inference_scope` | **REQUIRED FOR CORRECTNESS** | Prevents checkpoint evidence from satisfying a final requirement. |
| `checkpoint_eval_id` | **REQUIRED FOR CORRECTNESS** | Must be explicitly `NULL` for final; non-NULL for checkpoint if the table supports both scopes. |
| `inference_start`, `inference_end` | **REQUIRED FOR CORRECTNESS** | Profitability depends on the evaluated interval and window length. |
| Metric value | **REQUIRED FOR CORRECTNESS** | Required when an observation exists; absence must be represented by no observation/NULL, not zero. |
| Metric name and unit | **REQUIRED FOR CORRECTNESS** | A raw log-return sum is not interchangeable with return percentage, profit factor, or average return. |
| Metric definition canonical/version/hash | **REQUIRED FOR CORRECTNESS** | Encodes exit rule, direction, costs, sizing, overlap convention, and calculation version. |
| Actionable/trade count | **REQUIRED BEFORE NONZERO SCORING** | Distinguishes one strong trade from thousands and distinguishes no-action windows from zero profit. Persist in 3A for audit. |
| Prediction/evidence count | **REQUIRED BEFORE NONZERO SCORING** | Separates total predictions from actionable observations and permits coverage analysis. |
| Winning/losing trade counts | **USEFUL FOR OBSERVABILITY** | Supports audit and distribution checks; required later only if the chosen metric uses them. |
| Gross positive/negative result | **USEFUL FOR OBSERVABILITY** | Permits reproduction of gross total and profit factor. |
| Gross aggregate result | **USEFUL FOR OBSERVABILITY** | Existing `tradeLogReturnSum`; useful, but not a comparable policy metric by itself. |
| Net result | **REQUIRED BEFORE NONZERO SCORING** if net profitability is selected | There is no current net calculation. It is otherwise not a necessary duplicate field. |
| Transaction-cost assumptions | **REQUIRED FOR CORRECTNESS** | Explicit zero-cost assumptions are acceptable for a diagnostic v1 observation; silence is not. |
| Sizing/capital normalization | **REQUIRED BEFORE NONZERO SCORING** | Necessary for cross-symbol and overlapping-horizon comparability. |
| Calculation timestamp | **USEFUL FOR OBSERVABILITY** | Operational traceability; must not substitute for version identity. |
| Immutable content canonical/hash | **REQUIRED FOR CORRECTNESS** | Prevents silent meaning changes and supports idempotent persistence. |

Minimum recommendation snapshot:

- profitability observation ID;
- snapshotted metric value;
- metric-definition hash;
- inference-result ID/content hash;
- actionable count;
- prediction count.

The full calculation assumptions remain recoverable from the immutable producer observation.

# 5. Canonical Profitability Metric / Unresolved Metric Decisions

The repository does not define one authoritative canonical “profitability” metric.

Existing calculations include:

- aggregate gross terminal log return;
- average terminal log return per actionable prediction;
- gross positive and negative log returns;
- win rate;
- profit factor;
- long/short versions of those diagnostics.

None is presently suitable as a cross-campaign decision metric:

- Aggregate return grows with inference-window length and trade count.
- Average return ignores evidence volume and trade overlap.
- Profit factor is unstable at low counts and currently maps “no losses” to zero.
- Log returns are dimensionless but not automatically risk-, horizon-, liquidity-, or cost-comparable.
- The simulated windows can overlap and do not represent a capital-constrained execution sequence.
- Gross returns omit symbol-specific spread and transaction costs.

Phase 3A may persist the already-computed primitives under precise names such as “gross terminal-horizon log-return sum” and “average terminal-horizon log return per actionable prediction.” It must not relabel either as the canonical campaign profitability metric.

Before nonzero weight or threshold activation, an explicit policy decision is required covering:

1. Gross versus cost-adjusted net result.
2. Aggregate versus per-trade or capital-normalized result.
3. Terminal-horizon versus first-hit exit convention.
4. Handling of overlapping forecasts.
5. Position sizing, leverage, and capital allocation.
6. Symbol-specific spread/cost assumptions.
7. Horizon/time normalization.
8. Minimum actionable count.
9. Outlier clipping and missing-value behavior.
10. Whether direction-specific asymmetry matters.

# 6. Exact Future Profitability Data Flow

| Boundary | Current state | Smallest future Phase 3A change |
|---|---|---|
| Inference calculation | `PredictionStats` is discarded after output. | Return the aggregated statistics through `InferenceEvaluationResult` or a dedicated result structure. |
| Authoritative persistence | `inference_eval_result` has no trade/profit fields and is mutable. | Insert an immutable profitability observation in the same transaction as final inference persistence, with scope and content identity. |
| Recommendation source discovery | Final analysis is loaded, but no inference result is joined. | Join one exact final observation using experiment/model/interval/config identity; never “latest by model only.” |
| Recommendation persistence | Snapshots analysis metrics only. | Add nullable profitability observation provenance and scalar/count snapshots for new recommendations. |
| Evaluation input | Loads recommendation snapshot plus current analysis IDs/status. | Load snapshotted profitability and current observation identity/content for validation. |
| Evaluation result | Evidence canonical has no profitability. | Propagate the observation ID/value/definition/counts into immutable evaluation evidence and queryable result columns. Do not change score input. |
| Score component exposure | Nine active components only. | No score component in 3A. Expose profitability under a separate diagnostic evidence section. |
| Durable ranking | Ranks evaluation `finalScore`. | Leave score and order unchanged. Ranking retains evaluation-result linkage, which provides profitability audit provenance. |
| Campaign planning | Reloads ranking member plus recommendation metrics; existing profit field is NULL. | Join the exact evaluation result for diagnostics. Keep existing decision-bearing `profitabilityMetric` NULL in 3A. |
| Campaign admission | Optional `minimumProfitability` gate. | Do not activate it in 3A. |
| Approval/materialization | Reconstructs and verifies the plan. | No Phase 3A decision-contract change. |
| Campaign Manager dispatch | Dispatches durable operational requests tied to materialization. | No change. It must not query or recalculate profitability. |

The current planning SQL notably joins ranking members to recommendations but not the ranking member’s evaluation result: [ExperimentRecommendationCampaignPlanningRepository.cpp:143](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:143>). Future diagnostics should follow:

```text
ranking_member.recommendation_evaluation_result_id
    -> evaluation_result.source_profitability_observation_id
    -> immutable profitability observation
```

They should not reload the current inference observation and thereby change historical meaning.

# 7. Behavior-Neutral Phase 3A Design

## Recommended approach: A

**Persist profitability but leave it entirely outside score calculation.**

Do not add a zero-weight component.

A zero-weight component would still:

- require changing the database component-name constraint;
- change component count;
- change evaluation evidence and result identities;
- likely introduce a new scoring canonical/version;
- cause old and new formal policies to coexist;
- complicate broad ranking scopes even when numeric scores are equal.

Keeping profitability outside the scoring policy preserves the existing canonical scoring policy and all nine components.

## Campaign planning representation

The existing `RecommendationCampaignCandidateInput::profitabilityMetric` is already part of candidate canonical evidence and is consumed by `minimumProfitability`: [ExperimentRecommendationCampaignPlanning.cpp:365](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:365>).

Therefore Phase 3A should introduce a separate observational structure, for example:

```text
observedFinalProfitability {
    observation_id
    inference_result_id
    value
    metric_definition_hash
    actionable_count
    prediction_count
}
```

It should be printable but excluded from campaign decision canonicalization. Mapping it into the existing `profitabilityMetric` field belongs to the later activation phase.

## Behavior-neutral invariants

Phase 3A should prove:

- Same source eligibility decisions for identical pre-profitability evidence.
- Same generated recommendation configurations and duplicate decisions.
- Same recommendation score values, preferably bit-for-bit.
- Same nine score components, weights, and ordering.
- Same scoring-policy canonical text and hash.
- Same advisory-ready/non-actionable dispositions except for independently invalid profitability provenance, which should be diagnostic rather than blocking in 3A.
- Same ranking order when ranking equivalent evaluation populations.
- Same campaign inclusion/exclusion decisions and reasons.
- Same conversion, materialization, operational request, and dispatch behavior.
- NULL profitability means unavailable and does not reject when no profitability policy is active.
- Checkpoint observations cannot populate final profitability fields.
- Existing immutable ranking/campaign snapshots are not modified.
- Old recommendation/evaluation rows remain readable with NULL profitability provenance.

Evaluation identities will legitimately differ when new profitability evidence is included in the evaluation evidence canonical. That does not change the score, but it reinforces the need to fix broad-scope duplicate/historical evaluation selection before decision-bearing profitability.

# 8. Scoring-Policy Identity and Ranking Compatibility

## Current mechanism

The scoring policy canonical contains every weight and normalization parameter and is hashed: [ExperimentRecommendationScoring.cpp:281](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:281>).

Evaluation runs persist:

- evaluation policy canonical/hash;
- evaluation and evaluator versions;
- scoring policy canonical/hash;
- scoring version.

See [034_experiment_recommendation_evaluation.sql:4](</Volumes/Developer SSD/ExpertAdvisor/Database/Migrations/034_experiment_recommendation_evaluation.sql:4>).

Ranking evaluation objects retain these identities: [ExperimentRecommendationRanking.hpp:67](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.hpp:67>).

## Confirmed incompatibility

`LoadEvaluationsForRanking` restricts by run only when `evaluationRunId` is supplied. Scan, symbol, horizon, family, symbol/horizon, and global scopes can load evaluation results from multiple runs and policies. It also does not select only one evaluation per recommendation.

`RankRecommendationEvaluationEvidence` validates that policy identities are present, but not equal. It sorts advisory scores directly.

By contrast, `CompareRecommendationEvaluations` explicitly calls differing evaluation/scoring policies incomparable: [ExperimentRecommendationRanking.cpp:564](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:564>). The comparison helper and durable ranker therefore disagree.

Normal CLI operation exposes all broad scopes, including global: [ExperimentScheduler.cpp:24098](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:24098>).

## Current operational occurrence

A read-only query on 2026-08-22 found:

- 4 evaluation runs;
- 1 distinct scoring-policy canonical/hash;
- 3 ranking snapshots;
- 0 snapshots containing more than one scoring policy.

Thus the defect is executable but is not currently manifested in persisted production rankings.

## Effect of profitability changes

- Phase 3A approach A does not create a new scoring-policy identity.
- A formally versioned zero-weight component would create a new policy/component contract.
- Old and new evaluation results can coexist under broad ranking scopes.
- A later nonzero profitability policy makes cross-policy numeric comparisons mathematically invalid.

## Remediation alternatives

| Alternative | Assessment |
|---|---|
| A. Rank only within one scoring-policy identity | Necessary minimum invariant, but also require one evaluation per recommendation. |
| B. Select latest/current policy | Acceptable only if “current” is explicit canonical policy configuration, not inferred merely from timestamps. |
| C. Recompute under current policy | Preferred when policy changes: evaluate the intended population in one completed evaluation run, then rank that run. |
| D. Define cross-policy comparability | Not recommended. Scores with changed weights/normalization do not have an established common scale. |
| E. Campaign-grade rankings require `evaluation_run` scope | Smallest strong repository-consistent boundary. A run already has one evaluation/scoring policy identity. |

Recommended design:

1. Campaign-consumable ranking snapshots must be evaluation-run scoped.
2. Core ranking validation must assert homogeneous evaluation/scoring canonical identities and versions.
3. Ranking input must contain at most one result per recommendation.
4. Policy changes require recomputation under the selected current policy.
5. Broad historical scopes may remain advisory only if they explicitly select a policy and deterministically select one evaluation per recommendation.

**Timing: AFTER Phase 3A, BUT BEFORE NONZERO PROFITABILITY OR MINIMUM-PROFITABILITY ACTIVATION.**

# 9. Source-Evidence Staleness / Mutability

## Confirmed scenario

Final analysis persistence uses:

```sql
ON CONFLICT (experiment_id, model_id)
WHERE analysis_scope='final'
DO UPDATE SET
    infer_accuracy = EXCLUDED.infer_accuracy,
    ...
    leader_score = EXCLUDED.leader_score,
    updated_at = now()
```

The primary `analysis_id` is retained: [ExperimentScheduler.cpp:11461](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11461>).

Recommendations snapshot:

- `source_analysis_id`;
- `source_leader_score`;
- `source_infer_accuracy`;
- predicted-neutral proportion;
- `source_evidence_count`.

See [ExperimentRecommendationRepository.cpp:723](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:723>).

Evaluation reloads current:

- source experiment status/phase;
- `last_model_id`;
- analysis ID;
- analysis status;
- analysis scope.

It does not load current leader score, inference accuracy, neutral proportion, or evidence count.

Therefore:

1. Recommendation snapshots values from analysis ID X.
2. Final analysis reruns.
3. X remains the same.
4. Values change.
5. Evaluation sees the same model/analysis IDs and completed final scope.
6. Recommendation remains valid.

This scenario is confirmed.

A read-only database check found:

- 321 final analysis rows;
- 3 with `updated_at > created_at`;
- currently 0 recommendation snapshots differing from their linked analysis metrics.

The flaw exists, but no current mismatch was observed.

## Profitability interaction

Profitability must not repeat this design. An observation ID cannot be treated as authoritative if its contents may change.

Preferred remediation:

- New profitability observations are append-only.
- Every observation has an immutable content canonical/hash.
- Recommendations snapshot both observation ID and content hash.
- Evaluation validates that the referenced current content identity still matches.

For existing analysis evidence, the smallest repository-consistent remediation is:

1. Add an analysis evidence canonical/hash or revision.
2. Compute it from the decision-bearing analysis content.
3. Snapshot it in recommendations.
4. Recompute/load the current identity during evaluation.
5. Mark a mismatch `stale_source_evidence`.

Alternative assessment:

| Alternative | Assessment |
|---|---|
| Immutable analysis observations | Strongest, but larger redesign. |
| Revision column incremented on upsert | Viable if transactionally enforced and never reset. |
| Content hash/evidence identity | Preferred minimal solution; directly represents meaning. |
| Exact snapshot-value comparison | Better than current behavior but weaker for future fields and canonical floating-point handling. |
| Inference-result ID + calculation version | Insufficient alone because current inference and analysis rows can retain IDs while changing content. |

**Timing: AFTER Phase 3A, BUT BEFORE profitability affects score or admission.** Phase 3A must itself use immutable profitability observations from the outset.

# 10. Provenance and Auditability

Existing recommendation provenance is insufficient to answer:

> Exactly which final inference result and profitability calculation caused this recommendation to receive this profitability evidence?

A recommendation currently retains only source experiment, model, and analysis IDs. The analysis row does not retain `inference_eval_result.id`.

Moreover, `ApplyStructuredInferenceMetrics` selects only accuracy, acceptance, reject reason, and completed epochs. It does not select the inference row ID, and confusion counts may remain log-derived.

The smallest provenance extension is:

```text
recommendation
  -> source_profitability_observation_id
      -> inference_eval_result_id
      -> inference_result_content_hash
      -> experiment_id/model_id
      -> final scope/checkpoint_eval_id NULL
      -> interval/configuration identity
      -> metric definition canonical/hash
      -> exact calculation inputs and results
```

Recommendations and evaluation results should also snapshot the observation ID, value, definition hash, and counts so audits do not depend on a later mutable join.

Log-derived profitability should **never** be accepted as authoritative recommendation evidence. Logs may remain diagnostics or recovery evidence, but they lack:

- a structured durable identity;
- reliable atomicity with inference persistence;
- calculation-version identity;
- scope-enforced foreign keys;
- immutable content semantics.

# 11. `minimumProfitability` Semantics

1. **Should Phase 3A populate profitability evidence while leaving `minimumProfitability` normally unset?**
   Yes, but it should populate a separate observational field, not the existing decision-bearing `profitabilityMetric`.

2. **If a user explicitly sets `minimumProfitability` after Phase 3A, should the gate immediately become functional?**
   No. Activation should require an explicit, supported metric-definition identity and a later campaign-policy contract version.

3. **Would immediate activation violate behavior neutrality?**
   Yes. The current CLI already exposes `--campaign-min-profitability`: [ExperimentScheduler.cpp:2725](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:2725>). Filling the current candidate field would make an existing option begin admitting/rejecting candidates.

4. **Should activation be separate?**
   Yes. It should occur only after metric selection, normalization, stale-evidence validation, ranking homogeneity, and threshold calibration.

5. **What does NULL mean?**
   `NULL` means unavailable. It is never zero. It should reject only when an explicitly activated policy requires a supported profitability observation.

The existing fail-closed behavior is appropriate once the policy is intentionally activated. It should not be silently activated by plumbing.

# 12. Profitability Normalization and Evidence Strength

Before any nonzero decision role, the following must be resolved:

- **Symbol comparability:** spread, volatility, tick size, and market behavior differ.
- **Horizon comparability:** raw return and opportunity frequency scale with horizon.
- **Inference-window comparability:** aggregate return grows with evaluated window length.
- **Capital/sizing comparability:** overlapping forecasts cannot all assume independent full capital.
- **Evidence strength:** prediction count is not trade count. Neutral predictions and invalid-price windows do not produce trades.
- **Costs:** current calculations are gross.
- **Outliers:** profit factor and mean return can be dominated by small denominators or extreme observations.
- **Directionality:** short results use negated terminal return; long/short asymmetry may matter.
- **Missing values:** no trades must be unavailable/insufficient evidence, not necessarily zero profitability.
- **Normalization:** any clipping or transformation must be based on observed distributions, not arbitrary constants.

After Phase 3A, inspect empirical distributions stratified by:

- symbol;
- horizon;
- inference interval and interval length;
- actionable count and prediction count;
- long/short side;
- gross sum and average return;
- win/loss counts;
- cost-adjusted scenarios;
- volatility/risk regime;
- missing and zero-action cases.

Also inspect:

- quantiles and extreme outliers;
- stability across reruns and adjacent time intervals;
- correlation with leader score and inference accuracy;
- whether profitability adds information beyond existing components;
- rank changes under candidate normalization choices;
- threshold sensitivity and minimum-trade requirements.

# 13. Legacy Recommendation Path Boundary

The legacy boundary can safely remain outside Phase 3A.

Hard implementation constraints:

- Do not modify `LSTM/ExperimentMetaAnalyzer.cpp`.
- Do not add profitability to `RecommendationRankingScore`.
- Do not add joins from legacy meta-analysis to modern profitability observations.
- Do not alter `--queue-meta-recommendations`.
- Do not route legacy direct experiment insertion through modern recommendation/campaign policy as part of this increment.
- Add a regression proving legacy output and direct-queue behavior are unchanged.

The legacy path ranks completed experiment records by leader score or inference accuracy and generates nearby configurations: [ExperimentMetaAnalyzer.cpp:1214](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:1214>). It is architecturally independent and should remain so.

# 14. Scheduler Continuation Boundary

This is the required ownership contract; profitability/trade fields are not currently implemented in these policies.

| Evidence | Checkpoint policy | Scheduler continuation | Modern recommendation/campaign | Boundary |
|---|---:|---:|---:|---|
| Checkpoint profitability | Yes | Yes, if later incorporated as continuation evidence | No | Must remain checkpoint-scoped. |
| Final profitability | No | No for this integration | Yes | Modern final source evidence only. |
| Checkpoint trade count | Yes, as checkpoint evidence strength | Yes, if checkpoint profitability is considered | No | Never satisfies final recommendation evidence. |
| Final trade count | No | No for this integration | Yes | Supports final profitability strength/observability. |
| Checkpoint trend | No direct campaign role | Yes | No | Existing trend belongs to continuation. |
| Final recommendation source metric | No | No | Yes | Final persisted recommendation/campaign policy only. |

Current continuation evidence deliberately combines checkpoint and final analysis points for its existing leader/inference policies: [ExperimentScheduler.cpp:12077](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12077>). That does not authorize final profitability to enter continuation as part of this work.

The modern recommendation implementation must query:

```sql
inference_scope = 'final'
AND checkpoint_eval_id IS NULL
```

It must never fall back to a checkpoint observation.

# 15. Schema / Migration Impact

No migration was created. A behavior-neutral Phase 3A likely needs one additive migration after the current migration 072.

## New append-only profitability observation table

Purpose:

- Producer-owned authoritative calculation record.
- Explicit final/checkpoint scope.
- Immutable content/version identity.

Key constraints:

- FK to `inference_eval_result.id`, `model.model_id`, and `experiment.experiment_id`.
- Optional FK to `experiment_checkpoint_eval.checkpoint_eval_id`.
- Scope-shape constraint matching migration 018.
- Finite numeric checks.
- Nonnegative count checks.
- Wins + losses not greater than actionable count.
- Unique content canonical or `(inference result content hash, metric-definition hash)`.
- Revoke `UPDATE` and `DELETE` from application role.

Backfill:

- None. Existing inference logs must not be parsed into authoritative rows.
- Old inference rows remain valid but have no profitability observation.

## `experiment_recommendation`

Add nullable fields for:

- source profitability observation ID;
- snapshotted value;
- metric-definition hash;
- inference-result ID/content hash;
- actionable count;
- profitability prediction count.

Constraints:

- All provenance fields either consistently NULL or consistently present.
- No default other than NULL.
- No guessed backfill.

Old recommendation rows remain readable and mean “profitability unavailable.”

## `experiment_recommendation_evaluation_result`

Add corresponding nullable immutable provenance fields. Include the evidence in `evidence_canonical`, but not in scoring-policy canonical or score components.

Old rows remain readable.

## No Phase 3A changes

Do not alter:

- `experiment_recommendation_evaluation_component`;
- component-name constraint;
- ranking score columns;
- ranking-member score semantics;
- campaign approval/materialization schema;
- operational request schema;
- experiment schema;
- checkpoint/continuation policy schema.

Existing scoring-policy identities remain valid because their canonical text is unchanged.

## Later migrations

- **3B:** ranking snapshot policy-homogeneity metadata/constraints if needed.
- **3C:** analysis evidence revision or content identity.
- **Policy activation:** profitability component name/weight, score version, campaign policy version, and explicit supported metric identity.

# 16. Required Test Matrix

| # | Test | Phase |
|---|---|---|
| 1 | Final inference calculation persists one exact final observation with matching experiment/model/result/interval. | 3A |
| 2 | Checkpoint observation cannot be loaded as modern recommendation evidence. | 3A |
| 3 | Recommendation provenance resolves to the exact final inference row/content and metric definition. | 3A |
| 4 | Missing observation remains NULL/unavailable without changing eligibility or score. | 3A |
| 5 | Prediction, actionable, long/short, win/loss, and gross-return aggregation propagates exactly. | 3A |
| 6 | New recommendations snapshot profitability evidence idempotently; old recommendations remain NULL/readable. | 3A |
| 7 | Evaluation propagates and validates final observation provenance without adding a score component. | 3A |
| 8 | Score fields and all nine component rows are bit-for-bit equal before/after plumbing. | 3A |
| 9 | Ranking order is identical for equivalent single-run evaluation populations. | 3A |
| 10 | Campaign decisions/reasons are identical with the observational evidence present but inactive. | 3A |
| 11 | Mixed scoring/evaluation policies are rejected or separated; one result per recommendation. | 3B |
| 12 | Same analysis ID with changed metric content becomes stale after remediation; changed profitability content cannot retain the same observation identity. | 3C |
| 13 | Old inference/recommendation/evaluation rows remain readable and mean unavailable. | 3A |
| 14 | Unset minimum is neutral; Phase 3A explicit minimum remains unavailable/fail-closed; later activation tests supported metric and threshold behavior. | 3A + later |
| 15 | `ExperimentMetaAnalyzer` results and direct queue path are unaffected. | 3A |
| 16 | Observation selection and all diagnostics remain deterministic under ties and reruns. | 3A |

Additional Phase 3A migration tests should cover:

- append-only permissions;
- scope-shape constraints;
- final/checkpoint cross-link rejection;
- finite-value checks;
- idempotent duplicate calculation persistence;
- inference-result content drift detection.

Before nonzero scoring, add tests for:

- normalization by symbol/horizon/window;
- minimum actionable count;
- no-trade and missing evidence;
- outlier clipping;
- cost assumptions;
- policy identity/version changes;
- rank changes under controlled profitability values;
- active `minimumProfitability` with exact metric-definition matching.

# 17. Exact Implementation Surface

## PHASE 3A — Profitability plumbing/provenance/observability

| File/surface | Function/class | Change |
|---|---|---|
| [LSTM/main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp>) | `PredictionStats`, `ProcessBatchPredict`, `RunInferenceEvaluation`, `InferenceEvaluationResult` | Retain aggregated trading statistics as structured results. Preserve current calculation semantics. |
| [LSTM/main.cpp:6311](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6311>) | `PersistCompletedInferenceResult` | Obtain exact persisted final inference row/content identity and insert immutable profitability observation in the same transaction. |
| [LSTM/main.cpp:6383](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6383>) | Checkpoint persistence | Preserve explicit checkpoint scope; optional checkpoint observation support must not enter modern source queries. |
| New additive migration | New observation table; nullable recommendation/evaluation fields | Add immutable scope/version/provenance contract without backfill. |
| [ExperimentRecommendation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp>) | `RecommendationSource` | Add optional final profitability evidence structure. |
| [ExperimentRecommendationRepository.cpp:409](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:409>) | `LoadRecommendationSources` | Join exact final observation with explicit scope and full lineage/interval conditions. |
| [ExperimentRecommendationRepository.cpp:723](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:723>) | Recommendation insert | Snapshot optional observation provenance/value/counts for new rows. |
| [ExperimentRecommendationEvaluation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.hpp>) | Evaluation input/result | Add optional snapshotted and current profitability provenance. |
| [ExperimentRecommendationEvaluation.cpp:254](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:254>) | Evidence canonical/evaluation | Validate final scope/content identity and propagate evidence. Do not copy it into `RecommendationScoringInput`. |
| [ExperimentRecommendationEvaluationRepository.cpp:293](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:293>) | Evaluation loader/persistence | Load snapshot/current observation and persist immutable evaluation provenance. |
| [ExperimentRecommendationCampaignPlanning.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.hpp>) | Candidate diagnostics | Add separate observed-final-profitability diagnostic structure. Do not populate existing admission field. |
| [ExperimentRecommendationCampaignPlanningRepository.cpp:143](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:143>) | Plan input loader | Join through ranking member’s evaluation result for diagnostic provenance. |
| [ExperimentRecommendationCampaignPlanningService.cpp:75](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningService.cpp:75>) | Output | Print observation ID, metric identity, value, actionable count, and availability. |
| Existing/new recommendation, evaluation, planning tests | Relevant test suites | Add the Phase 3A matrix above. |

Files that should remain untouched in 3A:

- `ExperimentRecommendationScoring.*`
- `ExperimentRecommendationRanking.*`
- continuation/checkpoint policy
- legacy `ExperimentMetaAnalyzer`
- conversion execution
- campaign approval/materialization
- Campaign Operations Manager and dispatch

## PHASE 3B — Scoring-policy/ranking compatibility

| File | Change |
|---|---|
| [ExperimentRecommendationRankingRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingRepository.cpp>) | Require/choose one explicit policy identity; select one evaluation per recommendation. |
| [ExperimentRecommendationRanking.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp>) | Validate homogeneous evaluation/scoring policy and evaluator versions before sorting. |
| [ExperimentRecommendationRanking.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.hpp>) | Carry explicit population policy identity if needed. |
| Ranking migration/tests | Persist/enforce campaign-grade policy identity and evaluation-run scope. |

## PHASE 3C — Stale-evidence remediation

| File | Change |
|---|---|
| [ExperimentScheduler.cpp:11343](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11343>) | Compute analysis content canonical/hash or increment an enforced revision on every material update. |
| Analysis migration | Add revision/content identity without changing existing `analysis_id`. |
| `ExperimentRecommendationRepository.*` | Snapshot analysis evidence identity. |
| `ExperimentRecommendationEvaluationRepository.*` | Load current evidence identity/content. |
| `ExperimentRecommendationEvaluation.cpp` | Mark mismatches stale. |

## LATER — Profitability policy activation

| Surface | Change |
|---|---|
| `ExperimentRecommendationScoring.*` | Add normalized profitability input/component and new scoring-policy version only after metric choice. |
| Evaluation component migration | Add new component name and versioned shape. |
| Ranking | Recompute complete populations under the new policy; never mix old/new scores. |
| Campaign planning | Map supported observed evidence into decision-bearing `profitabilityMetric`; bump planning contract/version. |
| CLI/config | Require explicit profitability metric identity alongside threshold/weight. |
| Tests | Distribution, normalization, threshold, weight, missing-value, and policy-transition coverage. |

# 18. Implementation Sequencing Decision

## Decision: OPTION 1

Implement in this order:

1. **Phase 3A: behavior-neutral final profitability persistence, provenance, propagation, and observability.**
2. **Phase 3C: stale analysis-evidence/content-identity remediation.**
3. **Phase 3B: campaign-grade ranking homogeneity and one-evaluation-per-recommendation enforcement.**
4. **Empirical profitability distribution and normalization analysis.**
5. **Later policy phase: activate `minimumProfitability` and/or a nonzero score component under a new explicit policy identity.**

Steps 2 and 3 may be reversed or developed independently, but both must be complete before step 5.

Why Phase 3A can safely come first:

- It need not change the scoring policy.
- It need not change score values or component count.
- It need not activate campaign admission.
- New profitability observations can be immutable from inception, avoiding another stable-ID/mutable-content defect.
- It provides the persisted distributions needed to make the unresolved normalization decision empirically.

Why scoring/staleness do not need to precede Phase 3A:

- Phase 3A is observational, not decision-bearing.
- It does not require cross-policy score comparison.
- Existing analysis staleness is not worsened if profitability uses its own immutable content identity.
- Fixing either first would not resolve the absence of authoritative persisted profitability data.

If Phase 3A is instead defined to populate the existing `profitabilityMetric`, add a zero-weight score component, or change candidate admission, then it is no longer behavior-neutral and this sequence is invalid. In that case, scoring-policy and stale-evidence remediation must precede or accompany it.

# 19. Risks / Open Questions

1. Which existing return primitive, if any, should later become the canonical decision metric?
2. Should the execution convention use terminal close or first-hit exit behavior?
3. How should overlapping forecast windows consume capital?
4. What symbol-specific spread/cost source is authoritative and versioned?
5. Does an actionable prediction represent a trade, a signal, or an independent hypothetical position?
6. Should zero actionable trades be unavailable or a defined zero metric? Recommendation: unavailable.
7. Should Phase 3A persist final observations only, or use one scoped table for both final and checkpoint observations?
8. Should existing active recommendations remain NULL permanently or receive append-only enrichment? Backfilling values into existing recommendation rows is not recommended.
9. The current final inference row is mutable. A profitability observation must carry the inference content identity or final inference persistence must become revisioned/immutable.
10. Broad ranking scopes can also contain more than one evaluation of the same recommendation, independent of scoring-policy mixing.
11. Evaluation evidence identities will change when profitability provenance is added even though score policy remains unchanged; ranking tests must use controlled, homogeneous evaluation populations.
12. Current source analysis can mix structured accuracy with log-derived confusion/count evidence.
13. The repository’s active branch is `lstm-feature-development`, while the supplied development status says `phase6`; this should be confirmed before implementation begins.

# 20. Final Recommendation

Implement behavior-neutral Phase 3A next:

- persist exact structured final inference return/trade primitives in an immutable, scope-enforced observation;
- snapshot the observation and calculation provenance in new recommendations and evaluations;
- expose it diagnostically through evaluation and campaign-planning output;
- leave scoring, ranking, campaign admission, conversion, materialization, dispatch, continuation, checkpoint policy, and legacy meta-analysis behavior unchanged.

Then remediate analysis content staleness and ranking policy homogeneity before activating any threshold or nonzero weight.

No canonical profitability policy should be chosen until Phase 3A data has been analyzed across symbols, horizons, inference windows, actionable counts, costs, and outliers.

## Inspection record

- Files changed: **none**
- Database changes: **none**
- Scheduler/campaign/recommendation/experiment state changes: **none**
- Builds/tests run: **none**, as required by the inspection-only constraint
- Database inspection: read-only transaction, rolled back
- `git diff --stat`: empty
- `git diff --check`: clean
- `git status --short`:

```text
?? LSTM_CampaignManager_Phase1_CandidateSelection_Inspection_Output.md
?? LSTM_CampaignManager_Phase2_ScoringRanking_Inspection_Output.md
?? lstm_watch.sql
```

These pre-existing untracked files were not modified.