---
title: "LSTM Campaign Manager Phase 2 Scoring Ranking Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase2_ScoringRanking_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 2 Scoring Ranking Inspection

# 1. Executive Summary

The current persisted recommendation/campaign pipeline is deterministic, snapshot-based, and final-model-oriented. It does not evaluate mutated candidates by training or inference before ranking them. Instead, it combines:

- final source-experiment evidence;
- a one-parameter structural mutation;
- deterministic mutation-distance heuristics;
- explicit policy weights and penalties;
- duplicate/provenance gates;
- durable evaluation and ranking snapshots;
- campaign lifecycle and quota gates.

The executable path is:

```text
completed source experiment
  -> final analysis for experiment.last_model_id
  -> source eligibility and source ranking
  -> one-parameter candidate mutation
  -> persisted experiment_recommendation
  -> standalone score and advisory evaluation
  -> durable ranking snapshot
  -> campaign plan and review
  -> campaign approval
  -> campaign materialization into conversion proposals
  -> Campaign Operations admission/budget/request/production admission
  -> Campaign Manager RUN_ONCE
  -> conversion execution + experiment creation/reuse + activation
```

Principal conclusions:

1. Modern recommendation scoring uses nine components: seven positive components and two penalties. The exact formula is documented in §6.
2. Source `leader_score`, `infer_accuracy`, predicted-class counts, and derived neutral proportion come from a final `experiment_analysis_result` attached to `experiment.last_model_id`.
3. Checkpoint analysis cannot directly enter the modern source, evaluation, ranking, or campaign-planning path.
4. Campaign planning does not define another score. It consumes durable ranking order and applies hard gates and quotas.
5. `minimumProfitability` is executable policy, but repository loading supplies no profitability metric. Configuring it currently excludes every candidate with `profitability_metric_unavailable`.
6. The older `ExperimentMetaAnalyzer` has a separate recommender and direct queueing path. It is not the Phase 4–6 campaign pipeline and uses a different scoring/mutation model.
7. Modern ordering is deterministic for identical persisted state. The legacy plateau-signal ordering has an incomplete tie-break.
8. The cleanest Phase 3 foundation is an authoritative final-inference profitability record with explicit identity, interval, sample/trade counts, and normalization semantics, snapshotted into recommendation evidence.

No code, schema, configuration, database, scheduler, campaign, or experiment state was changed.

# 2. End-to-End Recommendation/Campaign Data Flow

## Modern persisted workflow

```text
SOURCE EXPERIMENT
experiment
  status='completed'
  phase='done'
  last_model_id IS NOT NULL
        |
        v
SOURCE METRICS
experiment_analysis_result
  experiment_id = experiment.experiment_id
  model_id      = experiment.last_model_id
  scope         = final
  status        = completed
        |
        v
SOURCE ELIGIBILITY / SOURCE RANK
EvaluateRecommendationSource
SelectRecommendationSources
        |
        v
CANDIDATE MUTATION
GenerateRecommendationCandidates
  exactly one of:
    core_lr_mult
    head_lr_mult
    label_threshold
    prediction_horizon
        |
        v
DEDUPLICATION / PERSISTENCE
FindExperimentDuplicate
FindRecommendationDuplicate
PersistRecommendationIdempotently
  -> experiment_recommendation
        |
        +--------------------------+
        |                          |
        v                          v
STANDALONE SCORE              ADVISORY EVALUATION
ScoreExperimentRecommendation EvaluateExperimentRecommendation
  -> score run/result/components -> evaluation run/result/components
        |                          |
        +-------------+------------+
                      v
DURABLE RANKING
RankRecommendationEvaluationEvidence
  -> ranking snapshot/members
                      |
                      v
CAMPAIGN PLAN / REVIEW
LoadRecommendationCampaignPlanInput
PlanRecommendationCampaign
ReviewRecommendationCampaignPlan
                      |
                      v
CAMPAIGN APPROVAL
BuildRecommendationCampaignApprovalEvidence
  -> experiment_recommendation_campaign_approval
                      |
                      v
MATERIALIZATION
reconstruct plan/review/approval
BuildProposedExperimentSpecification
  -> campaign materialization + members
  -> conversion proposals
                      |
                      v
CAMPAIGN OPERATIONS
operational campaign admission
authorization + budget
operational request acceptance
production admission/build binding
                      |
                      v
CAMPAIGN MANAGER RUN_ONCE
ORDER BY operational_request_id
LaunchRecommendationCampaignInTransaction
  -> execute proposal
  -> create/reuse experiment
  -> activate pending experiment
  -> durable request/member bindings
```

Primary entry points are:

- [`RunGenerateExperimentRecommendationsCommand`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:307>)
- [`LoadRecommendationSources`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:417>)
- [`ScoreExperimentRecommendation`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:321>)
- [`EvaluateExperimentRecommendation`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:325>)
- [`RankRecommendationEvaluationEvidence`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:406>)
- [`LoadRecommendationCampaignPlanInput`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:97>)
- [`PlanRecommendationCampaign`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:881>)
- [`RunMaterializeRecommendationCampaignCommand`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignMaterializationService.cpp:142>)
- [`RunManagerOnceInternal`](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:127>)

Campaign Manager does not recalculate candidate metrics or scores. Its selection SQL is:

```sql
WHERE request.request_state='ready'
  AND request.production_dispatch_enabled=false
  AND campaign_operations_future_actions_allowed(
        request.operational_campaign_id)
ORDER BY request.operational_request_id
LIMIT $1
```

Evidence: [`SelectDispatchCandidatesForManager`](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp:219>).

# 3. Candidate Object Model

## Source experiments

`RecommendationSource` contains:

- `experimentId`, `modelId`, `analysisId`;
- authoritative source invocation/configuration;
- `leaderScore`;
- `inferenceAccuracy`;
- optional `predictedNeutralProportion`;
- `evidenceCount`.

The configuration includes symbol, horizon, threshold, learning-rate multipliers, target epochs, date ranges, Donchian mode/lookback, warmup scope, checkpoint interval, and resume model identity. See [`RecommendationSource`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:106>).

## Generated mutations

`GeneratedRecommendationCandidate` represents a copied source configuration with exactly one ordinary semantic parameter changed. It also carries:

- source and proposed values;
- absolute and optional relative delta;
- optional horizon delta;
- semantic canonical identity;
- invocation canonical identity.

It has no independently observed inference result.

## Evaluated recommendations

A persisted `experiment_recommendation` is the immutable bridge between source evidence and a mutation. It snapshots:

- source IDs and metrics;
- source rank;
- mutation metadata;
- semantic and invocation canonical identities;
- generation and structural ordinals;
- originating recommendation policy.

Standalone scoring persists to:

- `experiment_recommendation_score_run`;
- `experiment_recommendation_score`;
- `experiment_recommendation_score_component`.

Advisory evaluation persists separately to:

- `experiment_recommendation_evaluation_run`;
- `experiment_recommendation_evaluation_result`;
- `experiment_recommendation_evaluation_component`.

Evaluation recomputes the same score formula after checking current provenance and duplicates.

## Campaign plans

A `RecommendationCampaignCandidateInput` combines:

- durable ranking member and global ordinal;
- ranking bucket and score;
- source metric snapshots;
- recommendation semantic/invocation identity;
- optional campaign Donchian arm;
- prior conversion workflow evidence;
- the currently unpopulated profitability fields.

Campaign planning does not modify the recommendation score.

## Materialized experiments

Campaign materialization creates or reuses conversion proposals, not experiments. Actual experiment creation/reuse occurs during launch/dispatch:

```cpp
LaunchRecommendationCampaignInTransaction(...)
```

The dispatch transaction performs conversion execution, experiment creation/reuse, activation, reservation commitment, and durable Campaign Operations binding. Evidence: [`CampaignOperationsDispatchService.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:557>).

# 4. Eligibility and Filtering

## Source discovery

The source SQL requires:

```sql
e.status = 'completed'
AND e.phase = 'done'
```

and joins analysis with:

```sql
a.experiment_id = e.experiment_id
AND a.model_id = e.last_model_id
AND COALESCE(a.analysis_scope,'final') = 'final'
```

It also supports source-experiment, symbol, and horizon filters and orders by `experiment_id ASC`. Evidence: [`ExperimentRecommendationRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:421>).

The loader skips:

- input-width expansion experiments;
- missing `last_model_id`;
- missing matching final analysis;
- non-completed analysis;
- invalid or ambiguous configuration/date mappings.

## Source eligibility

`EvaluateRecommendationSource` applies, in order:

1. recommendation policy enabled and valid;
2. positive source experiment ID;
3. valid semantic configuration;
4. valid invocation configuration;
5. leader score present and finite;
6. inference accuracy present and finite;
7. evidence count at least `minimumEvidenceCount`;
8. leader score at least `minimumLeaderScore`;
9. inference accuracy at least `minimumInferenceAccuracy`;
10. if maximum neutral proportion is configured, neutral proportion must exist;
11. any present neutral proportion must be finite and within `[0,1]`;
12. present neutral proportion must not exceed the configured maximum.

Defaults are:

- minimum leader score: `0`;
- minimum inference accuracy: `0`;
- maximum neutral proportion: `0.80`;
- minimum evidence: `1`.

Evidence: [`EvaluateRecommendationSource`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:220>).

Notably, this stage does not enforce inference accuracy ≤ 1. The scoring stage does.

## Evaluation gates

Evaluation precedence is fail-closed:

```text
invalid persisted evidence
  > unsupported mutation family
  > missing model/analysis/sample evidence
  > stale source provenance
  > exact experiment duplicate
  > advisory-ready score
```

Staleness includes any mismatch in:

- scan status;
- source experiment status/phase;
- current analysis status/scope;
- current `last_model_id`;
- current analysis ID.

Evidence: [`ExperimentRecommendationEvaluation.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:334>).

## Exact experiment conflicts

Evaluation reconstructs canonical identities for experiments in:

```sql
status IN ('pending','running','paused','completed')
```

Failed and cancelled experiments do not block evaluation here. Conflict precedence is pending, active, then completed, with experiment ID as the secondary key.

## Ranking admission

All evaluated rows may enter the durable ranking snapshot, but they are bucketed:

- `advisory_ready`;
- `blocked`;
- `non_actionable`.

Campaign planning later excludes every non-`advisory_ready` member.

# 5. Complete Metric Inventory

## Performance-derived metrics

| Metric | Source and type | Production | Treatment and direction | Scope |
|---|---|---|---|---|
| `leader_score` | `experiment_analysis_result.leader_score`, nullable `double precision`; snapshotted non-null into `experiment_recommendation.source_leader_score` | `ComputeLeaderScore` | Required/finite; source rank descending; scoring clamps to `[0,1]`, default weight `0.25`; campaign minimum gate; higher preferred | Final in modern workflow |
| `infer_accuracy` | `experiment_analysis_result.infer_accuracy`, nullable `double precision`; snapshotted non-null | Final structured inference result or inference-log fallback | Required/finite; source rank descending; scoring requires `[0,1]`, weight `0.25`; campaign minimum gate; higher preferred | Final |
| `pred_down_count`, `pred_neutral_count`, `pred_up_count` | nullable `bigint` analysis columns | Confusion-matrix aggregation during analysis | Not independently scored; produce evidence count and neutral proportion | Final |
| `source_evidence_count` | sum of `COALESCE(pred_*_count,0)`, persisted `bigint` | Source SQL | Generation minimum; scoring minimum; linear saturation, weight `0.15`; higher preferred until saturation | Final-derived |
| `source_predicted_neutral_proportion` | `pred_neutral_count / sum(pred_*_count)`, nullable `double` | Source SQL | Generation maximum gate; scoring balance component, weight `0.10`; campaign requires present and below maximum; lower neutral dominance preferred | Final-derived |
| `accept_accuracy` | nullable analysis metric | Analysis sets it to inference accuracy in the current structured path | Enters modern recommendation only indirectly through `leader_score` | Final-derived |
| prediction imbalance | derived from confusion counts | `PredictionImbalancePenalty` | Enters only through `leader_score`; dominance above 60% reduces leader score, floor `0.25` | Final-derived |
| profitability | no database column/source in this pipeline; optional C++ `double` plus identity string | Never populated by repository | Ignored when threshold unset; missing-value exclusion when threshold set | No current scope |

If the prediction-count total is zero, neutral proportion is `NULL`. If neutral count itself is `NULL` while down/up counts make the total positive, the numerator remains SQL `NULL`, so the ratio is also `NULL`.

Other stored analysis metrics—train accuracy, validation accuracy, loss, accept rate, best metric, actual counts, and individual confusion cells—do not directly enter modern recommendation scoring, ranking, or campaign planning.

## Structural/scoring fields

| Field | Type/source | Effect |
|---|---|---|
| `source_rank` | positive integer assigned within source group | Reciprocal component `1/source_rank`, weight `0.05`; also candidate-generation ordering |
| `absolute_delta` | non-negative double | Fallback distance and mutation penalty when relative delta is unavailable |
| `relative_delta` | optional non-negative double | Preferred distance; lower preferred |
| `horizon_delta` | optional integer | Horizon penalty numerator; required for horizon mutations |
| source horizon | positive integer | Denominator for normalized horizon penalty |
| changed parameter | enum/text | Chooses parameter preference and horizon penalty; controls mutation-family ranking |
| parameter preference | configured `[0,1]` per family | Positive component, default weight `0.05`; defaults are all `1.0` |
| structural proximity | derived double `[0,1]` | Positive component, default weight `0.15`; smaller mutation preferred |
| mutation penalty | derived double `[0,1]` | Penalty, default weight `0.10`; smaller mutation preferred |
| horizon penalty | derived double `[0,1]` | Penalty, default weight `0.05`; zero for non-horizon mutations |
| `generation_ordinal` | positive integer | Identity/validity evidence; not a score term |
| `structural_rank` | positive integer | Identity/validity and persistence order; despite its name, it is not a score term |
| semantic/invocation/policy canonical text | non-empty strings | Eligibility, determinism, deduplication, replay, and final tie-break evidence |
| duplicate type | text | Must be valid evidence; generation generally suppresses duplicates before scoring |

All weights may be zero individually, including currently nonzero defaults. The combined first seven positive weights must remain greater than zero.

## Ranking and campaign fields

| Field | Effect |
|---|---|
| `final_score` | Primary ordering within the `advisory_ready` durable ranking bucket |
| evaluation disposition | Determines ranking bucket and non-ready ordering |
| ranking global ordinal | Primary campaign-plan order and therefore quota-admission priority |
| ranking bucket | Must be `advisory_ready` with a score for campaign inclusion |
| leader/inference snapshots | Campaign hard minimums; both must also be finite unit values |
| neutral proportion | Missing always excludes at campaign planning; finite value above maximum excludes |
| workflow state/integrity | Prior proposals/reviews/executions/activations may exclude |
| symbol/horizon/source experiment | Scope filtering and sequential per-group quotas |
| maximum selected/candidate count | Top-N campaign admission |
| target epochs | Carried as evidence; does not contribute to score |
| Donchian campaign arm | Optional campaign expansion and conversion identity change |
| profitability metric and identity | Paired optional fields; currently both always absent from repository loading |

# 6. Exact Scoring Formulas

## Analysis-produced leader score

For final and checkpoint analysis, the underlying formula is:

```text
acceptAccuracy = accept_accuracy if present else infer_accuracy

maxPredFrac =
  max(pred_down_count, pred_neutral_count, pred_up_count) / total_predictions

imbalancePenalty =
  1                                           if no confusion or total = 0
  1                                           if maxPredFrac <= 0.60
  max(0.25, 1 - (maxPredFrac - 0.60) / 0.40) otherwise

leaderScore =
  infer_accuracy
  * (0.75 + 0.25 * acceptAccuracy)
  * imbalancePenalty
```

Evidence: [`ComputeLeaderScore`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:10271>).

Current structured inference application sets `acceptAccuracy = inferAccuracy`, so this usually simplifies to:

```text
leaderScore =
  inferAccuracy * (0.75 + 0.25 * inferAccuracy) * imbalancePenalty
```

## Source experiment ranking

Within the configured source grouping, eligible sources are ordered lexicographically by:

```text
leader_score DESC
infer_accuracy DESC
evidence_count DESC
experiment_id ASC
```

No combined source score is computed. The top `topSourcesPerScope` are retained.

Scopes are:

- symbol + horizon, default;
- symbol;
- global.

## Candidate recommendation score

Define:

```text
clamp01(x) = min(1, max(0, x))
P = sum of the seven positive weights
```

Components:

```text
L = clamp01(sourceLeaderScore)

A = sourceInferenceAccuracy
    # validation requires 0 <= A <= 1

E =
  1, if evidenceSaturationCount == minimumEvidenceCount
  clamp01(
    (evidenceCount - minimumEvidenceCount)
    / (evidenceSaturationCount - minimumEvidenceCount)
  ), otherwise

N =
  0.5, if neutral is missing and missing is allowed
  1.0, if neutral <= preferredNeutralProportion
  0.0, if maximumNeutralProportion == preferredNeutralProportion
  clamp01(
    (maximumNeutralProportion - neutral)
    / (maximumNeutralProportion - preferredNeutralProportion)
  ), otherwise

D =
  relativeDelta, if present
  absoluteDelta, otherwise

Dmax =
  maximumRelativeMutation, if relativeDelta is present
  maximumAbsoluteStructuralDistance, otherwise

S = clamp01(1 - D / Dmax)

Q = configured preference for changed parameter

R = 1 / sourceRank

H =
  0, for non-horizon mutations
  clamp01(abs(horizonDelta) / sourcePredictionHorizon), for horizon mutation

M =
  clamp01(relativeDelta / maximumRelativeMutation), if relativeDelta exists
  clamp01(absoluteDelta / maximumAbsoluteStructuralDistance), otherwise
```

Then:

```text
rawPositive =
  (wL*L + wA*A + wE*E + wN*N + wS*S + wQ*Q + wR*R) / P

rawPenalty =
  (wH*H + wM*M) / P

rawTotal = rawPositive - rawPenalty

finalScore = clamp(rawTotal, scoreFloor, scoreCeiling)
```

Default weights make `P = 1.0`, producing:

```text
finalScore = clamp[0,1](
    0.25*L
  + 0.25*A
  + 0.15*E
  + 0.10*N
  + 0.15*S
  + 0.05*Q
  + 0.05*R
  - 0.05*H
  - 0.10*M
)
```

Evidence: [`ExperimentRecommendationScoring.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:370>).

Important details:

- Penalties are divided by the positive weight sum, not by the penalty-weight sum.
- Every normalized component is clamped again in `AddComponent`.
- Leader score is clamped rather than rejected when outside `[0,1]`.
- Structural proximity and mutation penalty use the same distance and maximum in opposing directions, so mutation magnitude affects the total twice.
- Missing neutral evidence receives a neutral score of `0.5` only when `allowMissingNeutralProportion=true`.
- Default evidence count `1` normalizes to `0`; it does not receive full evidence credit.
- Non-finite source metrics, deltas, components, or totals invalidate the score.

## Advisory evaluation score

Evaluation does not define a second formula. After provenance, evidence, and duplicate gates, it invokes the same `ScoreExperimentRecommendation` function.

## Campaign planning/admission score

There is no campaign score formula. Campaign planning preserves durable ranking order and applies gates/quotas. It does not combine leader, inference, neutral, or profitability into a new number.

## Legacy meta-analyzer score

The separate legacy recommender uses:

```text
legacyRankingScore =
  finite leaderScore, if present
  finite inferAccuracy, otherwise
  -1, otherwise
```

It requires the score to be positive. It then orders descending by this single value and ascending by synthetic/real experiment ID.

# 7. Ranking and Tie-Breaking

## Modern source selection

```text
leader DESC
inference DESC
evidence DESC
experiment_id ASC
```

`std::sort` is used, but the final ID key gives distinct sources a total deterministic order.

## Generated-candidate order

Within one source:

```text
parameter family:
  core=0, head=1, label=2, horizon=3
proposed value ASC
semantic canonical text ASC
invocation canonical text ASC
```

Across selected sources:

```text
group key ASC
source rank ASC
parameter family ASC
proposed value ASC
semantic canonical ASC
invocation canonical ASC
source experiment ID ASC
```

The scan limit is applied after this ordering. An explicit `--recommendation-max` replaces, rather than caps against, `maximumRecommendationsPerScan`.

## Standalone score ranking

Exact comparator:

```text
final score DESC
raw positive score DESC
raw penalty score ASC
source leader score DESC
source inference accuracy DESC
source evidence count DESC
structural distance ASC
semantic canonical text ASC
recommendation policy canonical text ASC
recommendation ID ASC
```

`scoreRank` uses competition ranking based only on exact `double` equality of final score. `tieGroup` changes only when exact final score changes. `rankingOrdinal` is always the total-order position.

## Evaluation-run local order

`RankRecommendationEvaluations` orders:

```text
has score before missing score
final score DESC
disposition text ASC
evaluation identity canonical ASC
recommendation ID ASC
```

This ordinal is persisted in the evaluation run but is not the durable campaign ranking.

## Durable ranking snapshot

Bucket order:

```text
advisory_ready
blocked
non_actionable
```

Within `advisory_ready`:

```text
final score DESC
semantic hash ASC
evaluation identity hash ASC
evaluation result ID ASC
```

Within blocked:

```text
pending duplicate
active duplicate
completed duplicate
semantic hash
evaluation hash
evaluation result ID
```

Within non-actionable:

```text
insufficient evidence
stale source
unsupported family
invalid evidence
semantic hash
evaluation hash
evaluation result ID
```

The output limit is applied after sorting all buckets. `bucketRank` is a simple ordinal, not a tied rank.

## Campaign-plan order

```text
ranking global ordinal ASC
recommendation ID ASC
ranking member ID ASC
canonical candidate evidence ASC
```

Quotas are applied in that order, so ranking position controls which candidate wins symbol/horizon/source/campaign capacity.

## Campaign Manager order

Manager dispatch is independent of recommendation score:

```text
operational_request_id ASC
```

## Determinism assessment

For the modern pipeline, repeated evaluation of identical persisted state should produce the same order. SQL membership reads and C++ comparators supply deterministic secondary keys.

Exceptions/qualifications:

- Floating-point equality is exact; numerically close values are not ties.
- Durable ranking uses semantic hash before evaluation identity and ID. Hash collisions do not destroy determinism, but may give an arbitrary semantic ordering relative to canonical text.
- A ranking scope can contain evaluation results produced by different scoring-policy identities; the ranking code does not require policy homogeneity.
- Legacy `DetectPlateaus` sorts only by `abs(delta)`. Equal absolute deltas have no secondary key, and `std::sort` is unstable. This affects advisory presentation order, not the modern campaign ranking.

# 8. Metric Provenance and Final-vs-Checkpoint Scope

## Modern source provenance

Confirmed executable predicates:

```sql
a.experiment_id = e.experiment_id
AND a.model_id = e.last_model_id
AND COALESCE(a.analysis_scope,'final') = 'final'
```

Source loading also requires `analysis_status='completed'` in C++.

Evaluation repeats the same model join and then requires:

```cpp
currentSourceAnalysisScope == "final"
sourceModelId == currentSourceModelId
sourceAnalysisId == currentSourceAnalysisId
```

Therefore a checkpoint row cannot directly satisfy the modern source or evaluation path.

Migration 016 makes `analysis_scope` `NOT NULL DEFAULT 'final'` and creates:

```sql
UNIQUE (experiment_id, model_id)
WHERE analysis_scope='final'
```

plus a separate checkpoint-evaluation uniqueness index. Evidence: [`016_checkpoint_analysis_scope.sql`](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/016_checkpoint_analysis_scope.sql:1>).

## Raw final metric origin

Final analysis initially parses inference logs, then overlays a structured inference result when one exists with:

```sql
model_id = experiment.last_model_id
inference_scope = 'final'
checkpoint_eval_id IS NULL
status = 'completed'
```

and exact symbol, horizon, threshold, and inference date range. Evidence: [`ApplyStructuredInferenceMetrics`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9997>).

Thus final analysis metrics can originate from:

- a durable final `inference_eval_result`; or
- the final inference log fallback.

The recommendation layer sees the final analysis result rather than distinguishing those two raw production sources.

## Scope conclusion

Phase 1’s conclusion remains correct for the current persisted recommendation/campaign workflow:

> Checkpoint inference/analysis does not directly enter recommendation generation, scoring, evaluation, ranking, planning, approval, materialization, or Campaign Manager dispatch. Final inference/analysis may enter.

## Legacy exception

`ExperimentMetaAnalyzer::LoadExperimentRecords` filters final analysis scope but joins only:

```sql
a.experiment_id = e.experiment_id
AND COALESCE(a.analysis_scope,'final')='final'
```

It does not require `a.model_id = e.last_model_id`. A completed experiment can therefore produce multiple legacy meta-analysis records or admit a stale final-model analysis.

This is separate from the modern campaign workflow, but it is a genuine final-model identity ambiguity. Evidence: [`ExperimentMetaAnalyzer.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:400>).

# 9. Trend and Historical Logic

## Modern pre-campaign trend logic

No performance trend, moving average, slope, lineage history, or cross-generation comparison enters modern recommendation source selection, scoring, durable ranking, or initial campaign planning.

Modern history at planning time is workflow history only:

- proposal state;
- latest review;
- execution;
- activation;
- converted experiment;
- consistency/integrity diagnostics.

These states gate reconsideration but do not change the score.

## Legacy plateau analysis

The legacy meta-analyzer groups completed records by:

```text
symbol
prediction horizon
threshold
core LR
head LR
```

For each group:

```text
epoch = completedEpochs if present else targetEpochs
score = leaderScore if optional is present else inferAccuracy
```

Only positive finite values participate.

It computes the mean score at each distinct epoch, then compares only the two greatest epochs:

```text
delta = mean(latest epoch) - mean(previous epoch)
```

Classification:

```text
delta < -0.005          -> regression
abs(delta) <= 0.005     -> plateau
delta >= 0.015          -> consistent_improvement
otherwise               -> small_improvement
```

At least two distinct epochs are required. Confidence is based on total record count across all epochs:

- ≥30 very high;
- ≥15 high;
- ≥8 moderate;
- ≥3 low;
- otherwise insufficient.

The plateau signal only generates narrative `BuildRecommendations` advice. It does not affect `BuildNextExperimentRecommendations` candidate ordering or the modern campaign path.

Evidence: [`DetectPlateaus`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:823>).

## Post-campaign historical comparison

Phase 5 outcome assessment compares each materialized member’s source and resulting experiment using:

```text
delta = resultValue - sourceValue
```

for `inference_accuracy` and `leader_score`.

Comparison is allowed only when context agrees on symbol, horizon, threshold, inference window/label definition, and inference date range. Both inference and analysis repository queries require exact final scope; inference also requires `checkpoint_eval_id IS NULL`.

The outcome policy uses exact sign:

- positive delta favorable for higher-is-better metrics;
- zero neutral;
- negative unfavorable.

No materiality threshold, confidence interval, weighting, or statistical trend is applied. A follow-up proposal becomes operator-review-eligible only when evidence is sufficient, all members are comparable, and the campaign interpretation is favorable.

This is post-campaign follow-up evidence, not initial recommendation ranking.

# 10. Candidate Mutation and Deduplication

Default mutable parameters and mutations:

| Parameter | Source | Default proposed values |
|---|---|---|
| `core_lr_mult` | optional source double | source `-0.25`, source `+0.25` |
| `head_lr_mult` | optional source double | source `-0.5`, source `+0.5` |
| `label_threshold` | source double | source `-0.0001`, source `+0.0001` |
| `prediction_horizon` | source integer | explicit configured permitted horizons; disabled by default |

Rules:

- offsets/horizons are sorted and deduplicated;
- proposed numeric values must be finite and positive;
- unchanged values are rejected;
- missing optional source LR values reject their mutations;
- ordinary generation must change exactly one semantic field;
- mutations do not compound;
- source checkpoint interval and resume model are retained in invocation identity;
- semantic and invocation identities remain separate.

Campaign Donchian-arm expansion can deliberately add a second semantic difference after the recommendation’s one ordinary mutation. Conversion validates this as:

```text
expected differences = 1 + campaign-arm-change
```

## Deduplication layers

1. Generated candidates: exact semantic canonical text; hashes only detect/report collisions.
2. Existing experiment: reconstructed exact semantic canonical identity. Invocation equality is diagnostic; semantic equality alone suppresses.
3. Terminal failed/cancelled experiment: suppressed by default, configurable through `terminalExperimentsAreDuplicates`.
4. Existing recommendation: exact semantic canonical plus exact recommendation-policy canonical.
5. Active and historical matching recommendations both suppress generation.
6. Campaign planning: duplicate conversion invocation plus campaign arm is suppressed.
7. Conversion materialization: exact conversion canonical identity is enforced again.

The existing-experiment lookup first narrows by symbol, horizon, and target epochs, then reconstructs canonical identity. It orders non-terminal rows before failed/cancelled rows and by experiment ID.

# 11. Campaign Planning and Admission

## Planning defaults

The header default is disabled, but CLI plan/review/approve/reject commands explicitly enable it.

Defaults:

- maximum selected: `10`;
- maximum candidates considered: `100`;
- minimum leader: `0`;
- minimum inference accuracy: `0`;
- maximum neutral: `0.80`;
- minimum profitability: unset;
- maximum per source experiment: `1`;
- per-symbol/horizon limits: unset;
- completed workflows exclude;
- rejected/failed/cancelled workflows are not reconsidered;
- inconsistent workflows always exclude.

CLI configuration is parsed in [`ExperimentScheduler.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:2662>).

## Planning decisions

Every candidate first accumulates global reasons:

- policy disabled;
- outside candidate limit;
- missing/mismatched rank;
- non-advisory bucket or absent score;
- unsupported ranking contract;
- invalid or inconsistent identity/provenance;
- leader/inference below minimum;
- neutral missing or above maximum;
- profitability unavailable/below threshold;
- workflow lifecycle/integrity exclusions.

Only candidates with no reason then compete sequentially for:

- duplicate conversion identity;
- per-source quota;
- per-symbol quota;
- per-horizon quota;
- overall campaign quota.

Only selected candidates increment counters.

## Approval and materialization

Campaign approval requires:

- exact review identity hash supplied by the operator;
- plan/review reconstruction equality;
- nonzero selected count for an approval decision;
- immutable canonical approval evidence.

Materialization reconstructs the current plan, review, and approval. Any mismatch causes stale-approval rejection.

Additionally, each selected recommendation still requires its own latest manual recommendation approval and a completed standalone score linked through that review. Campaign approval does not replace per-recommendation approval.

Conversion requires:

- recommendation status approved;
- effective latest manual approval;
- completed advisory-ready evaluation;
- completed finite standalone score;
- evaluation scoring-policy hash equal to standalone score-policy hash;
- exactly one recommendation mutation;
- exact source/proposed/canonical provenance.

## Campaign Operations admission

After materialization:

1. `AdmitOperationalCampaign` binds the immutable materialization and member count.
2. Authorization and an active budget are established.
3. `AcceptOperationalRequest` checks future-action controls and reserves one budget unit per materialized member.
4. Production admission validates the manager build/scheduler protocol contract.
5. `RUN_ONCE` dispatches ready requests in request-ID order.

No recommendation metric, profitability metric, or rank is recalculated at these operational stages.

# 12. Profitability Placeholder Audit

Relevant executable occurrences are confined to campaign planning:

- `RecommendationCampaignPlanningPolicy::minimumProfitability`;
- `RecommendationCampaignCandidateInput::profitabilityMetric`;
- `profitabilityMetricIdentity`;
- CLI `--campaign-min-profitability`;
- candidate identity validation;
- plan reasons;
- plan logs and canonical evidence.

No relevant authoritative schema column, producer, SQL selection, scoring component, ranking field, P&L, expectancy, win-rate, or trade-count metric exists in the current recommendation/campaign implementation.

Classification:

| Occurrence | Status |
|---|---|
| Policy field and canonical parsing | Executable |
| CLI option | Executable |
| Plan threshold comparison | Executable |
| Candidate profitability/identity fields | Executable API placeholder |
| Planning repository population | Absent; fields remain `NULL` |
| Score formula | No profitability term |
| Durable ranking | No profitability key |
| Database schema | No authoritative metric |
| Planning tests | Test-only synthetic future metric |
| Phase 4D documentation | Correctly documents placeholder/fail-closed state |
| Other profitability references | Documentation disclaimers, not claims or calculations |

Current behavior:

1. **`minimumProfitability` unset:** profitability is ignored. Missing metric and identity do not exclude.
2. **`minimumProfitability` configured:** every repository-loaded candidate receives `profitability_metric_unavailable`; therefore no candidate can be selected.
3. **Metric `NULL`:** same as above when threshold configured; harmless when unset.
4. **Hypothetical finite metric supplied with identity:** value `< minimum` excludes; equality or greater passes this gate.
5. **Finite metric without identity, or identity without metric:** `identityValidationFailed`.
6. **Non-finite metric:** `identityValidationFailed`.
7. **Non-finite minimum:** policy validation fails.

The executable explanation says:

> “No authoritative profitability metric exists in the current schema.”

This agrees with current behavior.

# 13. Scheduler-Continuation Boundary

The systems remain distinct.

Shared elements:

- `experiment`;
- `experiment_analysis_result`;
- `leader_score`;
- `infer_accuracy`;
- underlying analysis-production helpers and canonical hashing patterns.

Not shared:

- policies;
- scoring functions;
- ranking comparators;
- evidence identities;
- persistence tables;
- candidate mutations;
- approvals;
- campaign operations.

Continuation loads checkpoint and final evidence explicitly, deduplicates by epoch, and may prefer final evidence at the same epoch. Its trend formula is:

```text
window = last patience distinct epochs

metric =
  leader_score if all window points have leader score
  infer_accuracy if all window points have inference accuracy
  insufficient otherwise

trend = latest - first

non_degrading passes if trend >= -maxDegradation
improving passes if trend >= minImprovement
```

Continuation then applies its own minimum metrics, `topN`, trend, minimum-evaluation, and lineage rules. Evidence: [`EvaluateContinuationTrend`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12544>).

Future metric separation should be:

- checkpoint profitability, if ever defined: checkpoint-policy or continuation evidence tied to checkpoint model/eval/epoch;
- final profitability: recommendation/campaign evidence tied to final model, final inference result, and final inference interval.

They should not be merged merely because both may represent economic performance.

# 14. Tests and Observability

## Existing tests

The repository has substantial coverage for:

- policy parsing/canonical identity: `ExperimentRecommendationTests.cpp`;
- source eligibility, missing/non-finite metrics, mutations, one-field rule, deterministic ordering, canonical dedup and hash collisions: `ExperimentRecommendationCandidateGeneratorTests.cpp`;
- source loading, final-model identity, missing/stale analysis, neutral/evidence extraction and duplicate semantics: `ExperimentRecommendationRepositoryTests.cpp`;
- all score components, weights, normalization, missing neutral, penalties, clamping and every standalone tie-break: `ExperimentRecommendationScoringTests.cpp`;
- scoring persistence/idempotence: `ExperimentRecommendationScoringRepositoryTests.cpp` and scoring migration tests;
- evaluation provenance, missing/stale evidence, duplicates, non-finite handling, precedence and deterministic ordering: `ExperimentRecommendationEvaluationTests.cpp`;
- explicit checkpoint/other-final-model isolation in repository loading: `ExperimentRecommendationEvaluationRepositoryTests.cpp`;
- ranking bucket/disposition order, exact score ties, hash collisions, signed zero, non-finite rejection, truncation and comparison: `ExperimentRecommendationRankingTests.cpp`;
- durable ranking persistence/immutability and read-only separation from checkpoint/continuation evidence: `ExperimentRecommendationRankingRepositoryTests.cpp`;
- campaign limits, workflow states, neutral/leader/inference gates, missing profitability, below-threshold hypothetical profitability, deterministic ordering and Donchian arms: `ExperimentRecommendationCampaignPlanningTests.cpp`;
- read-only planning repository behavior: `ExperimentRecommendationCampaignPlanningRepositoryTests.cpp`;
- approval reconstruction, materialization, conversion, launch, Campaign Operations admission and Manager RUN_ONCE through their corresponding Phase 4–H test suites;
- final-only post-campaign outcome comparison and follow-up policy through outcome/follow-up tests.

## Missing tests before profitability implementation

At minimum, Phase 3 should add tests for:

- authoritative final-profit metric extraction from the intended schema;
- explicit rejection of checkpoint profitability;
- exact `experiment.last_model_id` binding;
- wrong model, wrong interval, wrong symbol/horizon/threshold, and checkpoint-eval contamination;
- metric identity/value pairing;
- unset threshold with missing metric;
- configured threshold with missing metric;
- finite below, equal, and above threshold;
- `NaN` and infinities;
- trade/actionable sample minimums;
- source snapshot immutability and staleness after source evidence changes;
- profitability normalization boundaries and clipping;
- weighted score term with zero and nonzero weights;
- exact ranking tie-break behavior if profitability participates;
- campaign-plan identity changes when profitability identity or value changes;
- policy-version migration and historical replay;
- end-to-end materialization/dispatch proving Campaign Manager consumes the approved plan without recalculating profitability;
- mixed scoring-policy ranking behavior if that remains allowed.

## Current observability

Current machine-readable output exposes:

- source skip reasons and source rank;
- leader, inference, and evidence count;
- mutation parameter/source/proposed value and structural rank;
- duplicate classifications and hash collisions;
- every normalized score component, weight, contribution, and penalty flag;
- raw positive, raw penalty, raw total, final score, tie group, score rank, ordinal;
- evaluation disposition, reason, missing-evidence count, identities, components;
- durable bucket, bucket rank, global ordinal, tie-break fields, top positive/penalty component;
- campaign candidate metrics, profitability placeholder, rank, workflow state/integrity, decision, and all reason codes;
- approval/materialization member identities;
- Manager request outcome, replay classification, operation key, and diagnostics.

The primary markers include:

- `EXPERIMENT_RECOMMENDATION_SOURCE_*`;
- `EXPERIMENT_RECOMMENDATION_SCORE_COMPONENT`;
- `EXPERIMENT_RECOMMENDATION_SCORED`;
- `EXPERIMENT_RECOMMENDATION_EVALUATION`;
- `EXPERIMENT_RECOMMENDATION_RANKING_MEMBER`;
- `RECOMMENDATION_CAMPAIGN_PLAN_CANDIDATE`;
- `RECOMMENDATION_CAMPAIGN_MATERIALIZATION_MEMBER`;
- `CAMPAIGN_OPERATIONS_MANAGER_REQUEST`.

# 15. Risks / Ambiguities / Potential Defects

1. **Legacy final-model ambiguity.** `ExperimentMetaAnalyzer` does not bind final analysis to `experiment.last_model_id`.
2. **Analysis rows are upserted in place.** Final analysis uses `ON CONFLICT ... DO UPDATE`, retaining the same analysis ID while replacing metric values. Modern evaluation checks analysis/model IDs, not equality between live metric values and the recommendation snapshot. A re-analysis of the same experiment/model could therefore change the live row without marking an existing recommendation stale.
3. **Leader range inconsistency.** Source eligibility accepts any finite leader score and scoring clamps it to `[0,1]`; campaign identity validation requires it to be inside `[0,1]`. A recommendation may score successfully but later fail campaign identity validation.
4. **Inference range enforcement is delayed.** Source eligibility does not reject inference accuracy above `1`; scoring does.
5. **Neutral missing-value policies differ.** Source policy can disable the maximum-neutral requirement, and scoring can assign missing neutral a `0.5` component, but campaign planning always excludes a missing neutral metric.
6. **Mixed scoring policies can be ranked together.** Durable ranking does not require all evaluation rows in a scope to share scoring/evaluation policy identity. Final scores from materially different policies can therefore be compared.
7. **Explicit scan limit replaces configured maximum.** `requestedMaximum` is not capped by `maximumRecommendationsPerScan`.
8. **Legacy direct queueing bypasses the modern workflow.** `QueueMetaRecommendations` inserts pending experiments directly with hard-coded date ranges/checkpoint interval and default feature settings. It does not create recommendations, evaluations, ranking snapshots, approvals, or campaign materializations.
9. **Legacy duplicate identity is narrower.** It compares symbol, horizon, epochs, threshold, and two LR values, omitting several modern semantic fields and date ranges.
10. **Legacy plateau ordering lacks a complete tie-break.**
11. **Profitability is fail-closed but operationally unusable when enabled.** This is documented, but an operator may interpret the CLI option as presently supported.
12. **Evidence count is prediction count, not trade count.** It cannot establish profitability reliability by itself.
13. **Final metric provenance can fall back to logs.** The recommendation layer does not record whether analysis metrics came from a structured inference row or parsed log output.
14. **`COALESCE(analysis_scope,'final')` retains legacy-null semantics.** Current migration makes scope non-null, but the executable predicate conceptually treats a legacy null as final.

# 16. Recommended Phase 3 Profitability Integration Points

## Required foundation

Before choosing scoring policy, add an authoritative final-profitability evidence contract that records at least:

- exact source experiment ID;
- exact final model ID;
- exact final inference result/evaluation ID;
- explicit `inference_scope='final'`;
- absence of checkpoint evaluation identity;
- symbol, horizon, threshold, label/window definition;
- inference start/end;
- metric definition/version;
- gross versus net result;
- transaction-cost/slippage assumptions;
- actionable prediction/trade count;
- total inference observations;
- currency or return normalization;
- provenance/canonical identity.

The recommendation row should snapshot both value and identity, analogous to existing source model/analysis/metric snapshots.

## Alternative 1: hard eligibility threshold

Clean insertion points:

- generation source gate in `EvaluateRecommendationSource`; and/or
- campaign admission gate in `PlanRecommendationCampaign`, where the placeholder already exists.

Trade-off:

- source gating prevents all mutations from economically weak sources;
- campaign gating preserves advisory recommendations but prevents authorization.

Campaign planning is the smallest already-shaped integration. Source gating is cleaner if profitability is fundamental evidence rather than an operator admission preference.

## Alternative 2: weighted scoring term

Extend:

- `RecommendationScoringPolicy`;
- `RecommendationScoringInput`;
- `ScoreExperimentRecommendation`;
- score/evaluation component schema constraints;
- policy canonical versions;
- score/evaluation persistence and logs.

A profitability normalization function must be decided first. Raw P&L should not be inserted directly because symbols, horizons, capital bases, and inference intervals are not comparable.

The existing zero-weight capability supports a staged rollout: persist and expose the component first with weight zero, then deliberately activate it under a new policy identity.

## Alternative 3: ranking/tie-break metric

Possible locations:

- standalone `RankRecommendationScores`;
- durable `RankRecommendationEvaluationEvidence`.

This requires profitability to be present in persisted evaluation/ranking evidence. A ranking-policy version bump should make the tie-break explicit.

This is lower-impact than a weighted score but affects candidates only when preceding keys tie. It can also conceal economically significant differences if used only as a late tie-break.

## Alternative 4: trend/history evidence

There is no modern pre-campaign history subsystem to extend. A new one should use only comparable final records and should define:

- partition key;
- lineage treatment;
- lookback depth;
- interval normalization;
- minimum trades/samples;
- delta versus slope;
- materiality threshold;
- missing-history behavior.

The post-campaign outcome framework is a natural place to compare source/result profitability deltas for follow-up decisions, but it should not be mistaken for initial recommendation scoring.

## Recommended architectural combination

The smallest defensible progression is:

1. authoritative final profitability plus trade/actionable sample evidence;
2. snapshot value and identity into recommendation provenance;
3. fail-closed campaign minimum using the existing planning hook;
4. expose it in evaluation/ranking observability;
5. only after normalization policy is validated, add a versioned weighted component or explicit tie-break;
6. separately extend post-campaign outcome assessment if profitability should influence follow-up proposals.

No current repository evidence is sufficient to select a final profitability formula.

# 17. Files / Functions / SQL / Tables Inspected

Primary implementation:

- [`ExperimentRecommendation.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:1>)
- [`ExperimentRecommendationRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:409>)
- [`ExperimentRecommendationService.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:249>)
- [`ExperimentRecommendationCandidateGenerator.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:220>)
- [`ExperimentRecommendationScoring.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:321>)
- [`ExperimentRecommendationScoringService.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoringService.cpp:155>)
- [`ExperimentRecommendationEvaluation.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:325>)
- [`ExperimentRecommendationEvaluationRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:287>)
- [`ExperimentRecommendationRanking.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:406>)
- [`ExperimentRecommendationRankingRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingRepository.cpp:353>)
- [`ExperimentRecommendationCampaignPlanning.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:881>)
- [`ExperimentRecommendationCampaignPlanningRepository.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:97>)
- campaign review, approval, materialization, conversion, execution, activation, launch, status, outcome, and follow-up implementation/repositories;
- Campaign Operations foundation, admission, budget/request, production admission, dispatch, completion, and Manager services/repositories;
- [`ExperimentScheduler.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9997>) for analysis production, CLI configuration, checkpoint and continuation boundaries;
- [`ExperimentMetaAnalyzer.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/ExperimentMetaAnalyzer.cpp:400>) for the separate legacy recommendation path.

Key migrations/tables:

- `006_experiment_analysis.sql`: `experiment_analysis_result`;
- `016_checkpoint_analysis_scope.sql`: final/checkpoint scope and uniqueness;
- `026`–`035`: recommendation, score, evaluation, ranking;
- `036`–`041`: conversion, campaign approval/materialization;
- `042`–`044`: follow-up proposal workflow;
- `045`, `047`, `048`, `055`, `058`: Campaign Operations admission, budget/request, dispatch, production admission, Manager RUN_ONCE;
- `experiment`;
- `model`;
- `inference_eval_result`;
- `experiment_analysis_result`;
- `experiment_recommendation_*`;
- `campaign_operations_*`.

Tests inspected include the candidate, repository, scoring, evaluation, ranking, campaign-planning, conversion, approval, materialization, outcome/follow-up, and Campaign Operations H1–H3 suites.

# 18. Final Conclusions

The modern recommendation/campaign pipeline is an advisory source-evidence ranking system, not a candidate backtest. Candidate scores inherit final source performance and combine it with deterministic mutation heuristics.

The precise future profitability boundary is:

```text
authoritative FINAL inference profitability
  -> immutable source evidence snapshot
  -> evaluation provenance validation
  -> optional source/campaign hard gate
  -> optional versioned scoring component
  -> optional explicit durable ranking key
```

Checkpoint profitability must remain in checkpoint/continuation evidence and must not populate those final recommendation fields.

Completion status:

- Files changed: none.
- Behavioral change: none.
- Build/tests run: none; this was inspection-only, and no executable, scheduler, worker, or database-mutating test was launched.
- Inspection commands: read-only `rg`, `sed`, `nl`, and Git status/diff commands.
- Remaining assumptions: repository/schema behavior was inspected statically; no live database contents were queried.
- `git diff --stat`: empty.
- `git diff --name-only`: empty.
- `git status --short`:

```text
?? LSTM_CampaignManager_Phase1_CandidateSelection_Inspection_Output.md
?? lstm_watch.sql
```

These untracked files were present at the start and remained untouched.