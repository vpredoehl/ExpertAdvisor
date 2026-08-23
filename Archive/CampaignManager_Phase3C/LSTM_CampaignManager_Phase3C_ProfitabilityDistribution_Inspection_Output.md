---
title: "LSTM Campaign Manager Phase 3C Profitability Distribution Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3C Profitability Distribution Inspection

# Phase 3C Profitability Evidence Inspection Report

## 1. Executive Summary

**Readiness classification: `NOT_READY_INSUFFICIENT_PROFITABILITY_EVIDENCE`**

[Observed fact] At inspection time, PostgreSQL database `LSTM` contained:

- `inference_profitability_observation`: **0 rows**
- Completed FINAL `inference_eval_result` rows: **318**
- Completed FINAL results with any profitability observation: **0**
- Completed FINAL results without an observation: **318**
- Checkpoint profitability observations: **0**

Therefore:

- No authoritative FINAL-profitability distribution exists.
- No percentiles, correlations, cohort effects, outliers, concentration measures, normalization parameters, eligibility thresholds, or scoring weights can be estimated.
- Missing profitability must not be interpreted as zero profitability.
- Phase 3C policy selection is premature.

The absence is attributable where evidence permits:

- **315** completed FINAL results predate migration 073. Migration 073 explicitly does not backfill historical results because exact per-window source content cannot be reconstructed.
- **3** FINAL results were completed after migration 073 but were launched using the older noncanonical `DerivedData/Build/.../LSTM_Release` binary, which lacks the profitability persistence implementation. Their logs contain `SCHEDULER_INFER_RESULT_PERSISTED` but no `INFERENCE_PROFITABILITY_OBSERVATION_PERSISTED`.

[Observed fact] Current Campaign Manager coverage is also inadequate:

- Recommendations: 108 total; 2 Phase-3A-aware; 0 with profitability; 2 explicitly unavailable; 106 legacy.
- Evaluation results: 102 total; all 102 legacy; none has Phase 3A profitability provenance.
- Ranking members: 102; all reference legacy evaluation results.

[Observed fact] Profitability remains non-decision-bearing:

- No persisted scoring/evaluation component is profitability-based.
- All scoring and evaluation policies exclude profitability.
- All three ranking snapshots remain `verified_homogeneous`, with one scoring and one evaluation semantic identity.
- Code continues to declare `profitability_weight=0` and `profitability_score_contribution=0`.

## 2. Inspection Scope

[Observed fact] Inspection began against database `LSTM` at `2026-08-23 07:50:47-05`.

Read-only activities included:

- Repository and migration inspection.
- Source-path tracing.
- Binary string and metadata inspection.
- Log-file inspection.
- PostgreSQL queries wrapped in `BEGIN TRANSACTION READ ONLY` and `ROLLBACK`.
- `git status --short` and `git diff --stat`.

Not performed:

- No executable was launched.
- No scheduler state was inspected or altered through the executable.
- No inference was generated, rerun, requeued, or repaired.
- No source, tests, migrations, scripts, configuration, schema, database rows, or experiment state were modified.
- No build or tests were run because this was inspection-only and the result depends on persisted data, not compilation.

Repository status after inspection:

```text
git status --short
# no output

git diff --stat
# no output
```

Primary source locations:

- Metric definition: [InferenceProfitability.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitability.hpp:15)
- Metric accumulation: [InferenceProfitability.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitability.cpp:69)
- Observation identity and persistence: [InferenceProfitabilityRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:154)
- Authoritative selection: [InferenceProfitabilityRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:310)
- Exact FINAL result resolution: [InferenceProfitabilityRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:412)
- Inference-window integration: [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:2818)
- Transactional result/observation persistence: [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7929)
- Migration 073: [073_inference_profitability_observation.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:1)
- Campaign Manager Phase 3A selection: [ExperimentRecommendationRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:75)
- Phase 3A schema enforcement: [076_campaign_manager_final_profitability_provenance.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/076_campaign_manager_final_profitability_provenance.sql:236)
- Phase 3B homogeneity enforcement: [077_campaign_manager_ranking_semantic_homogeneity.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql:621)

## 3. Current Profitability Evidence Inventory

| Inventory item | Current value |
|---|---:|
| Total observations | 0 |
| FINAL observations | 0 |
| Checkpoint observations | 0 |
| Distinct experiments | 0 |
| Distinct models | 0 |
| Distinct symbols | 0 |
| Distinct horizons | 0 |
| Earliest observation creation | NULL |
| Latest observation creation | NULL |
| Duplicate identity groups | 0 |
| Multiple current-metric observations per result | 0 |
| Rows available for hash verification | 0 |

Exact inventory SQL:

```sql
BEGIN TRANSACTION READ ONLY;

SELECT count(*) total_observations,
       count(*) FILTER (
           WHERE ipo.inference_scope='final') final_observations,
       count(*) FILTER (
           WHERE ipo.inference_scope='checkpoint') checkpoint_observations,
       count(DISTINCT ipo.experiment_id) experiments,
       count(DISTINCT ipo.model_id) models,
       count(DISTINCT ier.symbol) symbols,
       count(DISTINCT ier.prediction_horizon) horizons,
       min(ipo.created_at) earliest_created,
       max(ipo.created_at) latest_created
FROM inference_profitability_observation ipo
LEFT JOIN inference_eval_result ier
  ON ier.id=ipo.inference_eval_result_id;

SELECT count(*) AS duplicate_identity_groups
FROM (
    SELECT observation_identity_canonical
    FROM inference_profitability_observation
    GROUP BY observation_identity_canonical
    HAVING count(*)>1
) duplicate;

SELECT count(*) AS duplicate_current_metric_per_result_groups
FROM (
    SELECT inference_eval_result_id, experiment_id, model_id,
           inference_scope, checkpoint_eval_id,
           metric_definition_canonical, metric_definition_hash
    FROM inference_profitability_observation
    GROUP BY inference_eval_result_id, experiment_id, model_id,
             inference_scope, checkpoint_eval_id,
             metric_definition_canonical, metric_definition_hash
    HAVING count(*)>1
) duplicate;

ROLLBACK;
```

### Identity and hash integrity

[Repository-defined semantic] The schema provides:

- Unique `observation_identity_canonical`.
- Tagged FNV-1a format checks for metric, source-content, and observation hashes.
- Immutable-row trigger.
- Completed-inference provenance trigger.
- Scope/checkpoint shape checks.

The repository constructs the canonical identity from experiment, model, inference result, scope, checkpoint, interval, metric definition, source-content hash, and statistics. It verifies the returned identity/hash after idempotent persistence.

[Observed fact] The two live triggers are enabled, and all 24 observation-table constraints are installed.

[Statistical inference] Data-level integrity cannot be positively verified because there are no rows. “Zero mismatches among zero rows” is vacuous, not proof that a populated dataset will be sound.

## 4. Authoritative FINAL Observation Definition

A Campaign Manager authoritative FINAL observation requires both stages below.

### Stage 1: exact FINAL inference-result resolution

The source experiment/model must resolve to exactly one `inference_eval_result` satisfying:

- `experiment.last_model_id = model.model_id`
- `model.experiment_id = experiment.experiment_id`
- Complete `train_config_meta` model configuration.
- Model horizon equals experiment horizon.
- Model threshold equals experiment threshold within the repository tolerance.
- Result model, symbol, horizon, threshold, window size, label rule, target type, dates, and completed epochs match.
- `status='completed'`
- `inference_scope='final'`
- `checkpoint_eval_id IS NULL`
- `parent_experiment_id IS NULL`
- Exactly one result exists; no recency fallback is permitted.

See [InferenceProfitabilityRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:421).

### Stage 2: authoritative observation selection

For the resolved result, the observation must match:

- Exact experiment ID.
- Exact final model ID.
- Exact inference result ID.
- `inference_scope='final'`.
- No checkpoint identity.
- Exact current `metric_definition_canonical`.
- Exact current `metric_definition_hash`.
- Exactly one matching observation.

The selector does not guess a source-content hash. More than one observation with exact provenance and the current metric definition is `ambiguous_profitability_observation`.

### Continuation treatment

- A FINAL observation remains a normal FINAL observation associated with the model’s experiment; its inference result must have `parent_experiment_id IS NULL`.
- A continuation child’s own final model can produce its own FINAL observation under the child experiment.
- Checkpoint observations instead require a checkpoint ID and use the parent experiment identity.
- Campaign Manager never substitutes checkpoint evidence for missing FINAL evidence.

### Multiplicity

The schema can retain multiple immutable observations for an experiment/model when they correspond to different inference results, intervals, source contents, metrics, or statistics. An identical canonical identity is idempotent.

Campaign Manager, however, requires:

1. One exact FINAL inference result.
2. One current-definition observation for that result.

Anything else fails closed.

### Current resolution results

Among 327 experiments having a `last_model_id`:

| Resolution | Experiments |
|---|---:|
| Exact FINAL result available | 251 |
| FINAL context mismatch | 63 |
| No exact FINAL result | 13 |
| Ambiguous exact FINAL result | 0 |

Among 26 distinct recommendation source experiment/model pairs:

| Resolution | Pairs | With observation |
|---|---:|---:|
| Exact FINAL result available | 24 | 0 |
| Context mismatch | 1 | 0 |
| No exact FINAL result | 1 | 0 |

These counts do not turn the 251 results into profitability evidence; every corresponding observation remains absent.

## 5. Profitability Metric Semantics

The canonical metric is defined in [InferenceProfitability.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitability.hpp:17).

| Primitive | Repository-defined meaning |
|---|---|
| `prediction_count` | One count for every evaluated inference window, including neutral predictions and directional predictions with unusable terminal prices. |
| `actionable_count` | Predicted Up or Down, with finite positive decision and terminal closes. Neutral and invalid-price directional predictions are excluded. |
| `winning_actionable_count` | Actionable directional log return strictly greater than zero. |
| `losing_actionable_count` | Actionable directional log return strictly less than zero. |
| Gross positive sum | Sum of positive directional terminal-horizon log returns. |
| Gross negative sum | Sum of negative directional terminal-horizon log returns; persisted as nonpositive. |
| Aggregate sum | Sum of all actionable directional terminal-horizon log returns. |
| Average per actionable | Aggregate sum divided by `actionable_count`; NULL when `actionable_count=0`. |

For an Up prediction:

```text
directional return = ln(terminal_close / decision_close)
```

For a Down prediction:

```text
directional return = -ln(terminal_close / decision_close)
```

Neutral is non-actionable. A zero directional return is actionable but is neither a win nor a loss. Therefore:

```text
winning_actionable_count + losing_actionable_count
    <= actionable_count
```

The difference represents zero-return actionable predictions.

The source-content hash incorporates, in order:

- Observation ordinal.
- Predicted class.
- Bit-exact float decision close.
- Bit-exact float terminal close.

The terminal close is the close at the configured terminal prediction horizon. It is not the earlier first-hit price used by the classification label; the call passes `labelInfo.closeT` and `labelInfo.targetClose` at [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:2882).

### Important interpretation limits

[Repository-defined semantic] This is explicitly **not portfolio P&L**:

- No transaction costs.
- No position sizing.
- No leverage.
- No capital or overlap constraint.
- No return clipping.

The aggregate is additive over the evaluated directional log-return samples, but overlapping windows can represent economically overlapping exposure. Additivity of the statistic does not make it a realizable portfolio return.

## 6. Coverage of Completed FINAL Inference

Exact coverage SQL:

```sql
BEGIN TRANSACTION READ ONLY;

SELECT count(*) AS completed_final,
       count(ipo.profitability_observation_id) AS with_any_observation,
       count(*)-count(ipo.profitability_observation_id)
           AS without_observation,
       count(DISTINCT ier.model_id) AS distinct_models,
       count(DISTINCT ier.symbol) AS distinct_symbols,
       count(DISTINCT ier.prediction_horizon) AS distinct_horizons,
       min(ier.completed_at) AS earliest,
       max(ier.completed_at) AS latest
FROM inference_eval_result ier
LEFT JOIN inference_profitability_observation ipo
  ON ipo.inference_eval_result_id=ier.id
 AND ipo.inference_scope='final'
WHERE ier.status='completed'
  AND ier.inference_scope='final';

ROLLBACK;
```

Result:

| Metric | Value |
|---|---:|
| Completed FINAL results | 318 |
| With profitability observation | 0 |
| Without profitability observation | 318 |
| Distinct models | 312 |
| Distinct symbols | 27 |
| Distinct horizons | 9 |
| Earliest completion | 2026-06-30 19:53:28-05 |
| Latest completion | 2026-08-23 03:25:20-05 |

### Missing-reason attribution

Migration 073 was recorded at:

```text
2026-08-22 16:34:08.42031-05
```

| Completion relative to migration 073 | FINAL results | Observations |
|---|---:|---:|
| Before migration | 315 | 0 |
| At/after migration | 3 | 0 |

For the 315 historical rows, migration 073 states that they are intentionally not backfilled because the exact per-window source content cannot be reconstructed from `inference_eval_result`.

The three later results were:

| Result | Experiment | Model | Symbol | Horizon | Completed |
|---:|---:|---:|---|---:|---|
| 840 | 594 | 1654 | audcadrmp | 4 | 2026-08-22 19:19:50-05 |
| 841 | 558 | 1658 | audchfrmp | 10 | 2026-08-23 00:50:29-05 |
| 842 | 560 | 1660 | audchfrmp | 8 | 2026-08-23 03:25:20-05 |

All three worker attempts used:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Build/Products/Release/LSTM_Release
```

That binary was modified `2026-08-20 07:03:17-05` and does not contain the metric canonical string or profitability persistence event.

The requested canonical binary:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
```

was modified `2026-08-22 20:22:47-05` and does contain:

- `inference_terminal_horizon_directional_log_return_v1`
- The profitability observation INSERT.
- The Phase 3A/3B zero-weight declarations.

The three inference logs contain the completed FINAL-result event but no profitability observation event. This provides a concrete explanation for the post-migration gap without reconstructing or rerunning inference.

## 7. Overall Profitability Distributions

There is no distribution.

The exact long-form PostgreSQL query used was:

```sql
BEGIN TRANSACTION READ ONLY;

WITH final AS (
  SELECT *
  FROM inference_profitability_observation
  WHERE inference_scope='final'
    AND checkpoint_eval_id IS NULL
), long AS (
  SELECT v.metric,v.value
  FROM final f
  CROSS JOIN LATERAL (VALUES
    ('actionable_count',
        f.actionable_count::double precision),
    ('actionable_rate',
        f.actionable_count::double precision /
        NULLIF(f.prediction_count,0)),
    ('winning_actionable_count',
        f.winning_actionable_count::double precision),
    ('losing_actionable_count',
        f.losing_actionable_count::double precision),
    ('aggregate_return',
        f.aggregate_terminal_horizon_log_return_sum),
    ('average_return',
        f.average_terminal_horizon_log_return_per_actionable_prediction)
  ) v(metric,value)
), metric_names(metric) AS (VALUES
  ('actionable_count'),
  ('actionable_rate'),
  ('winning_actionable_count'),
  ('losing_actionable_count'),
  ('aggregate_return'),
  ('average_return')
)
SELECT m.metric,
       count(l.metric) observation_rows,
       count(l.value) n,
       count(l.metric)-count(l.value) null_count,
       count(*) FILTER (WHERE l.value=0) zero_count,
       count(*) FILTER (WHERE l.value<0) negative_count,
       count(*) FILTER (WHERE l.value>0) positive_count,
       min(l.value), max(l.value), avg(l.value), stddev_samp(l.value),
       percentile_cont(ARRAY[.05,.10,.25,.50,.75,.90,.95])
         WITHIN GROUP (ORDER BY l.value) AS percentiles
FROM metric_names m
LEFT JOIN long l USING(metric)
GROUP BY m.metric
ORDER BY m.metric;

ROLLBACK;
```

All six metrics returned:

- Observation rows: 0
- N: 0
- In-table null count: 0
- Zero/negative/positive counts: 0
- Minimum, maximum, mean, standard deviation, and all percentiles: NULL/not estimable

The in-table null count of zero must not be confused with evidence availability: 318 completed FINAL results have no observation row at all.

## 8. Symbol Distribution

No authoritative FINAL observation has a symbol relationship because no observation exists.

The symbol stratification query returned zero rows:

```sql
SELECT ier.symbol, count(*) AS n
FROM inference_profitability_observation ipo
JOIN inference_eval_result ier
  ON ier.id=ipo.inference_eval_result_id
WHERE ipo.inference_scope='final'
GROUP BY ier.symbol
ORDER BY ier.symbol;
```

All symbol cohorts have an observed profitability sample size of zero and are too small for any distributional inference.

## 9. Horizon Distribution

No horizon has an authoritative profitability observation.

Although completed legacy FINAL results cover nine horizons, those results are not profitability observations and were not used as substitutes.

Every horizon profitability cohort has N=0.

## 10. Symbol/Horizon Distribution

The symbol/horizon query returned no rows:

```sql
SELECT ier.symbol, ier.prediction_horizon, count(*) AS n
FROM inference_profitability_observation ipo
JOIN inference_eval_result ier
  ON ier.id=ipo.inference_eval_result_id
WHERE ipo.inference_scope='final'
GROUP BY ier.symbol, ier.prediction_horizon
ORDER BY ier.symbol, ier.prediction_horizon;
```

The same applies to all requested additional strata:

- Experiment invocation/class.
- Continuation generation.
- Inference interval.
- `metric_definition_hash`.
- `source_content_hash`.

All have N=0.

`source_content_hash` should generally be treated as an exact content identity, not a scientific normalization cohort. Grouping by it is useful for detecting repeats or ambiguity, but ordinarily not for estimating a population distribution.

## 11. Zero-Actionable and Missing-Evidence Analysis

### Zero-actionable evidence

- Authoritative zero-actionable observations: 0
- Authoritative nonzero-actionable observations: 0
- Prevalence: not estimable

Repository semantics correctly distinguish:

- Zero-actionable observation: available evidence, aggregate normally zero, average NULL.
- Unavailable profitability: no observation and an explicit unavailable reason where Phase 3A applies.
- Zero aggregate with actionable observations: possible and distinct from zero-actionable.
- NULL average: defined only for zero-actionable, not as a generic missing-value representation.

### Missing evidence

- 318/318 completed FINAL results lack an observation.
- 24/24 exact FINAL recommendation source pairs lack an observation.
- Two Phase-3A recommendations explicitly record `no_profitability_observation`.
- The other 106 recommendations predate Phase 3A and do not claim either availability or unavailability.

## 12. Outliers and Concentration

Not estimable:

- No extreme positive or negative observations exist to inspect.
- No unusually high or low actionable counts exist.
- No zero-actionable observations exist.
- Total aggregate return is undefined over an empty population.
- Symbol/horizon contribution shares are undefined.
- No outlier experiment, model, observation, interval, symbol, or horizon IDs can be identified.

This does not establish that future evidence will lack outliers. It establishes only that the current authoritative population is empty.

## 13. Comparability and Normalization Analysis

### Raw aggregate return

[Repository-defined semantic] Raw aggregate return grows with the number of actionable windows and sums potentially overlapping directional returns.

[Recommendation] It should not be assumed comparable across:

- Different actionable counts.
- Different prediction counts.
- Different interval lengths.
- Different horizons.
- Different symbol volatility regimes.
- Different metric definitions.

It could become defensible only if empirical evidence shows comparable intervals and exposure opportunities, stable actionable rates, and similar symbol/horizon scale—or if the intended policy explicitly values total accumulated sample return rather than per-opportunity quality.

### Average return per actionable prediction

This removes the first-order scaling by actionable count and is structurally more comparable than aggregate return across observations of different activity levels.

It remains affected by:

- Symbol volatility.
- Horizon length.
- Directional mix.
- Market regime.
- Overlapping-window dependence.
- Small actionable counts.
- Selection conditioning on actionable predictions.

It is undefined for zero-actionable observations.

### Actionable-rate-conditioned measures

`actionable_count / prediction_count` measures directional coverage, not profitability. It could help distinguish:

- Positive returns from rare directional predictions.
- Similar average returns produced at very different coverage levels.
- Neutral-dominant models from consistently directional models.

It should not be folded silently into profitability.

### Within-symbol normalization

Potentially justified if symbols show materially different return scale but sufficient stable observations exist within each symbol. Invalidated by tiny symbol samples or rapid regime drift.

### Within-horizon normalization

Potentially justified if return dispersion systematically expands with prediction horizon. Invalidated if samples are too sparse or horizon effects interact strongly with symbol.

### Within-symbol/horizon normalization

Most specific but most data-hungry. It is likely scientifically cleaner when cohort sizes support it. Current N is zero in every cohort.

### Percentile/rank transforms

Could bound outlier influence and remove scale differences. They require sufficiently large, stable reference cohorts and can hide economically meaningful magnitude differences or create many ties in sparse/zero-inflated populations.

### Robust z-score or median/MAD normalization

Could be suitable for heavy-tailed evidence if each cohort has a stable median and nonzero MAD. It fails with tiny cohorts, zero MAD, severe multimodality, or regime shifts.

### Metric-definition identity

Values with different `metric_definition_hash` values must not be normalized together. A future policy should fail closed on mixed or unsupported metric semantics rather than attempt to make them comparable statistically.

## 14. Relationship to Existing Campaign Manager Metrics

The current scoring policy uses:

- Leader score.
- FINAL inference accuracy.
- Evidence strength.
- Predicted-neutral balance.
- Structural proximity.
- Parameter preference.
- Source rank.
- Horizon-change penalty.
- Relative-mutation penalty.

See [ExperimentRecommendationScoring.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:473).

The profitability correlation join returned:

```text
N=0
corr(aggregate, accuracy)=NULL
corr(average, accuracy)=NULL
corr(aggregate, leader score)=NULL
corr(average, leader score)=NULL
corr(aggregate, recommendation score)=NULL
corr(average, evaluation score)=NULL
```

Consequently, none of the requested empirical cases can be tested:

- High accuracy with negative profitability.
- Mediocre accuracy with positive profitability.
- High leader score with negative profitability.
- Strong recommendation score with poor profitability.
- Redundancy or independence between profitability and current metrics.

One missing-evidence case is observable. Recommendations 328 and 329 both have:

```text
source_experiment_id=454
source_model_id=1237
symbol=cadchfrmp
horizon=4
source_infer_accuracy=0.56221
source_leader_score=0.500677521025
source_evidence_count=18333
source_predicted_neutral_proportion=0.5544100801832761
source_final_inference_eval_result_id=405
profitability unavailable=no_profitability_observation
```

These demonstrate profitability unavailability despite complete-looking scientific source fields. They do not establish that those source metrics are “strong” in a profitability sense.

## 15. Phase 3A Recommendation/Evaluation Provenance Coverage

### Recommendations

| State | Count |
|---|---:|
| Total | 108 |
| `final_profitability_provenance_version=1` | 2 |
| Authoritative observation | 0 |
| Explicitly unavailable | 2 |
| Legacy/pre-3A | 106 |

Unavailable reasons:

| Reason | Count |
|---|---:|
| `no_profitability_observation` | 2 |

Exact SQL:

```sql
SELECT count(*) AS total,
       count(*) FILTER (
         WHERE final_profitability_provenance_version=1) AS phase3a_v1,
       count(*) FILTER (
         WHERE final_profitability_provenance_version IS NULL) AS legacy,
       count(*) FILTER (
         WHERE source_final_profitability_observation_id IS NOT NULL)
           AS available_observation,
       count(*) FILTER (
         WHERE final_profitability_provenance_version=1
           AND source_final_profitability_observation_id IS NULL)
           AS explicit_unavailable
FROM experiment_recommendation;

SELECT source_final_profitability_unavailable_reason,
       count(*) AS n
FROM experiment_recommendation
WHERE final_profitability_provenance_version=1
  AND source_final_profitability_observation_id IS NULL
GROUP BY source_final_profitability_unavailable_reason;
```

### Evaluation results

| State | Count |
|---|---:|
| Total | 102 |
| Phase-3A-aware | 0 |
| Available observation | 0 |
| Explicit unavailable provenance | 0 |
| Legacy | 102 |
| Separate profitability evidence hashes | 0 |

### Evaluation runs

| Run | Result rows | Phase 3A | Legacy |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 1 | 0 | 1 |
| 3 | 1 | 0 | 1 |
| 5 | 100 | 0 | 100 |

### Ranking populations

| Snapshot | Members | Phase 3A | Legacy |
|---:|---:|---:|---:|
| 1 | 1 | 0 | 1 |
| 3 | 1 | 0 | 1 |
| 4 | 100 | 0 | 100 |

No current evaluation or ranking population supports a profitability-aware comparison.

### Deployment bookkeeping observation

The live schema contains the Phase 3A and Phase 3B columns, functions, constraints, and triggers. However, querying `schema_migrations` for versions 073–077 returned ledger entries only for 073 and 074; 075, 076, and 077 were absent.

This is an observed migration-bookkeeping inconsistency. It does not corrupt profitability rows—there are none—but should be resolved through an authorized deployment process before relying on migration provenance operationally.

## 16. Phase 3B Semantic-Homogeneity Boundary Verification

All four evaluation runs passed the deployed Phase 3B semantic validator.

All 102 evaluation results passed validation against their parent runs.

All three ranking snapshots report:

```text
population_semantic_state=verified_homogeneous
distinct_scoring_semantic_count=1
distinct_evaluation_semantic_count=1
homogeneity_validation_result=verified_homogeneous
```

Their stored scoring semantics were successfully recomputed from the associated evaluation-run policy.

All 102 ranking members matched their evaluation result for:

- Final score.
- Recommendation semantic hash.
- Evaluation identity hash.

Decision-boundary checks:

| Check | Result |
|---|---:|
| Score rows whose policy mentions profitability | 0/16 |
| Evaluation runs whose policy mentions profitability | 0/4 |
| Evaluation decision evidence mentioning profitability | 0/102 |
| Score components named as profitability | 0/144 |
| Evaluation components named as profitability | 0/918 |
| Ranking scoring/evaluation semantics mentioning profitability | 0/3 snapshots |

Code declarations remain:

```text
profitability_weight=0
profitability_score_contribution=0
```

Relevant locations:

- Frozen profitability evidence canonical: [ExperimentRecommendation.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp:290)
- Evaluation reporting: [ExperimentRecommendationEvaluationService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationService.cpp:35)
- Ranking safety output: [ExperimentRecommendationRankingService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendationRankingService.cpp:30)
- Scoring semantic contract excludes profitability: [ExperimentRecommendationScoring.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:90)

## 17. Candidate Future Policy Shapes — Analysis Only

No production shape should be chosen yet.

### Eligibility/gating threshold

A future gate might be defensible if data demonstrates that profitability estimates below some actionable/effective sample size are unstable or sign-inconsistent.

Required supporting evidence would include:

- Estimate stability versus actionable count.
- Out-of-time sign persistence.
- Treatment of zero-actionable observations.
- Adjustment for overlapping predictions and effective sample size.
- Cohort-specific stability.

No numeric gate can currently be selected.

### Bounded score component

A bounded component could limit the impact of heavy tails and prevent profitability from dominating established scientific evidence.

Before adopting one, evidence should determine:

- Aggregate versus average target.
- Cohort normalization.
- Outlier behavior.
- Missing-evidence treatment.
- Whether percentile, rank, clipping, or robust scaling preserves useful distinctions.
- Correlation with existing components.

No weight or transform can currently be selected.

### Both

A gate plus bounded component is conceptually plausible: the gate establishes minimum reliability, while the component represents profitability direction/magnitude. It cannot be endorsed empirically with N=0.

### Current recommendation

**Neither yet.** Accumulate authoritative evidence first.

## 18. Risks and Failure Modes

- Missing observations could be incorrectly treated as zero profitability.
- Scheduler workers may continue using the older noncanonical binary and silently fail to accumulate evidence.
- The live migration ledger does not record Phase 3A/3B migrations even though their objects are installed.
- Raw aggregate return can reward activity, interval length, or overlapping exposure rather than better per-opportunity returns.
- Average return can be extremely unstable at small actionable counts.
- Window overlap means actionable count overstates independent effective sample size.
- Zero-actionable evidence requires explicit semantics; its average is undefined.
- Symbol and horizon effects may dominate a global scale.
- Sparse symbol/horizon cohorts can make normalization noisy or arbitrary.
- Different inference intervals and market regimes may not be exchangeable.
- Different metric-definition identities are not comparable.
- Multiple current-definition observations for one result must remain ambiguous.
- Source-content hashes are identities, not normalization categories.
- A future score trained on selectively available observations could introduce missingness or survivorship bias.
- Database checks enforce shape and provenance but do not independently recompute every arithmetic/hash equality; future inspections should verify populated rows against repository semantics.
- Legacy recommendations and evaluations must not be rewritten or silently treated as Phase-3A-aware.

## 19. Answers to Decision Questions A–J

**A. Is there enough evidence to design a defensible profitability-aware score?**

No. There are zero authoritative FINAL observations.

**B. Is raw aggregate profitability comparable across the current population?**

No current population exists for empirical comparison. Structurally, raw aggregate return is confounded by actionable count, interval length, horizon, symbol scale, and overlapping windows.

**C. Is average return per actionable prediction more comparable?**

Structurally, yes with respect to differing actionable counts. Empirically, this is unverified, and it remains sensitive to small samples, symbol, horizon, regime, and overlap.

**D. Should `actionable_count` act as evidence strength rather than profitability?**

Yes in principle. It is a reliability/coverage dimension, not profitability. It should not be treated as independent effective sample size without accounting for overlapping predictions.

**E. Are symbol/horizon cohorts likely necessary?**

Likely, because terminal-return scale plausibly varies with symbol and horizon. Current data cannot establish how strong those effects are.

**F. Are there enough observations per cohort?**

No. Every cohort has N=0.

**G. Is profitability distinct enough from inference accuracy and leader score?**

Unknown. Conceptually the metric measures different behavior, but empirical distinctness and incremental information cannot be established.

**H. Should implementation begin with a gate, bounded component, both, or neither?**

Neither yet.

**I. What additional evidence is required?**

- Naturally accumulated authoritative FINAL observations from the canonical binary.
- Verified coverage of new completed FINAL results.
- More than one stable inference interval or out-of-time period.
- Adequate samples across symbols and horizons.
- A meaningful number of actionable and zero-actionable observations.
- Stability analyses versus actionable count and effective sample size.
- Cohort distributions and outlier/concentration analysis.
- Correlations and discordant-case analysis against accuracy, leader score, and current scores.
- Verification that metric, source-content, observation identities, arithmetic, and provenance remain intact.
- Phase-3A-aware evaluation/ranking populations containing actual observations.

**J. What fail-closed rules must be preserved?**

1. Require `inference_scope='final'`.
2. Require `checkpoint_eval_id IS NULL`.
3. Never substitute checkpoint, legacy, backtest, reconstructed, or recommendation-snapshot evidence.
4. Resolve exactly one completed FINAL result matching the final model and full persisted inference context.
5. Require `parent_experiment_id IS NULL` for FINAL inference results.
6. Require exact experiment/model/result/interval provenance.
7. Require the supported metric canonical and hash.
8. Require exactly one matching current-definition observation.
9. Treat no observation, metric mismatch, provenance mismatch, and ambiguity as unavailable—not zero.
10. Preserve zero-actionable as available evidence with NULL average.
11. Reject nonfinite values and inconsistent count shapes.
12. Never compare or normalize across metric-definition identities.
13. Freeze profitability provenance in recommendation/evaluation snapshots.
14. Preserve legacy rows as legacy; do not infer Phase 3A semantics.
15. Keep scoring/evaluation/ranking populations semantically homogeneous.
16. Include any future profitability policy in scoring/evaluation/ranking semantic identities.
17. Do not allow profitability into ranking or tie-breaks unless an explicit, versioned policy authorizes it.
18. Keep missing-evidence handling explicit and deterministic.
19. Preserve immutability and idempotent retry behavior.
20. Keep profitability non-decision-bearing until a later reviewed implementation explicitly changes the zero-weight boundary.

## 20. Phase 3C Readiness Classification

`NOT_READY_INSUFFICIENT_PROFITABILITY_EVIDENCE`

Reason:

- The authoritative profitability table is empty.
- Completed FINAL coverage is 0/318.
- Every distribution and cohort has N=0.
- Phase 3A evaluation/ranking coverage is zero.
- No empirical normalization, reliability gate, distinctness claim, threshold, or weight can be supported.

The classification is not `BLOCKED_BY_PROVENANCE_OR_SEMANTIC_DEFECT` because no populated observation has been shown to be corrupt or semantically invalid. The binary-path and migration-ledger findings are operational risks that must be corrected before evidence can accumulate reliably, but the immediate readiness failure is total evidence absence.

## 21. Recommended Next Step

In a separately authorized operational phase:

1. Resolve why scheduler workers use `DerivedData/Build/...` instead of the specified canonical binary.
2. Reconcile the missing 075–077 migration-ledger entries without rewriting production evidence.
3. Allow new FINAL inference to accumulate profitability observations naturally; do not rerun or backfill historical inference solely for Phase 3C.
4. Audit each newly completed FINAL result for atomic observation coverage and identity integrity.
5. Accumulate enough observations across symbols, horizons, intervals, and actionable-count ranges.
6. Repeat this distributional inspection before choosing any profitability gate, transform, threshold, or weight.

Files changed: **none**. Builds/tests run: **none**, consistent with the inspection-only constraint.