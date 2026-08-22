---
title: "LSTM Campaign Manager Phase 1 Candidate Selection Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase1_CandidateSelection_Inspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 1 Candidate Selection Inspection

## 1. Executive finding

The repository’s executable architecture does not give the production Campaign Manager responsibility for scientific candidate discovery, scoring, ranking, or continuation decisions. The actual Campaign Manager RUN_ONCE cycle is a bounded operational dispatcher: it selects already accepted Campaign Operations requests in `ready` state and launches their already approved/materialized recommendation campaigns. Its query contains no experiment metrics, checkpoint fields, profitability fields, symbol, horizon, score, or rank.

The upstream recommendation workflow is a separate sequence. It:

1. Discovers completed experiments with completed final-model analysis.
2. Retains the top sources per configured scope using final `leader_score`, final `infer_accuracy`, and evidence count.
3. Generates one-parameter neighboring configurations.
4. Evaluates and scores those recommendation rows.
5. Creates an advisory ranking snapshot.
6. Applies campaign-planning gates and quotas to produce include/exclude decisions.
7. Requires explicit approval and materialization before Campaign Operations can dispatch anything.

Continuation of an existing model is yet another subsystem: scheduler continuation automation. It can use final-model evidence or checkpoint evidence, depending on `continuation_policy_source_mode`, and produces decisions such as `eligible`, `rejected_threshold`, `rejected_rank`, and `rejected_trend`.

Explicit answers:

- **Checkpoint inference is not used by recommendation candidate discovery, recommendation scoring, recommendation ranking, campaign planning, or the Campaign Manager RUN_ONCE dispatcher.** Checkpoint inference is consumed directly by checkpoint-policy logic and by scheduler continuation automation when the source mode is `best_checkpoint` or `latest_checkpoint`.
- **Profitability is not used in recommendation discovery, scoring, ranking, campaign selection, Campaign Manager dispatch, checkpoint policy, or scheduler continuation.** Campaign planning contains a placeholder `minimumProfitability` gate, but its repository loader never supplies a profitability metric. Enabling that gate therefore excludes candidates as `profitability_metric_unavailable`; it does not evaluate profitability.

## 2. Production call path

### Actual production Campaign Manager cycle

```text
CampaignOperationsH4Supervisor.py
  → LSTM_Release --campaign-operations-manager-run-once
  → RunCampaignOperationsCommand
  → RunCampaignOperationsManagerOnceCommand
  → RunManagerOnceInternal
  → SelectDispatchCandidatesForManager
  → DispatchOneRequestForProductionManager
  → LoadRecommendationCampaignLaunchMaterialization
  → LaunchRecommendationCampaignInTransaction
  → persist dispatch binding / control owner / reservation outcome
```

Evidence:

- The supervisor launches RUN_ONCE at [CampaignOperationsH4Supervisor.py:872](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH4Supervisor.py:872>).
- CLI routing is in [ExperimentScheduler.cpp:23456](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:23456>).
- Candidate selection and sequential dispatch are in [CampaignOperationsManagerService.cpp:127](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:127>).
- Materialization loading and transactional launch are in [CampaignOperationsDispatchService.cpp:557](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:557>) and [CampaignOperationsDispatchService.cpp:634](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:634>).

There is no call from this chain into recommendation discovery, recommendation scoring, checkpoint policy, or scheduler continuation evaluation.

### Upstream recommendation/campaign workflow

This is invoked through separate CLI operations, not by RUN_ONCE:

```text
RunExperimentRecommendationCommand
  → Generate recommendations
      LoadRecommendationSources
      → EvaluateRecommendationSource
      → SelectRecommendationSources
      → GenerateRecommendationCandidates
      → PersistRecommendation
  → Evaluate proposed recommendations
      LoadRecommendationsForEvaluation
      → EvaluateExperimentRecommendation
      → ScoreExperimentRecommendation
      → persist evaluation results/components
  → Create ranking snapshot
      LoadEvaluationsForRanking
      → RankRecommendationEvaluationEvidence
      → persist ranking members
  → Plan campaign
      LoadRecommendationCampaignPlanInput
      → PlanRecommendationCampaign
  → explicit approval
  → materialization into conversion proposals
  → Campaign Operations admission, budget, and request acceptance
  → Campaign Manager RUN_ONCE dispatch
```

The separate CLI routes are visible at:

- Generation: [ExperimentScheduler.cpp:23705](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:23705>)
- Evaluation: [ExperimentScheduler.cpp:24055](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:24055>)
- Ranking: [ExperimentScheduler.cpp:24098](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:24098>)
- Campaign planning/approval/materialization: [ExperimentScheduler.cpp:23909](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:23909>)

The standalone scoring command also exists, but campaign ranking does not consume its persisted score rows. Evaluation recomputes the same scoring formula and persists `experiment_recommendation_evaluation_result.final_score`; ranking snapshots consume those evaluation results.

## 3. Candidate discovery

### Source experiment SQL

`LoadRecommendationSources` executes the authoritative source query in [ExperimentRecommendationRepository.cpp:417](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:417>):

```sql
FROM experiment e
LEFT JOIN experiment_analysis_result a
  ON a.experiment_id = e.experiment_id
 AND a.model_id = e.last_model_id
 AND COALESCE(a.analysis_scope, 'final') = 'final'
WHERE e.status = 'completed'
  AND e.phase = 'done'
  AND (symbol filter is absent OR lower(btrim(e.symbol)) = filter)
  AND (horizon filter is absent OR e.prediction_horizon = filter)
  AND (experiment filter is absent OR e.experiment_id = filter)
ORDER BY e.experiment_id;
```

This establishes:

- Only `completed` / `done` experiments enter discovery.
- The selected model is exactly `experiment.last_model_id`.
- The selected analysis must be for that model and have final scope.
- No checkpoint model is selected.
- Optional CLI filters can restrict symbol, prediction horizon, or source experiment.
- There is no age, profitability, checkpoint-policy, checkpoint-inference, or dominance predicate.

### Post-query exclusions

The loader excludes a source when:

- `resume_expand_input_width=true`;
- `last_model_id` is null;
- no matching final analysis exists;
- final analysis status is not `completed`;
- dates or the effective semantic/invocation configuration cannot be mapped and validated.

See [ExperimentRecommendationRepository.cpp:464](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:464>).

### Eligibility metrics and thresholds

`EvaluateRecommendationSource` then requires [ExperimentRecommendationCandidateGenerator.cpp:220](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:220>):

- policy enabled and valid;
- valid experiment ID;
- valid semantic and invocation identities;
- nonmissing, finite `leader_score`;
- nonmissing, finite `infer_accuracy`;
- evidence count at least the policy minimum;
- `leader_score >= minimumLeaderScore`;
- `infer_accuracy >= minimumInferenceAccuracy`;
- when a maximum neutral proportion is configured, neutral proportion must exist;
- neutral proportion must be finite, in `[0,1]`, and not above the maximum.

Default discovery policy values are [ExperimentRecommendation.hpp:43](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:43>):

- minimum leader score: `0`
- minimum inference accuracy: `0`
- maximum predicted-neutral proportion: `0.80`
- minimum evidence: `1`
- top sources per scope: `3`
- scope: symbol plus horizon
- maximum recommendations per source: `4`
- maximum recommendations per scan: `20`

These are defaults; CLI policy text can override them.

### Which eligible sources are selected

Sources are grouped by the configured scope—default `(symbol, horizon)`—then sorted by:

```text
leader_score descending
infer_accuracy descending
evidence_count descending
experiment_id ascending
```

Only the first `topSourcesPerScope` sources in each group survive. See [ExperimentRecommendationService.cpp:249](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:249>).

### Candidate configurations

For each selected source, the generator creates single-parameter mutations [ExperimentRecommendationCandidateGenerator.cpp:342](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:342>).

Defaults are:

- `core_lr_mult`: source ± `0.25`
- `head_lr_mult`: source ± `0.5`
- `c_next_threshold`: source ± `0.0001`
- horizon changes disabled

A candidate is rejected if the source value is unavailable, the proposed value is nonfinite/nonpositive/unchanged, the change affects more than one semantic field, it duplicates another canonical candidate, or it falls outside the per-source/scan limit.

Existing experiments are checked using exact reconstructed semantic identities. Nonterminal experiment matches exclude the recommendation. Failed/cancelled matches also exclude by default because `terminalExperimentsAreDuplicates=true`; see [ExperimentRecommendationRepository.cpp:513](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:513>).

Persisted recommendation rows snapshot:

- source experiment/model/analysis IDs;
- source leader/inference/neutral/evidence metrics;
- changed parameter and deltas;
- semantic/invocation identities;
- policy identity;
- source rank and deterministic structural rank.

The insert is at [ExperimentRecommendationRepository.cpp:724](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:724>).

### Selection for evaluation

Evaluation does not select candidates based on a prior score. `LoadRecommendationsForEvaluation` loads `experiment_recommendation` rows where `status='proposed'`, restricted only by optional scan/recommendation filters and the requested limit, ordered by recommendation ID [ExperimentRecommendationEvaluationRepository.cpp:286](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:286>).

Evaluation then blocks candidates whose source provenance has become stale or that now conflict with exact pending, active, or completed experiments.

## 4. Candidate scoring and ranking

### Authoritative campaign score

`ScoreExperimentRecommendation` is defined at [ExperimentRecommendationScoring.cpp:321](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:321>). With default weights, the score is:

```text
positive =
    0.25 * clamp(leader_score, 0, 1)
  + 0.25 * infer_accuracy
  + 0.15 * evidence_strength
  + 0.10 * neutral_balance
  + 0.15 * structural_proximity
  + 0.05 * parameter_preference
  + 0.05 * (1 / source_rank)

penalty =
    0.05 * horizon_change_penalty
  + 0.10 * mutation_penalty

final_score = clamp(positive - penalty, 0, 1)
```

More generally, both positive and penalty sums are divided by the total positive weight. Under the defaults, that denominator is exactly `1.0`.

| Input | Persisted/source field | Transformation | Direction | Missing/invalid behavior |
|---|---|---|---|---|
| Leader quality | `experiment_recommendation.source_leader_score`, originally final `experiment_analysis_result.leader_score` | Clamp to `[0,1]`; weight `0.25` | Higher is better | Missing/nonfinite source is ineligible or evaluation-invalid |
| Inference accuracy | `source_infer_accuracy`, originally final `experiment_analysis_result.infer_accuracy` | Direct unit value; weight `0.25` | Higher is better | Must be present, finite, and at scoring time in `[0,1]` |
| Evidence strength | `source_evidence_count` | `clamp((count-1)/(5000-1),0,1)`; weight `0.15` | Higher until saturation | Below `1` invalid |
| Neutral balance | `source_predicted_neutral_proportion` | `1` at or below `1/3`; then linear decline to `0` at `0.80`; weight `0.10` | Lower neutral dominance is better | Scoring default assigns `0.5` if absent, but default discovery rejects absence and campaign planning also rejects it |
| Structural proximity | `relative_delta`, otherwise `absolute_delta` | `clamp(1-distance/max,0,1)`; max `0.50` relative or `1.0` absolute; weight `0.15` | Smaller mutation is better | Invalid/negative/nonfinite delta invalidates scoring |
| Parameter preference | `changed_parameter` plus scoring policy | Default preference is `1` for every supported family; weight `0.05` | Policy-dependent | Unsupported family invalidates scoring |
| Source rank | `source_rank` | Reciprocal `1/rank`; weight `0.05` | Smaller source rank is better | Must be positive |
| Horizon penalty | `horizon_delta` and source horizon | `clamp(abs(delta)/source_horizon,0,1)`; penalty weight `0.05` | Smaller is better | Required for a horizon mutation; horizons are disabled by default |
| Mutation penalty | `relative_delta`, otherwise `absolute_delta` | Distance divided by the same configured maximum and clamped; penalty weight `0.10` | Smaller is better | Invalid structural metadata invalidates scoring |

Default weights and bounds are declared in [ExperimentRecommendationScoring.hpp:13](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.hpp:13>).

The persisted `leader_score` is itself computed outside Campaign Manager as:

```text
infer_accuracy
* (0.75 + 0.25 * accept_accuracy)
* prediction_imbalance_penalty
```

See [ExperimentScheduler.cpp:10246](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:10246>). Thus final inference accuracy, acceptance accuracy, and prediction-class imbalance can influence campaign scoring indirectly through final `leader_score`. No profitability term appears in that formula.

### Scoring validation

Scoring rejects a recommendation when it is not `proposed`, identities/ranks are incomplete, metrics are nonfinite, inference accuracy is outside `[0,1]`, evidence is below minimum, structural metadata is invalid, or the parameter family is unsupported.

There is no default minimum final-score gate.

### Ranking procedures

There are three related but distinct orderings:

1. **Standalone score-run ranking** — final score descending, raw positive descending, raw penalty ascending, source leader/inference/evidence descending, structural distance ascending, then canonical identities and ID. Equal final scores share a score rank and tie group. See [ExperimentRecommendationScoring.cpp:498](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:498>).

2. **Evaluation ordering** — final score descending, then disposition, evaluation identity, recommendation ID [ExperimentRecommendationEvaluation.cpp:425](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:425>).

3. **Campaign ranking snapshot** — the authoritative campaign input:
   - bucket order: `advisory_ready`, `blocked`, `non_actionable`;
   - advisory-ready candidates: final score descending;
   - blocked/nonactionable candidates: disposition priority;
   - deterministic ties: recommendation semantic hash, evaluation identity hash, evaluation-result ID.

   See [ExperimentRecommendationRanking.cpp:386](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:386>).

There is no Pareto or dominance elimination in this path. The scheduler intelligence report has a separate dominance query comparing leader/inference/acceptance metrics at [ExperimentScheduler.cpp:22260](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:22260>), but that query is not called by recommendation generation, evaluation, ranking, or campaign planning.

## 5. Continuation decision

### Recommendation campaign include/exclude decision

Evaluation marks a candidate `advisory_ready` only when:

- persisted identities and numeric evidence are valid;
- source model/analysis evidence exists;
- scan and source experiment remain completed;
- current final model and analysis still exactly match the recommendation snapshots;
- no exact pending/running/paused/completed experiment conflicts;
- scoring succeeds.

See [ExperimentRecommendationEvaluation.cpp:325](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:325>).

Campaign planning then processes ranking members in global ranking order [ExperimentRecommendationCampaignPlanning.cpp:409](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:409>). A candidate is included only if it passes all hard gates:

- campaign planning enabled;
- within the maximum candidate count;
- valid ranking snapshot position and contract;
- ranking bucket is `advisory_ready` and score is present;
- identity/provenance is consistent;
- leader and inference metrics meet campaign thresholds;
- predicted-neutral metric exists and is not above the maximum;
- profitability gate passes, if configured;
- no disqualifying prior conversion workflow;
- no duplicate selected invocation;
- source/symbol/horizon quotas are available;
- campaign selected-count limit is available.

The exact decision loop is [ExperimentRecommendationCampaignPlanning.cpp:959](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:959>). Default planning limits and gates are [ExperimentRecommendationCampaignPlanning.hpp:13](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.hpp:13>):

- maximum selected: `10`
- maximum considered: `100`
- minimum leader: `0`
- minimum inference: `0`
- maximum neutral: `0.80`
- maximum per source experiment: `1`
- no symbol/horizon quota by default
- no profitability threshold by default

There is no minimum campaign `final_score`; score controls order, while the listed predicates are hard gates.

Existing workflows exclude selection by default when pending review, approved but unexecuted, executed/paused, activated/scheduler-owned, completed, failed, cancelled, or inconsistent, subject to the configured reconsideration flags. The state mapping is in [ExperimentRecommendationCampaignPlanning.cpp:423](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:423>).

An include decision is still advisory. Approval is explicitly persisted without launching an experiment [ExperimentRecommendationCampaignApprovalService.cpp:92](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignApprovalService.cpp:92>). Materialization reconstructs the exact approved plan and creates conversion proposals for included members [ExperimentRecommendationCampaignMaterializationService.cpp:188](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignMaterializationService.cpp:188>).

### Scheduler “continue / do-not-continue” decision

The literal model-continuation decision belongs to scheduler continuation automation.

Initial discovery SQL is [ExperimentScheduler.cpp:14121](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14121>):

```sql
SELECT experiment_id
FROM experiment
WHERE continuation_policy_enabled = true
  AND status = 'completed'
  AND phase = 'done'
ORDER BY experiment_id;
```

For each source, `EvaluateContinuationPolicy` requires:

- continuation policy enabled and valid;
- source completed/done;
- `continuation_candidate_excluded=false`, unless policy allows excluded sources;
- valid evidence for the configured source mode;
- target epochs greater than the source epoch;
- resumable source model with matching ownership/configuration;
- no equivalent continuation already queued;
- sufficient distinct evidence;
- configured minimum leader and inference thresholds;
- configured top-N rank;
- configured trend rule.

Thresholds are conjunctive here: any failed configured threshold gives `rejected_threshold`; an out-of-range rank gives `rejected_rank`; a failed trend gives `rejected_trend`; only all configured gates passing yields `eligible`. See [ExperimentScheduler.cpp:12970](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12970>).

Eligible candidates are sorted by:

```text
rank present before absent
rank ascending
leader_score descending
infer_accuracy descending
completed epoch descending
source experiment ID ascending
model ID ascending
```

The scheduler queues only the configured maximum per scan [ExperimentScheduler.cpp:14083](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14083>) and [ExperimentScheduler.cpp:14337](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14337>).

## 6. Checkpoint-inference relationship

Final and checkpoint inference are intentionally separated.

Final inference lookup requires:

```sql
inference_scope = 'final'
AND checkpoint_eval_id IS NULL
```

See [ExperimentScheduler.cpp:9997](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9997>).

Checkpoint inference lookup requires:

```sql
checkpoint_eval_id = ...
AND model_id = checkpoint_model_id
AND inference_scope = 'checkpoint'
```

See [ExperimentScheduler.cpp:10156](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:10156>).

Recommendation source discovery joins only:

```sql
a.model_id = e.last_model_id
AND COALESCE(a.analysis_scope, 'final') = 'final'
```

Therefore checkpoint analyses and checkpoint model IDs cannot enter recommendation discovery or its persisted source snapshots.

Classification:

- **Campaign Manager RUN_ONCE:** no checkpoint usage.
- **Recommendation eligibility:** no checkpoint usage.
- **Recommendation score/rank:** no checkpoint usage.
- **Campaign planning:** no checkpoint usage.
- **Indirect persisted-metric propagation:** none found; checkpoint leader/inference fields are not copied into the final recommendation metrics.
- **Possible lifecycle influence:** checkpoint policy may stop training earlier, which can change which eventual model becomes `last_model_id`. Campaign recommendations still consume only that model’s separately persisted final analysis, not the checkpoint result itself.

### Checkpoint policy

Checkpoint policy loads completed checkpoint-scope analysis [ExperimentScheduler.cpp:11571](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11571>) and ranks checkpoint evaluations by:

```text
checkpoint leader_score descending
checkpoint infer_accuracy descending
checkpoint epoch descending
checkpoint_eval_id ascending
```

Its rule combination differs from scheduler continuation: after grace, any passed configured rule causes `continue`; it requests a stop only when configured rules exist and all fail [ExperimentScheduler.cpp:11721](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11721>).

### Scheduler continuation

Continuation evidence explicitly unions checkpoint and final analyses [ExperimentScheduler.cpp:12077](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12077>).

Source-mode behavior is [ExperimentScheduler.cpp:12257](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12257>):

- `best_checkpoint`: best checkpoint by leader, inference, epoch, model ID, analysis ID;
- `latest_checkpoint`: latest epoch, then checkpoint/analysis ID;
- `final_model`: final analysis matching `last_model_id`.

Thus checkpoint inference directly affects scheduler continuation only for checkpoint source modes. It remains outside Campaign Manager and recommendation-campaign scoring.

## 7. Profitability relationship

No executable profitability input enters the decision path.

The campaign-plan input query selects ranking score, source leader score, source inference accuracy, neutral proportion, identities, and workflow state, but selects no profitability column and joins no trading/profitability result [ExperimentRecommendationCampaignPlanningRepository.cpp:143](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:143>). Consequently `profitabilityMetric` and its identity remain unset.

If `minimumProfitability` is configured:

```text
missing profitability metric → exclude: profitability_metric_unavailable
metric below threshold       → exclude: below_minimum_profitability
```

See [ExperimentRecommendationCampaignPlanning.cpp:1008](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:1008>). The code explicitly states that no authoritative profitability metric exists in the current schema [ExperimentRecommendationCampaignPlanning.cpp:861](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:861>).

A source/schema search found no executable fields for P&L, expectancy, trade count, win rate, directional payoff, or simulated-trading profitability. The only profitability-related production code is this unpopulated campaign-planning placeholder and its CLI option.

Post-campaign outcome assessment is also classification-metric based. It compares only `inference_accuracy` and `leader_score`, not profit [ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp:588](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp:588>).

## 8. Data-flow map

```text
FINAL MODEL PATH
experiment.last_model_id
  → final inference_eval_result
       scope='final', checkpoint_eval_id IS NULL
  → experiment_analysis_result
       analysis_scope='final'
       leader_score
       infer_accuracy
       pred_down_count / pred_neutral_count / pred_up_count
  → LoadRecommendationSources
       completed/done + final model/analysis filters
  → source eligibility and top-N source selection
  → one-parameter candidate generation
  → experiment_recommendation
       immutable source metric snapshots + mutation metadata
  → evaluation
       stale/duplicate gates + scoring formula
  → evaluation result final_score
  → ranking snapshot
       advisory-ready first, score descending
  → campaign plan
       metric gates + workflow exclusions + quotas
  → explicit approval
  → materialization / conversion proposals
  → Campaign Operations admission, budget, accepted ready request
  → Campaign Manager RUN_ONCE
  → transactional campaign launch
  → created/reused experiment enters scheduler lifecycle


CHECKPOINT PATH
checkpoint inference_eval_result
  → checkpoint experiment_analysis_result
  ├─→ checkpoint policy continue/stop decision
  └─→ scheduler continuation evidence when source_mode is
       best_checkpoint or latest_checkpoint
         → continuation eligible/rejected
         → ranked eligible queue
         → resume experiment

  ✗ does not enter recommendation discovery/scoring/ranking/campaign plan
  ✗ does not enter Campaign Manager RUN_ONCE


PROFITABILITY PATH
no authoritative persisted input loaded
  → profitabilityMetric remains NULL
  → ignored when minimumProfitability is unset
  → all candidates fail profitability_metric_unavailable when it is set

  ✗ no scoring contribution
  ✗ no ranking contribution
  ✗ no continuation contribution
```

## 9. Relevant files and functions

The small set most relevant to a later Phase 2 change is:

- [ExperimentRecommendationRepository.cpp:417](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:417>) — `LoadRecommendationSources`, duplicate checks, recommendation persistence.
- [ExperimentRecommendationCandidateGenerator.cpp:220](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCandidateGenerator.cpp:220>) — source eligibility and candidate generation.
- [ExperimentRecommendationService.cpp:249](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:249>) — source selection and scan orchestration.
- [ExperimentRecommendationScoring.hpp:13](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.hpp:13>) and [ExperimentRecommendationScoring.cpp:321](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.cpp:321>) — score policy and formula.
- [ExperimentRecommendationEvaluationRepository.cpp:286](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:286>) and [ExperimentRecommendationEvaluation.cpp:325](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:325>) — evaluation evidence, gates, and scoring invocation.
- [ExperimentRecommendationRanking.cpp:386](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRanking.cpp:386>) — authoritative ranking buckets and ordering.
- [ExperimentRecommendationCampaignPlanningRepository.cpp:97](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:97>) and [ExperimentRecommendationCampaignPlanning.cpp:959](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.cpp:959>) — campaign-plan inputs and include/exclude decision.
- [ExperimentScheduler.cpp:12077](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12077>) — checkpoint/final continuation evidence and scheduler continuation policy.
- [CampaignOperationsManagerService.cpp:127](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:127>) and [CampaignOperationsDispatchRepository.cpp:219](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp:219>) — actual Campaign Manager RUN_ONCE boundary.

## 10. Verification conclusion

Executable code establishes that the production Campaign Manager is an operational dispatcher of already accepted requests, not a scientific continuation selector. The upstream recommendation campaign uses final-model `leader_score`, final `infer_accuracy`, predicted-neutral proportion, evidence count, mutation distance, parameter preference, and source rank. It applies exact duplicate/provenance/workflow/metric/quota gates but no dominance rule and no minimum final-score gate.

Checkpoint inference is excluded from that recommendation path by final-scope/model joins. It is used independently by checkpoint policy and, when configured, scheduler continuation automation. Profitability is not populated or consumed; the current campaign profitability gate is an unavailable-data exclusion placeholder.

Verification state:

- Files changed: none.
- Behavioral changes: none.
- Build/tests: not run; this was static inspection only, and no executable or database workflow was launched.
- Current checked-out branch: `lstm-feature-development`.
- Remaining limitation: persisted production policy overrides were not inspected because the task prohibited database/state interaction; stated numeric values are executable defaults.
- `git status --short`:

```text
?? lstm_watch.sql
```

- `lstm_watch.sql` was already untracked and was not modified.
- `git diff --stat`: empty.