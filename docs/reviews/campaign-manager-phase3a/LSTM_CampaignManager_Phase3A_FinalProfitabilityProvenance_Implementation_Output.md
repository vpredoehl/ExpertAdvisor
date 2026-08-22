---
title: "LSTM Campaign Manager Phase 3A Final Profitability Provenance Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3A_FinalProfitabilityProvenance_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3A Final Profitability Provenance Implementation

Implemented Campaign Manager Phase 3A as evidence/provenance-only plumbing. FINAL profitability is frozen and observable, while scoring weight/contribution remain exactly zero and all decision behavior remains unchanged. Worktree is uncommitted.

## 1–2. Files and migration

Production:

- [ExperimentRecommendation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:16>)
- [ExperimentRecommendation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp:235>)
- [ExperimentRecommendationRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.hpp:167>)
- [ExperimentRecommendationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp:75>)
- [ExperimentRecommendationEvaluation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.hpp:77>)
- [ExperimentRecommendationEvaluation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluation.cpp:128>)
- [ExperimentRecommendationEvaluationRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.hpp:91>)
- [ExperimentRecommendationEvaluationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationRepository.cpp:40>)
- [ExperimentRecommendationEvaluationService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationEvaluationService.cpp:27>)
- [ExperimentRecommendationCampaignPlanning.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanning.hpp:130>)
- [ExperimentRecommendationCampaignPlanningRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp:32>)
- [ExperimentRecommendationCampaignPlanningService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignPlanningService.cpp:97>)
- [ExperimentRecommendationService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:74>)
- [ExperimentRecommendationScoring.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationScoring.hpp:81>)

Migration and documentation:

- [076_campaign_manager_final_profitability_provenance.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/076_campaign_manager_final_profitability_provenance.sql:1>) — next unused migration after 075; not applied to production.
- [CampaignManagerPhase3AFinalProfitabilityProvenance.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignManagerPhase3AFinalProfitabilityProvenance.rst:1>)

Tests:

- [ExperimentRecommendationPhase3AProfitabilityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3AProfitabilityTests.cpp:154>)
- [ExperimentRecommendationPhase3ASourceRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.cpp:1>)
- [ExperimentRecommendationPhase3ARepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3ARepositoryTests.cpp:1>)
- [ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql:1>)
- Corresponding four `.sh` runners.
- Updated isolated fixtures in the campaign planning, campaign approval, and evaluation repository tests.

## 3. Existing path reconstructed

The current path is:

1. `LoadRecommendationSources` discovers completed/done source experiments and their `last_model_id`.
2. The completed FINAL `experiment_analysis_result` supplies existing leader, accuracy, neutral-proportion, and evidence-count metrics.
3. `EvaluateRecommendationSource` performs candidate eligibility.
4. Candidate generation freezes source evidence into `experiment_recommendation`.
5. `EvaluateExperimentRecommendation` computes the existing nine-component score.
6. Evaluation run/result/component repositories persist evidence and evaluation identities.
7. `RankRecommendationEvaluations` and persisted ranking snapshots/members establish ordering and tie-breaks.
8. Campaign planning consumes ranking ordinal/member provenance, then review, approval, materialization, activation, and execution use the persisted campaign workflow.

The existing campaign `profitabilityMetric`/`minimumProfitability` decision channel remains unpopulated and inactive.

## 4–6. Authoritative source and loading

The loader uses the existing authoritative pair:

- `ResolveExactFinalInferenceResult(experiment_id, last_model_id)`
- `SelectAuthoritativeObservation(...)`

The FINAL result must exactly match model configuration, symbol, prediction horizon, threshold, window size, label rule, target type, completed epochs, and the experiment inference range. It must also be:

```text
status='completed'
inference_scope='final'
checkpoint_eval_id IS NULL
parent_experiment_id IS NULL
```

The profitability observation must match the exact experiment, model, inference-result ID, FINAL scope, no checkpoint ID, and the current metric canonical/hash.

Checkpoint exclusion is therefore enforced by both the resolver/selector and migration trigger. There is no `MAX(id)`, latest-time, stale-FINAL, checkpoint, other-model, other-experiment, or other-range fallback. Missing, ambiguous, or mismatched evidence is explicitly unavailable.

## 7–11. Immutable snapshot and evaluation provenance

The recommendation snapshot stores:

- Provenance version
- Exact FINAL inference result ID
- Exact profitability observation ID
- Unavailable reason
- Inference scope and range
- Actionable count
- Aggregate return
- Average return
- Metric-definition hash
- Source-content hash
- Observation-identity hash

Evaluation results duplicate those frozen fields and additionally store an observational `profitability_evidence_canonical` and `profitability_evidence_hash`. Retry equality checks include this evidence, preventing reinterpretation.

Historical recommendations retain all Phase 3A columns as `NULL`, meaning legacy/pre-Phase-3A. They are not backfilled.

Semantics are distinct:

- Legacy: provenance version is `NULL`.
- Phase-3A unavailable: version 1, explicit reason, no fabricated values.
- Zero actionable: available observation ID, count `0`, aggregate `0`, average `NULL`.

A database immutability trigger prevents updating the recommendation profitability snapshot.

## 12–19. Decision invariance

The implementation defines:

```cpp
kPhase3AProfitabilityScoringWeight = 0.0
kPhase3AProfitabilityScoreContribution = 0.0
```

Both have compile-time `static_assert(... == 0.0)` checks.

Profitability was not added to scoring components, eligibility predicates, ranking sort keys, tie-break keys, ranking identities, or campaign plan identities.

Deterministic tests prove exact equality for:

- Eligibility and disposition
- Final, raw-positive, raw-penalty, and raw-total scores
- Existing score components
- Decision-bearing evidence canonical/hash
- Evaluation identity canonical/hash
- Ranking order
- Existing semantic/hash tie-breaks
- Selected campaign recommendations
- Campaign plan identity, decisions, reasons, and order

Positive, negative, missing, and zero-actionable profitability all preserve the same decision behavior.

## 20–21. Observability and identity boundary

Existing recommendation status, evaluation output/listing, and campaign-plan candidate output now expose:

```text
final_profitability_evidence
final_profitability_unavailable_reason
final_inference_eval_result_id
final_profitability_observation_id
final_profitability_inference_scope
final_profitability_actionable_count
final_profitability_aggregate_terminal_horizon_log_return_sum
final_profitability_average_terminal_horizon_log_return_per_actionable_prediction
profitability_evidence_hash
profitability_weight=0
profitability_score_contribution=0
```

The profitability canonical/hash is observational only. It is deliberately excluded from recommendation/scoring policy hashes, evaluation decision evidence/identity, ranking identity/tie-breaks, and campaign plan identity. Thus new profitability evidence cannot invalidate or reorder existing decision populations.

## 22–23. Deferred boundaries

Phase 3B issue observed but not repaired: existing ranking populations can expose scoring/evaluator policy-homogeneity or stale cross-policy comparison concerns.

Phase 3C remains fully deferred: no threshold, gate, normalization, percentile, z-score, winsorization, nonzero weight, ranking, tie-break, source preference, or campaign decision was introduced.

## 24. Tests

PASS:

- `Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh`
- `Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.sh`
- `Tests/ExperimentRecommendationPhase3ARepositoryTests.sh`
- `Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sh`
- Candidate generator, scoring, evaluation, ranking, and campaign planning C++20 tests with `-Wall -Wextra -Werror`
- Evaluation and ranking migration tests
- Evaluation repository test
- Campaign planning repository test
- Campaign approval repository test
- `Tests/InferenceProfitabilityTests.sh`
- `Tests/InferenceProfitabilityRepositoryTests.sh`
- `Tests/ContinuationProfitabilityPolicyTests.sh`
- `ContinuationProfitabilityPolicyIsolationTests.py`
- `Tests/CheckpointPolicyHardeningTests.sh`
- `CheckpointPolicyIsolationTests.py`
- `Tests/CheckpointPolicyMigrationTests.sh`

One unrelated pre-existing failure:

- `Tests/ContinuationProfitabilityPolicyMigrationTests.sh` — FAIL: migration 074 checks an existing constraint name globally, skips creating the isolated schema’s constraint, and the fixture reports `zero actionable minimum was accepted`. Continuation policy/migration behavior was deliberately not changed in Phase 3A.

## 25–27. Diff and builds

`git diff --check`: PASS.

Debug:

```text
xcodebuild ... -configuration Debug ... build
```

Fails because Xcode’s user-script sandbox denies the provenance generator’s atomic temporary file.

```text
xcodebuild ... -configuration Debug ... ENABLE_USER_SCRIPT_SANDBOXING=NO build
```

PASS — `BUILD SUCCEEDED`. The provenance script still ran. One unrelated existing unreachable-code warning remains in `CampaignOperationsProductionAdmissionService.cpp:463`, plus the local LLVM22 missing-Info.plist warning.

Release:

```text
xcodebuild ... -configuration Release ... build
```

Correctly stopped with:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

Release remains pending independent verification after commit; the protection was not bypassed.

## 28. Remaining issues and worktree

No Phase 3A functional ambiguity remains. Outstanding unrelated items are the continuation migration-test defect and pre-existing build warnings above.

`git status --short`:

```text
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignPlanning.hpp
 M Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
 M Sources/ExperimentRecommendationCampaignPlanningService.cpp
 M Sources/ExperimentRecommendationEvaluation.cpp
 M Sources/ExperimentRecommendationEvaluation.hpp
 M Sources/ExperimentRecommendationEvaluationRepository.cpp
 M Sources/ExperimentRecommendationEvaluationRepository.hpp
 M Sources/ExperimentRecommendationEvaluationService.cpp
 M Sources/ExperimentRecommendationRepository.cpp
 M Sources/ExperimentRecommendationRepository.hpp
 M Sources/ExperimentRecommendationScoring.hpp
 M Sources/ExperimentRecommendationService.cpp
 M Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
 M Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
 M Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
?? Database/migrations/076_campaign_manager_final_profitability_provenance.sql
?? Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sh
?? Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql
?? Tests/ExperimentRecommendationPhase3AProfitabilityTests.cpp
?? Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh
?? Tests/ExperimentRecommendationPhase3ARepositoryTests.cpp
?? Tests/ExperimentRecommendationPhase3ARepositoryTests.sh
?? Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.cpp
?? Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.sh
?? docs/CampaignManagerPhase3AFinalProfitabilityProvenance.rst
```

`git diff --stat` for tracked files:

```text
17 files changed, 794 insertions(+), 20 deletions(-)
```

Untracked migration, tests, and documentation are not included by `git diff --stat` until staged. No commit was created.