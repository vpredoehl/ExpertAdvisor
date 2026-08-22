---
title: "LSTM Campaign Manager Phase 3A Migration 076 Final Inference Binding Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3A_Migration076_FinalInferenceBinding_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3A Migration 076 Final Inference Binding Targeted Correction

Implemented the targeted migration-076 provenance correction. Worktree remains uncommitted.

## Correction

1. Migration 076 was not operationally applied. The live `LSTM` database showed:

   - No `schema_migrations.version='076'` row.
   - No Phase 3A recommendation columns.
   - No Phase 3A validation function.

   These remained absent after testing; migration tests used transactional test schemas and rolled back.

2. Files changed by this correction:

   - [076_campaign_manager_final_profitability_provenance.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/076_campaign_manager_final_profitability_provenance.sql:236>)
   - [ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql:1>)

   No runner or C++ files were changed during this correction.

3. Previous defect: the trigger checked result ID, model ID, completed/FINAL status, and non-checkpoint shape, but did not bind the result to `source_experiment_id` or the experiment’s exact FINAL inference context. Consequently, unavailable-profitability rows could attach a wrong-but-final result without invoking observation validation.

4. The trigger now mirrors `ResolveExactFinalInferenceResult()` and requires:

   - `experiment.experiment_id = source_experiment_id`
   - `experiment.last_model_id = source_model_id`
   - `model.model_id = experiment.last_model_id`
   - `model.experiment_id = experiment.experiment_id`
   - Complete `train_config_meta`
   - Model horizon equals experiment horizon
   - Model threshold matches experiment threshold within the established tolerance
   - Result symbol, horizon, threshold, window, label rule, target type, completed epochs, and date range exactly match
   - `status='completed'`
   - `inference_scope='final'`
   - `checkpoint_eval_id IS NULL`
   - `parent_experiment_id IS NULL`
   - Exactly one matching result, with the referenced ID

5. `model.experiment_id` is part of the proof. Its FK and migration-070 immutability trigger prevent a linked model from being redirected to another experiment.

6. Exact range/context is enforced using `experiment.infer_start::date`, `experiment.infer_end::date`, and model metadata from `matrix`. The existing completed-FINAL unique index prevents duplicate rows for an identical inference context while still permitting the tested different-range adversary.

## Database-boundary proof

The migration test directly verifies:

- Correct available FINAL result succeeds.
- Correct unavailable FINAL result succeeds with observation ID `NULL`.
- Wrong-experiment completed FINAL result is rejected with SQLSTATE `23514`.
- Wrong-range completed FINAL result is rejected both with and without an observation.
- Checkpoint inference is rejected without relying on observation validation.
- Available positive profitability remains valid.
- Available zero-actionable profitability remains distinct and valid.
- Legacy rows remain unbackfilled.
- Migration 076 replays idempotently.
- Frozen snapshots remain immutable.

The unavailable validation is independent: the exact-result branch runs before and without requiring `source_final_profitability_observation_id`.

## Decision invariants

Profitability remains exactly zero-weighted in [ExperimentRecommendation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp:16>):

```cpp
kPhase3AProfitabilityScoringWeight = 0.0
kPhase3AProfitabilityScoreContribution = 0.0
```

Candidate eligibility, scoring, evaluation identity, ranking, tie-breaking, campaign selection, and campaign-plan identity regressions all passed. No scoring/ranking/C++ production logic changed.

Phase 3B and Phase 3C were not implemented; the targeted diff contains only database validation and adversarial fixtures.

## Tests

Passed:

- `Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sh` — twice; includes replay
- `Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.sh`
- `Tests/ExperimentRecommendationPhase3ARepositoryTests.sh`
- `Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh`
- `Tests/InferenceProfitabilityRepositoryTests.sh`
- `python3 Tests/ContinuationProfitabilityPolicyIsolationTests.py`
- `python3 Tests/CheckpointPolicyIsolationTests.py`
- Existing recommendation tests
- Existing candidate-generation tests
- Existing scoring tests
- Existing evaluation tests
- Existing ranking tests
- Existing campaign-planning tests
- `git diff --check` — PASS

One hand-assembled campaign-planning test command initially omitted its existing conversion-workflow dependency and failed to link. The corrected complete command passed; this was a test invocation issue, not a product failure.

No Xcode build was required because this correction changed only SQL and SQL fixtures, not production C++. Migration 076 was not operationally applied.

Unresolved issues: none within this targeted correction.

## Worktree

Targeted correction stats relative to the reviewed Phase 3A snapshot:

```text
076_campaign_manager_final_profitability_provenance.sql | 85 ++++++++++++++++++++--
1 file changed, 78 insertions(+), 7 deletions(-)

ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql | 163 ++++++++++++++++++---
1 file changed, 141 insertions(+), 22 deletions(-)
```

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
?? LSTM_CampaignManager_Phase3A_FinalProfitabilityProvenance_Implementation_Output.md
?? LSTM_CampaignManager_Phase3A_FinalProfitabilityProvenance_IndependentReview.tar.gz
?? LSTM_CampaignManager_Phase3A_FinalProfitabilityProvenance_IndependentReview/
?? LSTM_CampaignManager_Phase3A_Independent_Diff_Review_Findings.md
?? Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sh
?? Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql
?? Tests/ExperimentRecommendationPhase3AProfitabilityTests.cpp
?? Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh
?? Tests/ExperimentRecommendationPhase3ARepositoryTests.cpp
?? Tests/ExperimentRecommendationPhase3ARepositoryTests.sh
?? Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.cpp
?? Tests/ExperimentRecommendationPhase3ASourceRepositoryTests.sh
?? docs/CampaignManagerPhase3AFinalProfitabilityProvenance.rst
?? package_campaign_manager_phase3a_review.sh
```

`git diff --stat`:

```text
17 files changed, 794 insertions(+), 20 deletions(-)
```

The standard stat excludes untracked migration/test files, hence the separate targeted stats above.