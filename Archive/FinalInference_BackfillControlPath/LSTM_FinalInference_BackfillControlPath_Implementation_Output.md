---
title: "LSTM Final Inference Backfill Control Path Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FinalInference_BackfillControlPath_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Final Inference Backfill Control Path Implementation

Implemented the forced scheduler-managed FINAL inference rerun path without changing normal inference deduplication.

### Behavior

- Migration [078_operator_forced_final_inference_rerun.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/078_operator_forced_final_inference_rerun.sql>) adds:
  `experiment.operator_forced_final_inference_rerun_requested boolean NOT NULL DEFAULT false`.
- Confirmed `--requeue-inference=ID --yes` sets the flag and transitions to `pending/infer`.
- Ordinary pending inference still emits `SCHEDULER_SKIP_EXISTING_INFERENCE` and skips when an exact completed FINAL result exists.
- Forced pending inference emits `SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_DISPATCH` and launches through the existing fenced worker-attempt path.
- The flag is not cleared at reservation or launch.
- It is cleared atomically when the exact active attempt has matching completed FINAL evidence with `completed_at >= attempt.reserved_at`.
- Failed forced attempts remain `failed/infer`, retain the flag, and receive:
  `forced_final_inference_rerun_missing_attempt_result;...`
- Old infer logs cannot falsely satisfy a forced rerun.
- Requeueing after failure or after a previously successful rerun works.
- FINAL inference upsert/semantic identity logic in `LSTM/main.cpp` is unchanged.
- Checkpoint inference, requeue-analysis, and retry-failed behavior remain separate and unchanged.
- Scheduler status now exposes the durable flag in machine, table, and compact output.

The main implementation is in [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:10147>). The operator runbook is [SchedulerForcedFinalInferenceRerun.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/SchedulerForcedFinalInferenceRerun.rst>).

### Tests

Added [SchedulerForcedFinalInferenceRerunIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerForcedFinalInferenceRerunIntegrationTests.sh>), covering:

- Normal existing-result skip.
- Durable requeue intent.
- Forced dispatch and real worker reservation.
- Restart before launch.
- Two ownership handoffs while the exact worker is active, with no duplicate attempt.
- Deterministic failure with retained intent.
- Second requeue.
- Attempt-relative result recovery and flag consumption.
- Reuse of result ID `978022`.
- Profitability observation linked to reused result ID `978022`.
- Rejection of duplicate completed FINAL semantic identity.
- Unaffected requeue-analysis and checkpoint paths.
- Non-destructive migration replay.

Passed:

```text
SchedulerForcedFinalInferenceRerunIntegrationTests passed
SchedulerControlWorkerIdentityIntegrationTests passed
SchedulerAnalyzeOrphanRecoveryTests passed
SchedulerRetryFailedCheckpointPromotionIntegrationTests passed
```

All modified shell tests passed `bash -n`; `git diff --check` passed.

### Build

Isolated Debug build succeeded:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/ForcedFinalInferenceRerunDebug \
  ENABLE_USER_SCRIPT_SANDBOXING=NO build
```

Result: `BUILD SUCCEEDED`. It reports 398 existing libpqxx `exec_params` deprecation warnings; the newly added queries use the current parameter API and add no such warnings.

The required Release command was also attempted using isolated DerivedData, but the repository’s intentional provenance gate rejected the dirty worktree before compilation:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
BUILD FAILED
```

A production Release build therefore remains to be run from a clean tree. No commit was made.

### Files changed

- `Sources/ExperimentScheduler.cpp`
- `Database/migrations/078_operator_forced_final_inference_rerun.sql`
- `Tests/SchedulerForcedFinalInferenceRerunIntegrationTests.sh`
- `docs/SchedulerForcedFinalInferenceRerun.rst`
- Eight existing scheduler fixture scripts updated to apply migration 078.

No production migration or experiment state was changed. The production scheduler and training workers remained active and untouched. The production database currently does not contain the new column.

### Production verification

After deploying a clean Release build, apply the migration before starting that binary:

```bash
cd "/Volumes/Developer SSD/ExpertAdvisor"

psql -X -v ON_ERROR_STOP=1 -d LSTM \
  -f Database/migrations/078_operator_forced_final_inference_rerun.sql

CANONICAL_BIN="/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release"
EXPERIMENT_ID=547
```

Capture the current identity:

```bash
BEFORE_RESULT_ID="$(
  psql -X -At -d LSTM -v experiment_id="$EXPERIMENT_ID" <<'SQL'
SELECT r.id
FROM experiment e
JOIN inference_eval_result r
  ON r.model_id=e.last_model_id
 AND r.symbol=e.symbol
 AND r.prediction_horizon=e.prediction_horizon
 AND r.threshold_logret=e.c_next_threshold
 AND r.from_date=e.infer_start::date::text
 AND r.to_date=e.infer_end::date::text
 AND r.status='completed'
 AND r.inference_scope='final'
 AND r.checkpoint_eval_id IS NULL
WHERE e.experiment_id=:experiment_id;
SQL
)"
printf 'Before result ID: %s\n' "$BEFORE_RESULT_ID"
```

Requeue and monitor:

```bash
"$CANONICAL_BIN" --requeue-inference="$EXPERIMENT_ID" --yes
"$CANONICAL_BIN" --scheduler-status | rg "experiment_id=${EXPERIMENT_ID}|Forced FINAL Infer Rerun"
```

Inspect the durable request and worker attempt:

```bash
psql -X -d LSTM -v experiment_id="$EXPERIMENT_ID" <<'SQL'
SELECT e.experiment_id, e.status, e.phase,
       e.operator_forced_final_inference_rerun_requested,
       e.active_scheduler_worker_attempt_id,
       a.lifecycle_state, a.worker_pid, a.reserved_at
FROM experiment e
LEFT JOIN experiment_scheduler_worker_attempt a
  ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id
WHERE e.experiment_id=:experiment_id;
SQL
```

After completion:

```bash
psql -X -d LSTM -v experiment_id="$EXPERIMENT_ID" <<'SQL'
SELECT e.status, e.phase,
       e.operator_forced_final_inference_rerun_requested,
       r.id AS inference_eval_result_id,
       r.completed_at,
       p.profitability_observation_id
FROM experiment e
JOIN inference_eval_result r
  ON r.model_id=e.last_model_id
 AND r.symbol=e.symbol
 AND r.prediction_horizon=e.prediction_horizon
 AND r.threshold_logret=e.c_next_threshold
 AND r.from_date=e.infer_start::date::text
 AND r.to_date=e.infer_end::date::text
 AND r.status='completed'
 AND r.inference_scope='final'
 AND r.checkpoint_eval_id IS NULL
LEFT JOIN inference_profitability_observation p
  ON p.inference_eval_result_id=r.id
WHERE e.experiment_id=:experiment_id;
SQL
```

Confirm exactly one semantic FINAL row and that its ID equals `$BEFORE_RESULT_ID`:

```bash
psql -X -d LSTM -v experiment_id="$EXPERIMENT_ID" <<'SQL'
SELECT min(r.id) AS authoritative_result_id,
       max(r.id) AS same_result_id,
       count(*) AS completed_final_rows
FROM experiment e
JOIN inference_eval_result r
  ON r.model_id=e.last_model_id
 AND r.symbol=e.symbol
 AND r.prediction_horizon=e.prediction_horizon
 AND r.threshold_logret=e.c_next_threshold
 AND r.from_date=e.infer_start::date::text
 AND r.to_date=e.infer_end::date::text
 AND r.status='completed'
 AND r.inference_scope='final'
 AND r.checkpoint_eval_id IS NULL
WHERE e.experiment_id=:experiment_id;
SQL
```

Expected: `completed_final_rows=1`, the original ID is retained, `completed_at` is refreshed, a profitability observation exists, and the force flag is false.

### Repository state

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/ExperimentScheduler.cpp
 M Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh
 M Tests/SchedulerAnalyzeOrphanRecoveryTests.sh
 M Tests/SchedulerContinuationOwnershipIntegrationTests.sh
 M Tests/SchedulerControlWorkerIdentityIntegrationTests.sh
 M Tests/SchedulerOwnershipProcessIntegrationTests.sh
 M Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
 M Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh
 M Tests/WorkerAttemptReconciliationIntegrationTests.sh
?? Database/migrations/078_operator_forced_final_inference_rerun.sql
?? LSTM_CampaignManager_Phase3C_Independent_Verification_Findings.md
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Implementation_Output.md
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Inspection_Output.md
?? LSTM_CampaignManager_Phase3C_TargetedCorrections_Output.md
?? Sources/ProfitabilityDistribution.cpp
?? Sources/ProfitabilityDistribution.hpp
?? Tests/ProfitabilityDistributionTests.cpp
?? Tests/ProfitabilityDistributionTests.sh
?? Tests/SchedulerForcedFinalInferenceRerunIntegrationTests.sh
?? docs/CampaignManagerPhase3CProfitabilityDistribution.rst
?? docs/SchedulerForcedFinalInferenceRerun.rst
?? package_phase3c_profitability_distribution_review.sh
```

The project-file and Phase3C changes were pre-existing and were not modified for this work.

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj            |   8 +
 Sources/ExperimentScheduler.cpp                    | 244 +++++++++++++++++----
 ...InputWidthExpansionSchedulerIntegrationTests.sh |   2 +
 Tests/SchedulerAnalyzeOrphanRecoveryTests.sh       |   2 +
 ...hedulerContinuationOwnershipIntegrationTests.sh |   2 +
 ...hedulerControlWorkerIdentityIntegrationTests.sh |   4 +
 Tests/SchedulerOwnershipProcessIntegrationTests.sh |   2 +
 ...tryFailedCheckpointPromotionIntegrationTests.sh |   1 +
 .../SchedulerTrainOrphanCheckpointRecoveryTests.sh |   2 +
 .../WorkerAttemptReconciliationIntegrationTests.sh |   2 +
 10 files changed, 226 insertions(+), 43 deletions(-)
```

Standard `git diff --stat` does not include the three new untracked deliverables.