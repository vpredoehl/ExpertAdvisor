---
title: "LSTM Scheduler Train Orphan Checkpoint Recovery Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Scheduler_TrainOrphanCheckpointRecovery_IndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Scheduler Train Orphan Checkpoint Recovery Independent Reverification

# LSTM Scheduler Train Orphan Checkpoint Recovery — Independent Reverification

## Disposition

PASS WITH NON-BLOCKING FINDINGS

## Executive Summary

The correction is technically sound and commit-ready for the scoped orphan-checkpoint defect. The prior path directly promoted any model created during a missing train attempt; the corrected path validates metadata and distinguishes intermediate from final checkpoints. Exact-attempt fencing is preserved for both requeue and final advancement.

Remaining findings concern test depth, incomplete compatibility comparisons, and the fact that the current HEAD bundles unrelated changes.

## Evidence Reviewed

- [Correction output](</Volumes/Developer SSD/ExpertAdvisor/review_artifacts/lstm_feature_development/scheduler_orphan_recovery/LSTM_Scheduler_TrainOrphanCheckpointRecovery_Correction_Output.md>)
- Correction transcript
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14539>)
- [Focused regression test](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh:1>)
- Parent-to-HEAD production diff
- Relevant migrations and model metadata helpers
- Read-only process inspection

Independent checks run:

- `bash -n Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh`
- `git diff --check`
- `git status --short`
- `git diff --stat`

## Root Cause

Confirmed.

Before correction, the missing-train branch called `TransitionAfterTrainModelAvailable()` directly after selecting the latest model created during the attempt. That helper advances post-training lifecycle state without checking `completed_epochs`.

The corrected branch at [ExperimentScheduler.cpp:15943](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:15943>) calls `RecoverTrainOrphanFromModel()`, which loads metadata, validates configuration, compares completed epochs with the target, and only advances final checkpoints.

## Correction Verification

Intermediate checkpoints are correctly requeued by [RequeueTrainOrphanFromCheckpoint()](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14622>):

- `status='pending'`
- `phase='train'`
- `last_model_id` and `resume_model_id` set to the checkpoint
- exit/error fields cleared
- phase-control fields, including `current_epoch`, `current_operation`, and `stop_after_checkpoint_epoch`, left unchanged

`BuildTrainCommand()` consumes `resume_model_id`, so the next scheduler pass naturally resumes training from the checkpoint.

Final checkpoints with `completed_epochs >= target_epochs` still call `TransitionAfterTrainModelAvailable()` at [line 14683](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14683>) and advance to pending/infer when an inference range exists.

## Concurrency and Lifecycle Safety

The correction is safely fenced.

- The missing-process path first requires the experiment to be running in the expected phase and bound to the exact attempt at [lines 15798–15823](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:15798>).
- Intermediate requeue adds `active_scheduler_worker_attempt_id=$3` and requires exactly one affected row.
- Final advancement forwards the attempt ID into `TransitionAfterTrainModelAvailable()`, which applies the same predicate.
- Attempt terminalization and active-attempt clearing also require the exact attempt.
- Predicate mismatch causes `RequireAffectedExactlyOne()` failure; the transaction cannot commit a partial recovery.

Unusable or mismatched metadata leaves `completedEvidence=false`; the later path marks the experiment failed and the attempt as `process_missing_no_result`.

## Metadata and Model Selection

`LoadQueueResumeMeta()` loads:

- symbol
- prediction horizon
- threshold
- completed epochs
- core/head learning-rate metadata
- training date range
- Donchian-20 mode

`RecoverTrainOrphanFromModel()` validates symbol, horizon, threshold, and training dates. This is sufficient for the demonstrated defect, especially because the model is constrained to the same experiment and attempt time window.

`FindLatestModelForExperimentSince()` restricts candidates by `experiment_id` and `created_at >= attempt start`, then selects the highest completed epoch, breaking ties by model ID. This behaves sensibly with multiple checkpoints and avoids unrelated experiments.

## Dry-Run Verification

The source skips durable orphan reconciliation whenever `dryRun` is set. It does not create or terminalize worker attempts and now emits:

`SCHEDULER_ORPHAN_RECOVERY_SKIPPED,dry_run=1,...`

The focused test verifies the new marker and absence of `SCHEDULER_ORPHAN_RECOVERY_DONE`.

A nuance: scheduler authority acquisition itself still writes invocation/lease state during startup. Therefore the implementation guarantees no durable orphan reconciliation, not globally zero database mutation.

## Regression Test Assessment

The disposable test meaningfully covers:

- intermediate checkpoint recovery
- final checkpoint advancement
- metadata mismatch failure
- lifecycle binding absence
- stop-after-checkpoint preservation
- replay without extra attempts
- dry-run output and no attempt-count change

The fixtures provide the required 14-field training metadata, symbol metadata, and date-range metadata. They omit explicit Donchian metadata, relying on the enabled default.

Important gaps:

- The unbound-attempt case fails before entering the new exact-attempt requeue/finalization predicates, so it does not directly test a mid-operation stale-attempt race.
- Dry-run checks attempt count but not complete experiment/attempt row snapshots.
- Replay checks attempt lifecycle snapshots but not experiment-row immutability.
- No test covers multiple candidate checkpoints or a corrupt highest-epoch candidate with a valid earlier fallback.

These are test-strength limitations, not evidence of a production fencing defect.

## Known Validation Limitations

The broad process suite was appropriately deferred because a live scheduler and training worker are active. I observed an active scheduler and worker and did not stop, signal, or perturb them.

The reported SQL ownership test failure involving `model.name` is an unrelated fixture/schema-drift infrastructure defect. It does not undermine this correction because the focused test uses a disposable database and independently exercises the relevant recovery path.

No schema migration is required by the orphan-recovery change.

## Findings

### Finding 1

- Severity: NON-BLOCKING
- Location: [focused test lines 125–154](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh:125>)
- Explanation: Exact-attempt fencing is validated indirectly. The unbound case exits before the new `RequireAffectedExactlyOne()` requeue/final-transition predicates execute.
- Required correction: None before commit. A future test should introduce a deterministic stale-binding race or failpoint.

### Finding 2

- Severity: NON-BLOCKING
- Location: [RecoverTrainOrphanFromModel() lines 14660–14664](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14660>)
- Explanation: `QueueResumeMeta` loads Donchian mode and learning-rate metadata, but orphan recovery does not compare those fields with the experiment. Other persisted model configuration fields are also not independently checked here.
- Required correction: None before this scoped commit. Consider expanding the compatibility predicate and adding corresponding fixtures later.

### Finding 3

- Severity: NON-BLOCKING
- Location: [focused test lines 129–161](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh:129>)
- Explanation: Dry-run and replay assertions do not snapshot every potentially mutated row.
- Required correction: None before commit. Add full experiment/attempt snapshots in a future test-hardening change.

### Finding 4

- Severity: NON-BLOCKING
- Location: Current HEAD commit `2817bcf`
- Explanation: The orphan correction-specific diff touches `Sources/ExperimentScheduler.cpp` and the focused test, but the aggregate commit also contains unrelated legacy model compatibility production files, tests, and review artifacts.
- Required correction: If a standalone correction commit is required, curate or split the unrelated changes. This does not invalidate the orphan correction itself.

## Commit Readiness

Yes. The scoped orphan-checkpoint correction should be accepted. It is already present in HEAD `2817bcf`; the current worktree is clean.

- `git status --short`: clean
- `git diff --stat`: empty
- `git diff --check`: passed

## Final Judgment

The root cause is confirmed, the semantic error is corrected at the proper recovery boundary, intermediate checkpoints resume training without premature inference, final checkpoints retain ordinary advancement, and exact worker-attempt fencing is preserved. The remaining issues are non-blocking test and metadata-hardening opportunities.