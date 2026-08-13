# LSTM Scheduler Train Orphan Checkpoint Recovery — Independent Reverification

## Executive Summary

**Disposition: PASS WITH NON-BLOCKING TEST-COVERAGE / HARDENING FINDINGS. Commit-ready for the scoped correction.**

The packaged correction fixes the demonstrated production defect. The missing-train-worker branch in `RecoverOrphanedRunningExperiments()` no longer treats every model produced during the dead attempt as proof of completed training. It now routes the candidate through `RecoverTrainOrphanFromModel()`, which distinguishes an intermediate checkpoint from a model whose persisted `completed_epochs` reaches the experiment target. Intermediate checkpoints are requeued as `pending/train` with both `last_model_id` and `resume_model_id` set to the checkpoint; at-target models retain the existing post-train advancement behavior.

The correction also closes the important exact-attempt fencing gap in the checkpoint-requeue helper by threading `workerAttemptId` through both intermediate and final recovery mutations. The focused disposable-database regression exercises the core failure mode, final-model advancement, stale/unbound attempt behavior, mismatch rejection, stop-after-checkpoint preservation, replay idempotence, and `recover-orphans-only` dry-run reporting.

I found no blocking defect in the submitted correction. Two narrow non-blocking gaps remain: the focused test does not explicitly assert preservation of all checkpoint-inference/scientific configuration columns, and the recovery compatibility predicate does not presently compare every available persisted configuration field (notably Donchian-20 mode and LR metadata) before adoption. Those do not invalidate the demonstrated fix because the selected model is constrained to the same `experiment_id`, and the update itself changes only lifecycle/model-pointer fields, but they are worthwhile hardening follow-ups.

## Evidence Reviewed

The reverification used the packaged input as the authoritative basis, including:

- `LSTM_Scheduler_TrainOrphanCheckpointRecovery_Correction_Output.md`
- `LSTM_Scheduler_TrainOrphanCheckpointRecovery_Correction_Transcript.txt`
- the current `Sources/ExperimentScheduler.cpp` diff
- the full untracked `Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh`
- repository status captured in the package
- reported focused-test, Release-build, and `git diff --check` validation evidence

The package records that the broad process suite was intentionally deferred because live scheduler/workers were active and that an existing SQL ownership fixture was blocked by unrelated schema drift (`model.name` missing). Those limitations are not treated as defects in this scoped review.

## Root Cause

**Confirmed.**

The defective missing-process train path selected the latest model created for the experiment during the worker attempt and directly called `TransitionAfterTrainModelAvailable(...)`. That helper is a post-training transition helper: after setting `last_model_id`, it advances to inference/analyze (or otherwise terminalizes according to existing completion state). It does not determine whether the model represents an intermediate checkpoint.

The repository already had the correct semantic discriminator in `RecoverTrainOrphanFromModel()`: load recoverable model metadata, validate it against the experiment, compare persisted `completed_epochs` with `target_epochs`, advance only at/above target, and otherwise requeue training from that checkpoint. The production failure on experiment 551 is exactly explained by bypassing that helper: model 1601 represented epoch 40 while the target was epoch 80, yet the old branch treated its existence as completed training.

## Correction Verification

### Intermediate checkpoint behavior

**PASS.**

The changed missing-train-worker branch now calls:

`RecoverTrainOrphanFromModel(transaction, experiment, *modelId, attemptId)`

rather than calling `TransitionAfterTrainModelAvailable(...)` directly.

For `completed_epochs < target_epochs`, the recovery helper calls the requeue path. The requeue mutation sets:

- `status='pending'`
- `phase='train'`
- `last_model_id=<checkpoint>`
- `resume_model_id=<checkpoint>`
- `exit_code=NULL`
- `error_message=NULL`

It does not change `current_epoch`, `current_operation`, `stop_after_checkpoint_epoch`, training/inference ranges, checkpoint configuration, or scientific hyperparameters.

The focused test reproduces the material shape of the experiment-551 condition with `current_epoch=57`, checkpoint epoch 40, target 80, and `stop_after_checkpoint_epoch=60`, and asserts the recovered state is `pending:train`, not `pending:infer`, while preserving epoch 57 and stop epoch 60.

### Final checkpoint / full model behavior

**PASS.**

When persisted `completed_epochs >= target_epochs`, the recovery helper still delegates to `TransitionAfterTrainModelAvailable(...)`. The correction now also forwards the exact `workerAttemptId` into that transition. The focused test creates an epoch-80 model for target 80 and verifies normal advancement to `pending/infer`.

Thus the patch changes only the classification of intermediate recoverable checkpoints and preserves the established final-training transition.

### Exact-attempt fencing

**PASS.**

The intermediate-checkpoint helper previously performed an unfenced experiment update. The patch adds an optional `workerAttemptId` predicate:

`active_scheduler_worker_attempt_id=$3`

when an attempt ID is supplied and requires exactly one affected row through `RequireAffectedExactlyOne(...)`.

The same attempt ID is forwarded into the final-model transition path, which already supports exact-attempt predicates. This preserves the lifecycle binding through the recovery mutation until the reconciliation code terminalizes the durable worker attempt and clears the binding.

## Concurrency and Lifecycle Safety

**PASS for the scoped correction.**

The recovery routine first verifies that the missing process's exact durable attempt still owns the lifecycle row (`status='running'`, expected phase, and matching `active_scheduler_worker_attempt_id`). If the binding has changed, it does not perform destructive recovery on the experiment; the attempt is instead reconciled as `lifecycle_predicate_changed` only when it is no longer bound.

For a still-owned train lifecycle, the newly corrected checkpoint requeue itself repeats the exact-attempt predicate. This is important defense in depth: even though the experiment row was locked earlier in the transaction, the mutation remains explicitly fenced and fails closed if its ownership predicate is not satisfied.

After successful intermediate or final recovery, the worker attempt is terminalized as completed with `process_missing_result_recovered`, and the exact active-attempt binding is then cleared. On unusable model/no-result paths, the experiment and attempt retain the existing failure semantics.

The focused test's unbound-attempt case verifies that an attempt lacking the exact lifecycle binding cannot adopt its checkpoint and leaves the experiment running/train while the stale attempt is terminalized with `lifecycle_predicate_changed`.

## Metadata and Model Selection

**PASS for the demonstrated defect, with one non-blocking hardening finding.**

The selected recovery model is constrained by `FindLatestModelForExperimentSince()` to the same `experiment_id` and to creation at/after the durable attempt start, and candidate ordering prefers the highest persisted completed epoch. `RecoverTrainOrphanFromModel()` then loads required resume metadata and rejects a candidate whose symbol, horizon, threshold, or train dates do not match the experiment.

That is enough to prevent the demonstrated intermediate-checkpoint promotion bug and is exercised by the focused symbol-mismatch test.

However, the packaged code exposes additional compatibility metadata—most notably `donchian20Mode`, plus persisted core/head LR values—but the recovery comparison shown in the package does not compare all of those fields before checkpoint adoption. Because the candidate is tied to the same experiment, this is not a blocker for this correction, but checking the full available scientific configuration would make the "recoverable model" predicate stronger and more self-validating.

A second edge case is that `FindLatestModelForExperimentSince()` returns only one candidate. If the highest-epoch candidate were unusable but an earlier checkpoint from the same attempt were valid, the current path fails rather than searching backward for the best usable checkpoint. The prompt's explicit mismatch test permits rejection and does not require fallback selection, so this is not a commit blocker.

## Dry-Run Verification

**PASS.**

The source intentionally skips durable orphan reconciliation when `--dry-run` is supplied. Before this patch, combining `--recover-orphans-only --dry-run` could still emit `SCHEDULER_ORPHAN_RECOVERY_DONE,recovered_or_failed=0`, which misleadingly looked like a recovery pass had actually been evaluated.

The correction changes only the machine output for that combination:

`SCHEDULER_ORPHAN_RECOVERY_SKIPPED,dry_run=1,reason=durable_reconciliation_disabled`

and suppresses `SCHEDULER_ORPHAN_RECOVERY_DONE`.

The focused test verifies the corrected marker and verifies that worker-attempt count is unchanged. The package supports the narrower claim that orphan lifecycle reconciliation is not performed in dry-run. It does not establish that scheduler startup/authority acquisition is globally mutation-free, and this reverification does not infer that broader property.

## Regression Test Assessment

**PASS, with a non-blocking assertion gap.**

The disposable-database test directly covers the required high-value cases:

- intermediate checkpoint 40 / target 80 returns to `pending/train` and records both model pointers;
- final checkpoint/model at target 80 advances to `pending/infer`;
- exact lifecycle ownership is required;
- mismatched checkpoint metadata is rejected and not adopted;
- `current_epoch=57` and `stop_after_checkpoint_epoch=60` survive recovery;
- rerunning recovery leaves the durable attempt snapshot unchanged;
- `recover-orphans-only --dry-run` reports recovery as skipped;
- real `recover-orphans-only` exits after reconciliation rather than entering ordinary scheduling.

The test is appropriately isolated in a disposable database cloned from production schema, and the package reports it passed.

The main missing assertion is explicit preservation of checkpoint-inference policy/configuration fields and the newer Donchian/scientific configuration fields. Structurally, the SQL requeue statement does not modify them, so preservation follows from the code, but a focused regression assertion would make that contract executable rather than merely structural.

## Known Validation Limitations

The reported broad scheduler process suite was not run because a live scheduler and worker were active. Given the review's explicit safety constraints, deferring that suite is appropriate and is not a defect.

The existing SQL ownership test was reported blocked by unrelated fixture schema drift involving a missing `model.name` column. The focused correction has its own exact-attempt disposable-database coverage, so this unrelated fixture problem does not block the scoped conclusion.

The packaged evidence reports a successful canonical Release build and `git diff --check`. This reverification is based on the supplied package and did not independently build or mutate a database in the ChatGPT sandbox.

## Findings

### Finding 1

- **Severity:** NON-BLOCKING
- **Location:** `RecoverTrainOrphanFromModel()` compatibility predicate
- **Explanation:** The helper loads metadata that includes Donchian-20 mode and LR values, but the shown compatibility predicate checks symbol, horizon, threshold, and train dates only. Full persisted scientific-configuration comparison would better ensure an adopted checkpoint is semantically identical to the experiment, especially as feature modes evolve.
- **Required correction:** None before this commit. Recommended follow-up: compare `meta.donchian20Mode` with `experiment.donchian20Mode` and consider comparing other persisted scientific fields that are authoritative for resume compatibility.

### Finding 2

- **Severity:** NON-BLOCKING
- **Location:** `Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh`
- **Explanation:** The test proves stop-after-checkpoint preservation and the key lifecycle/model-pointer fields, but it does not explicitly assert that checkpoint-inference policy/configuration and other scientific configuration columns are unchanged across intermediate recovery.
- **Required correction:** None before this commit. Recommended follow-up: seed distinctive checkpoint-inference/Donchian configuration values and assert they are unchanged after recovery.

### Finding 3

- **Severity:** NON-BLOCKING
- **Location:** `FindLatestModelForExperimentSince()` + missing-train recovery candidate selection
- **Explanation:** Recovery chooses one highest-completed-epoch model, then rejects/fails if that candidate is unusable. It does not fall back to an earlier valid checkpoint from the same attempt. This does not violate the focused mismatch test and is not implicated in the observed experiment-551 defect, but it is a resilience edge case.
- **Required correction:** None for this scoped patch. Consider a future "highest valid recoverable candidate" selection if corrupt/partial model rows are a realistic operational concern.

No BLOCKING findings.

## Commit Readiness

**YES — commit-ready for the scoped scheduler orphan-recovery correction.**

The production behavior that incorrectly promoted an intermediate checkpoint to completed training is corrected at the right semantic boundary, and the exact-attempt fencing is stronger than before. The focused regression evidence directly covers the incident pattern and the required lifecycle outcomes. The remaining findings are hardening/test-completeness improvements, not reasons to withhold this correction.

Before committing, include both the production change and `Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh`. The implementation report may also be archived according to the repository's normal review-artifact convention.

## Final Judgment

**PASS.** The root cause is confirmed, the patch reuses the existing checkpoint-aware recovery semantics instead of duplicating them, intermediate and final training models are now distinguished correctly, and exact durable worker-attempt ownership is enforced on both recovery outcomes. Dry-run output no longer falsely implies that durable orphan reconciliation ran. The focused disposable-database regression is well targeted and supports commit readiness. The remaining metadata-completeness and test-assertion gaps are non-blocking hardening opportunities rather than defects in the incident correction itself.
