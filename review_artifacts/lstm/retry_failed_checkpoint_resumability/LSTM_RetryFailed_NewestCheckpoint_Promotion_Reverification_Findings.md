# LSTM Retry-Failed Newest Checkpoint Promotion — Independent Reverification

**Date:** 2026-08-15
**Package reviewed:** `LSTM_RetryFailed_NewestCheckpoint_Promotion_Reverification.tar.gz`
**Scope:** static implementation review, packaged evidence review, integration-test design review, shell syntax verification, package integrity check.

## Verdict

**NEEDS ONE CODE CORRECTION BEFORE I WOULD CALL THIS FULLY REVERIFIED.**

The core retry-promotion design is sound: failed training retries are serialized with `FOR UPDATE`, same-experiment periodic checkpoints are ranked by completed epoch then model ID, compatibility is checked against the experiment configuration, promotion is applied atomically with the failed→pending lifecycle reset, dry-run is non-mutating, non-train retries are left alone, and the existing worker-identity reset remains intact.

However, the new selector currently verifies only the metadata loaded by `LoadQueueResumeMeta()`. It does **not** verify the model is actually resumable according to the repository's own resumability definition. `PrintModelInfo`/model inspection defines a resumable model as having at least:

- `train_config_meta`
- `optimizer_meta`
- `train_symbol_meta`
- `param`
- `bias`

The new retry selector only proves the candidate has sufficient queue-resume metadata (train config, symbol/range and Donchian mode through `LoadQueueResumeMeta`). Therefore a higher-epoch periodic checkpoint with missing optimizer/model tensors can be selected and written into `resume_model_id`, even though the same source file would report that model as **not resumable**.

This is not merely a missing test assertion: the new integration fixtures themselves create checkpoint models without `optimizer_meta`, `param`, or `bias`, then treat those fixtures as promotable. As a result, the test proves selection and dry-run command construction, but it does not prove that the selected checkpoint could successfully enter the real resume path.

## Recommended correction

Before accepting the implementation, add a single shared resumability predicate and require it in `TryLoadCompatibleRetryResumeMeta()` (or immediately before calling `LoadQueueResumeMeta`). At minimum it should enforce the same fields already used by the repository's model-info resumability determination:

```cpp
MatrixParamExists(w, modelId, "train_config_meta") &&
MatrixParamExists(w, modelId, "optimizer_meta") &&
MatrixParamExists(w, modelId, "train_symbol_meta") &&
MatrixParamExists(w, modelId, "param") &&
MatrixParamExists(w, modelId, "bias")
```

Prefer factoring this into a shared helper so retry selection and model-info reporting cannot drift.

Then extend `SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh` with a higher-epoch checkpoint that has compatible metadata but is intentionally missing one required resume payload (for example `optimizer_meta`). The selector should skip it and choose the newest lower, fully resumable checkpoint. Ideally, the positive fixtures used for promotion should also contain the minimum model/optimizer payload required by the real resume loader, rather than metadata-only synthetic models.

## What reverified successfully

### 1. Race/dispatch safety

`RunSchedulerControlCommand()` loads the target row with `FOR UPDATE` when `--yes` is applying a change. Checkpoint selection occurs while that row lock is held, and `resume_model_id` is updated in the same transaction as the failed→pending transition. This prevents the scheduler from dispatching the row between selection and mutation.

Relevant source: `ExperimentScheduler.cpp` around lines 6789–6821 and 7145–7214.

### 2. Compatibility checks are appropriately broad

`QueueResumeCompatibilityFailure()` checks:

- target epoch is greater than checkpoint completed epoch
- symbol
- prediction horizon
- threshold
- training start/end range
- core learning-rate multiplier
- head learning-rate multiplier
- Donchian mode

The retry-specific requirements use the persisted experiment values, with default LR values substituted when the experiment columns are null. This is materially better than simply choosing the greatest model ID.

Relevant source: lines 4730–4775 and 6849–6862.

### 3. Selection ordering is deterministic

Compatible candidates are ranked by:

1. greatest `completedEpochs`
2. greatest `modelId` on an epoch tie

That directly addresses the 554 scenario where model 1605 / epoch 60 should supersede model 1601 / epoch 40.

Relevant source: lines 6904–6925.

### 4. Promotion does not regress a valid existing resume source

If the existing effective resume source loads compatibly, a candidate at the same or an older epoch is not promoted (`no_newer_compatible_checkpoint`). This preserves the current resume source when there is no actual progress improvement.

Relevant source: lines 6934–6938.

### 5. Legacy Donchian behavior is intentionally preserved

The test covers legacy checkpoints with missing Donchian metadata as enabled-compatible, while rejecting that legacy fallback for a `zero_ablation` experiment. This matches the migration/backfill semantics shown in the package and the existing compatibility model used elsewhere in the codebase.

### 6. Non-train retry behavior is preserved

Checkpoint promotion runs only when `action == "retry_failed" && row->phase == "train"`. The integration test verifies a failed infer retry remains infer and does not emit checkpoint-selection diagnostics.

Relevant source: lines 7178–7183; test lines 95 and 99.

### 7. Worker identity reset remains intact

The transition still clears PID, PGID, process-start identity, executable, command line, active attempt, pause/control state, timestamps, exit state, error state, and current operation for retry/requeue actions. The new `resume_model_id` mutation is additive and scoped to a positive promotion decision.

Relevant source: `ApplySchedulerControlTransition()` around lines 7083–7120.

### 8. Historical attempts and unrelated rows are protected by tests

The new test snapshots `experiment_scheduler_worker_attempt` and an unrelated experiment row, performs retries, and asserts both remain unchanged.

Test lines 76–101.

### 9. Dry-run behavior is covered

The test verifies the promotion decision is printed during dry-run and that the experiment row is not mutated.

Test lines 78–81.

### 10. Scheduler dispatch command uses the promoted model

The integration test runs scheduler-once in dry-run mode and checks the generated train command includes both the expected Donchian mode and promoted `--resume-model-id`.

Test lines 103–104.

## Integration-test coverage reviewed

The 105-line test covers:

- 40→60 promotion from an external resume source (554-like case)
- newest checkpoint selection when no resume source exists
- refusal to regress to an older same-experiment checkpoint
- epoch tie-break by highest model ID
- symbol mismatch
- horizon mismatch
- threshold mismatch
- train-range mismatch
- core-LR mismatch
- head-LR mismatch
- Donchian mismatch
- legacy missing-Donchian compatibility for enabled
- rejection of legacy fallback for zero-ablation
- checkpoint at target epoch rejected
- failed infer retry unchanged
- dry-run non-mutation
- worker identity cleanup
- historical worker-attempt preservation
- unrelated experiment preservation
- promoted model appearing in scheduler-generated train command

The principal missing case is **metadata-compatible but non-resumable checkpoint payload**.

## Package integrity finding

All packaged files except `SHA256SUMS.txt` itself match the supplied checksums. `SHA256SUMS.txt` contains a checksum entry for itself; because writing that entry necessarily changes the file, self-verification fails. This is a packaging-script defect, not a source-code defect.

Observed result:

- migrations: checksum OK
- `ExperimentScheduler.cpp`: checksum OK
- implementation output: checksum OK
- test scripts: checksum OK
- diff/repository-state/reference files: checksum OK
- `SHA256SUMS.txt`: **self-check fails**

Fix future packaging by excluding `SHA256SUMS.txt` from its own manifest, or generate a second detached checksum for it after the manifest is complete.

## Validation I could perform independently in this environment

- extracted and inspected every packaged file
- verified all non-self-referential SHA-256 entries
- `bash -n SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh` — **PASS**
- `bash -n Tests/SchedulerControlWorkerIdentityIntegrationTests.sh` — **PASS**
- checked the modified source and new test for trailing whitespace — **PASS**
- traced the retry selection, compatibility, locking, transition, dry-run, and dispatch paths in the supplied source
- cross-checked the new test assertions against the implementation

I could **not independently rerun the Release build or integration test binary execution** from this tarball because it does not contain a runnable isolated `LSTM_Release` or the complete Xcode project/source tree needed to rebuild one. The supplied implementation report states those tests passed, but that execution claim is packaged evidence rather than an independently repeated run here.

## Final disposition

**Code:** conditional reject / revise-one-item. The checkpoint promotion algorithm is otherwise well-structured and targeted, but it should not promote a checkpoint until it has proven the candidate is actually resumable by the same repository criteria used elsewhere.

**Test suite:** strong coverage, but currently validates metadata-only synthetic models as resumable. Add an incomplete-payload rejection case and make positive checkpoint fixtures minimally resumable.

**Packaging:** usable, with one harmless but real `SHA256SUMS.txt` self-reference defect.

Once the resumability predicate and corresponding test are added, I would expect this change to be ready for another short re-verification rather than another broad design review.
