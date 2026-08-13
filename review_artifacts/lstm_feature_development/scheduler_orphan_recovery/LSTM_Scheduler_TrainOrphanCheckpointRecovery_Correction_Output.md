---
title: "LSTM Scheduler Train Orphan Checkpoint Recovery Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Scheduler_TrainOrphanCheckpointRecovery_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Scheduler Train Orphan Checkpoint Recovery Correction

Implementation report

- Root cause: missing-train recovery called `TransitionAfterTrainModelAvailable()` for any model created during the attempt, promoting checkpoint 1601 as final training.
- Correction: recovery now routes through `RecoverTrainOrphanFromModel()`, which validates metadata, compares `completed_epochs` with `target_epochs`, and requeues intermediate checkpoints.
- Exact attempt fencing is preserved for both requeue and final advancement via `active_scheduler_worker_attempt_id`.
- Dry-run now reports `SCHEDULER_ORPHAN_RECOVERY_SKIPPED` instead of implying recovery occurred.

Files changed:

- [Sources/ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:14622)
- [Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh:1)

Validation:

- Focused orphan recovery test: PASS.
- Stop-after-checkpoint preservation, ownership, mismatch rejection, idempotence, and no-launch checks: PASS.
- Release build: PASS.
- `git diff --check`: PASS.
- Existing SQL ownership test: blocked by unrelated fixture schema drift (`model.name` missing).
- Broad process suite not run because a live scheduler and workers are active.

No live database or production experiment rows were modified.

Dry-run behavior is intentional; the output correction was warranted to avoid misleading recovery claims.

Final judgment: PASS for the scoped correction; broad independent process-suite reverification remains deferred for scheduler safety.

`git status --short`:

```text
 M Sources/ExperimentScheduler.cpp
?? Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh
```

`git diff --stat`:

```text
 Sources/ExperimentScheduler.cpp | 45 +++++++++++++++++++++++++++++------------
 1 file changed, 32 insertions(+), 13 deletions(-)
```