---
title: "LSTM Retry Failed Newest Checkpoint Promotion Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_RetryFailed_NewestCheckpoint_Promotion_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Retry Failed Newest Checkpoint Promotion Implementation

Implemented retry checkpoint promotion.

Files changed:

- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh)

Behavior:

- Failed `train` retries now inspect same-experiment durable periodic checkpoints.
- Candidates require complete existing resume metadata and compatibility: symbol, horizon, threshold, range, LR multipliers, target epoch, and effective Donchian mode.
- Highest completed epoch wins; ties use highest model ID.
- Promotion occurs only when it improves over the effective existing resume source. `resume_model_id` is updated atomically with the retry transition.
- Legacy missing Donchian metadata remains enabled-compatible; it is rejected for `zero_ablation`.
- Non-train retries, requeues, lifecycle worker reset, and historical attempt rows remain unchanged.
- Added `SCHEDULER_RETRY_CHECKPOINT_SELECTION` diagnostics.

Validation:

- `git diff --check` — passed.
- Isolated clean-worktree Release build — passed.
- New isolated retry-promotion integration test — passed.
- Existing `SchedulerControlWorkerIdentityIntegrationTests.sh` — passed.
- No production database rows, workers, or scheduler processes were touched.

Root cause: retry reset lifecycle state but retained the prior `resume_model_id`, ignoring newer same-experiment checkpoint durability.

Race safety: the failed experiment row is locked `FOR UPDATE`; selection and `resume_model_id` mutation occur in the same scheduler-control transaction, before the row becomes dispatchable.

Residual risk: the active checkout’s direct Release build is intentionally blocked by the repository’s clean-tree provenance guard, so Release validation was performed in a disposable clean worktree using the same build command and an isolated binary.

`git status --short`:

```text
 M Sources/ExperimentScheduler.cpp
?? Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
```

`git diff --stat`:

```text
 Sources/ExperimentScheduler.cpp | 285 ++++++++++++++++++++++++++++++++--------
 1 file changed, 233 insertions(+), 52 deletions(-)
```

(The new untracked integration test adds 105 lines and is not included in normal `git diff --stat`.)