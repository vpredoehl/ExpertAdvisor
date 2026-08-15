---
title: "LSTM Retry Failed Checkpoint Resumability Hardening"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_RetryFailed_Checkpoint_Resumability_Hardening_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Retry Failed Checkpoint Resumability Hardening

Implemented and verified the retry checkpoint hardening.

- Root cause: retry promotion only validated durable resume metadata; it could choose a newer checkpoint missing required model/optimizer tensors.
- Canonical state contract: `model_meta`/LSTM gate shape; complete `param`, `bias`, `returnHeadWeight`, `returnHeadBias`; six-field `target_meta`; supported SGD `optimizer_meta` with no moment buffers. Existing train configuration/symbol/range/Donchian compatibility checks remain unchanged.
- Added shared `PgModelIO::validateTrainingResumeState` / `hasTrainingResumeState`, used by production resume, retry selection, and model-info resumability reporting.
- Metadata-compatible but incomplete checkpoints are now skipped before epoch ranking. Locking, dry-run behavior, non-training retry isolation, and atomic promotion remain intact.

Changed:

- [Headers/PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp)
- [LSTM/main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp)
- [Sources/ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh)

Regression fixture updates make every positive checkpoint contain actual persisted resume state. Added a newer epoch-70 checkpoint missing `optimizer_meta`; retry correctly promotes the complete epoch-60 checkpoint instead.

Passed:

```bash
bash Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh DerivedData/ExpertAdvisor/Build/Products/Debug/LSTM_Release
bash Tests/SchedulerControlWorkerIdentityIntegrationTests.sh DerivedData/ExpertAdvisor/Build/Products/Debug/LSTM_Release
git diff --check
```

Build:

- Requested Release build was blocked before compilation by the project provenance script because it requires a clean worktree.
- Debug scheme build succeeded with `ENABLE_USER_SCRIPT_SANDBOXING=NO`; no warning was introduced by this change. Existing libpqxx deprecation warnings remain.

`git status --short`:

```text
 M Headers/PgModelIO.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Implementation_Output.md
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Reverification.tar.gz
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Reverification_Findings.md
?? Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
```

`git diff --stat`:

```text
 Headers/PgModelIO.hpp           |  51 +++++++
 LSTM/main.cpp                   |   2 +
 Sources/ExperimentScheduler.cpp | 311 +++++++++++++++++++++++++++++++---------
 3 files changed, 297 insertions(+), 67 deletions(-)
```

The integration test is currently untracked, so it is not included in the normal `git diff --stat`; its standalone delta is 120 lines.