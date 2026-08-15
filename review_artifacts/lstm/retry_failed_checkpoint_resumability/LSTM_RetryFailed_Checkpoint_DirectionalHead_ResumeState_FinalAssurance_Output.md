---
title: "LSTM Retry Failed Checkpoint Directional Head Resume-State Final Assurance"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_RetryFailed_Checkpoint_DirectionalHead_ResumeState_FinalAssurance_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Retry Failed Checkpoint Directional Head Resume-State Final Assurance

Implemented the resumability correction.

- `UpNeutralDownReturn` now requires structurally valid `returnHeadDirWeight` (`hidden×3`) and `returnHeadDirBias` (`1×3`).
- Canonical `target_meta` parsing is shared by resume validation, resume config, and `loadAll`.
- Regression targets retain optional directional-head loading.
- Retry selection, production resume, and model-info continue through the same canonical resumability helper.

Files changed for this correction:

- [Headers/PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp:511)
- [LSTM/main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:4390)
- [Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh:38)

The fixture now includes all real directional tensors for positive classification resumes; newer checkpoints missing either directional tensor are skipped, the missing-optimizer case remains skipped, and a regression checkpoint without directional tensors is accepted.

Resume-path persisted-read audit:

| Object | Classification |
|---|---|
| `model_meta`, `param`, `bias`, `returnHeadWeight`, `returnHeadBias`, `target_meta`, `optimizer_meta` | `REQUIRED_AND_VALIDATED` |
| `returnHeadDirWeight`, `returnHeadDirBias` for `UpNeutralDownReturn` | `REQUIRED_AND_VALIDATED` |
| `returnHeadDirWeight`, `returnHeadDirBias` for regression targets | `OPTIONAL_BY_DESIGN` |
| `donchian20_mode_meta` | `OPTIONAL_BY_DESIGN` (legacy default retained, then runtime consistency checked) |
| `train_config_meta`, `train_symbol_meta`, `train_range_meta` | `CONFIG_METADATA_VALIDATED_ELSEWHERE` by `LoadResumeCheckpointConfig` and runtime-config application |

No unexplained persisted reads remain in the span from `LoadResumeCheckpointConfig` through `loadAll` / `loadOptimizerMeta`.

Validation run:

- Debug build succeeded:
  `xcodebuild ... -configuration Debug ... ENABLE_USER_SCRIPT_SANDBOXING=NO build`
- `SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh` — passed.
- `SchedulerControlWorkerIdentityIntegrationTests.sh` — passed.
- `git diff --check` — passed.

The Release build was attempted only after the correction, but is blocked before compilation by the project provenance script because the worktree is dirty. Debug compilation emitted existing libpqxx deprecation warnings; this change introduced no new warnings.

Production scheduler and active training worker were observed only; neither was altered.

`git status --short`:

```text
 M Headers/PgModelIO.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
?? LSTM_RetryFailed_Checkpoint_Resumability_Hardening_Output.md
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Implementation_Output.md
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Reverification.tar.gz
?? LSTM_RetryFailed_NewestCheckpoint_Promotion_Reverification_Findings.md
?? Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
```

`git diff --stat`:

```text
 Headers/PgModelIO.hpp           | 195 +++++++++++++++++++++----
 LSTM/main.cpp                   |   8 +-
 Sources/ExperimentScheduler.cpp | 311 +++++++++++++++++++++++++++++++---------
 3 files changed, 418 insertions(+), 96 deletions(-)
```

`Sources/ExperimentScheduler.cpp` and the review artifacts were already modified/untracked before this correction and were not changed by it.