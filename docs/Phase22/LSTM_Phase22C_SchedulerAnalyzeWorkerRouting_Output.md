---
title: "LSTM Phase 22C Scheduler Analyze Worker Routing"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22C_SchedulerAnalyzeWorkerRouting_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Phase 22C Scheduler Analyze Worker Routing

Implemented Phase 22C routing.

- ANALYZE jobs now reserve and launch canonical `lstm-analyze-worker`; train and inference routing remain unchanged.
- Added optional `--analyze-worker` role-specific scheduler override; default resolves to the canonical sibling Release product.
- Preserved managed argv contract, persisted executable identity, and existing reconciliation safeguards.
- Compatibility path `LSTM_Release --analyze-experiment=...` remains intact.
- Report: [LSTM_Phase22C_SchedulerAnalyzeWorkerRouting_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase22C_SchedulerAnalyzeWorkerRouting_Output.md)

Validation passed:

- `SchedulerDaemonConfigurationTests`
- `SchedulerAnalyzeWorkerRoutingTests`
- `StandaloneAnalyzeWorkerCliTests`
- `git diff --check`
- Focused `lstm-analyze-worker` Release build: succeeded (RC 0)

Scheduler target build remains a prerequisite: Xcode 26.5 rejects target-plus-`-derivedDataPath` without a scheme, and no `lstm-scheduler` scheme exists. I did not add one or alter output policy. No production scheduler, experiments, database state, or semantic-worker artifacts were modified.

Final disposition: **GO WITH PREREQUISITES**.

`git status --short` shows the expected source/test changes plus the untracked report and routing test. `git diff --stat`: 6 tracked files, 67 insertions, 7 deletions (untracked files excluded).