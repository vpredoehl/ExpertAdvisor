---
title: "LSTM Feature Branch Semantic Forward Port"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeatureBranch_SemanticForwardPort_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Feature Branch Semantic Forward Port

Implemented the semantic forward-port without merging/cherry-picking `campaign-operations`.

Report: [LSTM_FeatureBranch_SemanticForwardPort_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_FeatureBranch_SemanticForwardPort_Output.md)

Changed feature/provenance behavior includes full-history warmup scope, persisted configurable Donchian lookback, historical defaults, v5/v6 recommendation reconstruction, scheduler propagation, and model/checkpoint metadata. Existing retry/reconciliation code was extended in place.

Validation:

- Passed: return-feature parity, Donchian compatibility/configured lookback, recommendation canonical/provenance test.
- Passed: syntax checks for `LSTM/main.cpp` and `ExperimentScheduler.cpp`.
- Release build invoked but blocked by the repository’s clean-tree provenance script; no commit was made per instruction.
- DB repository executables were not run: standalone linkage requires the broader project test harness. No production DB data was touched.

`git status --short` shows only task changes; the saved stash was untouched. `git diff --check` passes. `git diff --stat`: 20 tracked files, 731 insertions / 122 deletions, plus migrations, headers, and the report.