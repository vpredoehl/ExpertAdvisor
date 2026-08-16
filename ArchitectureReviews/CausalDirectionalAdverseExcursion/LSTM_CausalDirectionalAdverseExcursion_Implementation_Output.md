---
title: "LSTM Causal Directional Adverse Excursion Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalDirectionalAdverseExcursion_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Directional Adverse Excursion Implementation

Implemented the isolated append-only feature increment.

- Added causal adverse-excursion ratio at Tensor column 44; `feature_size=45`, current `n_in=49`.
- Preserved historical n_in=36…48 projections; n_in=48 excludes column 44.
- Advanced recommendation semantic default to v16; v3–v15, including explicit v15 reconstruction, remain supported.
- Added focused helper/Tensor/compatibility tests and updated affected maintained expectations.
- Wrote [implementation report](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalDirectionalAdverseExcursion_Implementation_Output.md).

Validation passed: focused suite, all requested causal/model-input/parity scripts, direct recommendation and outcome-assessment tests, changed repository-source compiles, and `git diff --check`.

Build attempt used the requested DerivedData path but stopped at the existing Release provenance guard because the worktree is intentionally dirty/uncommitted. No commit was made. Database-backed materialization execution was not run because an active scheduler/training worker exists in the separate production worktree.

`git status --short` shows only this increment’s modified/new files. `git diff --stat` reports 22 tracked files changed; new files are untracked and therefore omitted from that stat.