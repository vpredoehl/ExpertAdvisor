---
title: "LSTM Causal Directional Persistence Efficiency Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalDirectionalPersistence_Efficiency_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Directional Persistence Efficiency Implementation

Implemented the isolated append-only directional-efficiency feature.

- Column 41: causal eight-completed-close `net / path` efficiency using raw closes.
- `feature_size = 42`; current `n_in = 46`.
- `n_in = 45` remains a 41-column prefix and cannot consume column 41.
- Recommendation semantic default advanced to v13; v3–v12, including explicit v12 reconstruction, remain stable.
- No database, scheduler, or configurable-feature changes.

Documentation: [LSTM_CausalDirectionalPersistence_Efficiency_Implementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalDirectionalPersistence_Efficiency_Implementation_Output.md)

Passed focused/regression suites, including the new [LSTMCausalDirectionalEfficiencyTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Tests/LSTMCausalDirectionalEfficiencyTests.cpp), all requested LSTM feature suites, model-input compatibility, feature-vector parity, recommendation semantic tests, and `git diff --check`.

Release build was attempted with `DerivedData/ExpertAdvisor-next-feature`; it stopped at the expected provenance guard because the worktree is intentionally uncommitted: `Release provenance requires a clean source tree`. No commit was made.

`git status --short`: modified implementation/tests plus new header, focused tests, and implementation output; no unrelated files.
`git diff --stat`: 25 tracked files changed, 105 insertions, 46 deletions (plus four untracked deliverables not counted by `git diff --stat`).