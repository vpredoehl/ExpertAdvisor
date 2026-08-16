---
title: "LSTM Causal Return Direction Imbalance Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalReturnDirectionImbalance_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Return Direction Imbalance Implementation

Implemented the append-only causal return-direction imbalance feature.

- Column 43 added; `feature_size=44`, current `n_in=48`.
- `n_in=47` still projects only columns 0–42; `n_in=48` consumes column 43.
- Recommendation semantics advanced to v15; v3–v14 remain explicit, including v14 reconstruction.
- Implementation report: [LSTM_CausalReturnDirectionImbalance_Implementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalReturnDirectionImbalance_Implementation_Output.md)

Passed: dedicated new feature suite, all ten requested LSTM regression scripts, `ExperimentRecommendationTests`, and `git diff --check`.

Release build was invoked with the required derived-data path but stopped at the project’s expected clean-tree provenance gate (`Release provenance requires a clean source tree`); no final clean Release qualification is claimed.

`git status --short`: modified implementation/contracts/tests/report plus three untracked new feature/test files; no commit was made. `git diff --stat` reports 22 tracked files changed, 170 insertions, 59 deletions (untracked files excluded).