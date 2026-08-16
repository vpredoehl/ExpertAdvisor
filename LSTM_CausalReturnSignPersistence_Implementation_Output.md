---
title: "LSTM Causal Return Sign Persistence Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalReturnSignPersistence_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Return Sign Persistence Implementation

Implemented the append-only causal return-sign persistence feature.

- Added column 42; `feature_size=43`, current `n_in=47`.
- Preserved all historical input projections; `n_in=46` excludes column 42.
- Advanced recommendation semantic default to v14 while retaining explicit v3–v13 reconstruction, including v13.
- Added focused feature/integration tests and the requested implementation report: [LSTM_CausalReturnSignPersistence_Implementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalReturnSignPersistence_Implementation_Output.md).

Validation passed:

- New causal persistence test.
- All requested standalone LSTM regression scripts.
- `ExperimentRecommendationTests` and outcome-assessment unit test builds/runs.
- Syntax checks for changed materialization/outcome repository units.
- `git diff --check`.

Build result: the requested Release build was attempted using `DerivedData/ExpertAdvisor-next-feature`, but correctly stopped at the project’s clean-source provenance guard because changes are uncommitted. No Release qualification is claimed. A Debug build was also blocked during provenance generation by a DerivedData permission error.

`git status --short`: modified implementation/test files plus four new untracked files; no commit created. `git diff --stat`: 20 tracked files changed, 96 insertions / 44 deletions, plus the new helper, tests, shell runner, and report.