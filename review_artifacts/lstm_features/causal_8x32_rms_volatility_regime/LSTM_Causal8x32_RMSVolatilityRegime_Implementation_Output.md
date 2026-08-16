---
title: "LSTM Causal 8x32 RMS Volatility Regime Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Causal8x32_RMSVolatilityRegime_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal 8x32 RMS Volatility Regime Implementation

Implemented the append-only causal volatility-regime feature.

- New column 38: 8-over-32 prior-return RMS regime; `feature_size=39`, current `n_in=43`.
- Preserved historical projections through width 42; only width 43 consumes column 38.
- Recommendation semantics now default to v10; v3–v9 remain explicit historical reconstructions.
- Added focused feature suite and updated affected regression tests.
- Wrote [implementation output](</Volumes/Developer SSD/ExpertAdvisor-next-feature/LSTM_Causal8x32_RMSVolatilityRegime_Implementation_Output.md>).

Passed:

- `LSTMCausalVolatilityRegimeTests.sh`
- Requested feature/model/session/tick-volume/return-surprise regression suites
- `ExperimentRecommendationTests` warning-as-error direct compile/run
- `git diff --check`

Debug build compiled sources but failed in the existing provenance script due to a DerivedData filesystem permission error; no Release qualification claimed.

`git status --short`: 16 modified files, 4 new files; no commit created.
`git diff --stat`: 16 tracked files, 98 insertions / 39 deletions (new untracked files are not included in that command’s output).