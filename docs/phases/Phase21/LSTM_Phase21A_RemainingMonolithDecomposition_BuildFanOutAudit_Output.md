---
title: "LSTM Phase 21A Remaining Monolith Decomposition and Build Fan-Out Audit"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase21A_RemainingMonolithDecomposition_BuildFanOutAudit_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Phase 21A Remaining Monolith Decomposition and Build Fan-Out Audit

Created [LSTM_Phase21A_RemainingMonolithDecomposition_BuildFanOutAudit_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase21A_RemainingMonolithDecomposition_BuildFanOutAudit_Output.md).

Recommendation: extract exactly one next boundary, **MarketDataCore**—a narrow read-only candlestick discovery/query/materialization library. The report defers FeaturePipeline, ExperimentStore, TrainingEngine, InferenceEngine, and ModelCore with evidence-based reasons.

No production code, project settings, tests, schema, workers, builds, or database state were changed.

Validation performed:

- `git diff --check` — passed
- No build run, by audit policy
- `git status --short`:
  ```text
  ?? LSTM_Phase21A_RemainingMonolithDecomposition_BuildFanOutAudit_Output.md
  ```
- `git diff --stat` has no tracked-file changes; the report is untracked.