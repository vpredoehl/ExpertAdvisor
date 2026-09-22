---
title: "TG4 Causal LSTM Feature Contract"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Causal_LSTM_Feature_Contract_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Causal LSTM Feature Contract

Created [TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md](/Volumes/Developer%20SSD/ExpertAdvisor/Docs/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md).

- Files changed: documentation only.
- Current repository baseline: input width **77**, semantic layout **7**.
- Recommended first ablation: three break-bar pulses — `tg4_inner_break_any`, `tg4_source_tg3_structurally_eligible`, `tg4_source_tg3_confluent`.
- Key unresolved items: Tensor-vs-TG timestamp parity, canonical market-data parity, production ownership/persistence of frozen TG configuration, capacity behavior, and any future persistence/continuous aggregation semantics.
- Tests passed:
  - `Tests/TG3FibonacciConfluenceIntegrationTests.sh`
  - `Tests/TG4HistoricalEmpiricalEvaluationTests.sh`
  - `Tests/LSTMFeatureVectorParityTests.sh`
  - `Tests/LSTMModelInputCompatibilityTests.sh`
- No Xcode build run; no production source, model semantics, schema, frozen TG4 artifacts, database, scheduler, workers, or experiments changed.
- `git status --short`: `?? docs/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md`
- `git diff --stat` for the untracked document: 1 file, 490 insertions.

Proposed commit message: `Document TG4 causal LSTM feature contract`