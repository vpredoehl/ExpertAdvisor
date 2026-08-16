---
title: "LSTM Formal Feature Ablation Real-World Smoke Test"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FormalFeatureAblation_RealWorldSmokeTest_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Formal Feature Ablation Real-World Smoke Test

Smoke test stopped safely with verdict: **FAIL**.

The real `LSTM` database is missing the required 069/070 schema effects:

- `experiment.feature_ablation_mask` absent.
- Unique identity index excludes the mask.
- Migration 070 function/trigger absent.
- Migration ledger is not readable to `pqxx`.

No experiments were queued; no scheduler command was invoked. A pre-existing scheduler and experiment 549 worker remained untouched.

Report: [LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md)

Files changed: report only.
Build/tests: none needed; `git diff --check` passed.
`git status --short`: `?? LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md`
`git diff --stat`: no tracked diff.