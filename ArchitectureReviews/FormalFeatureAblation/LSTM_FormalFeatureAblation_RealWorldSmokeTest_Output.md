---
title: "LSTM Formal Feature Ablation Real-World Smoke Test"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FormalFeatureAblation_RealWorldSmokeTest_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Formal Feature Ablation Real-World Smoke Test

Verdict: **PASS WITH CAVEAT**.

Queued exactly two real DB experiments:

- Control: `576`, mask `''`
- Ablation: `577`, mask `directional_adverse_excursion`

All identity fields match except the mask; both use `duplicate_nonce=0`. No model rows exist yet.

Execution was safely blocked: 12 unrelated pending experiments would be dispatched first by the non-targetable scheduler. I did not start it, alter global scheduler state, or modify any existing experiment.

Report: [LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md)

Files changed: report only.
Build/tests: none (scheduler-safety blocker).
`git status --short`: `?? LSTM_FormalFeatureAblation_RealWorldSmokeTest_Output.md`
`git diff --stat`: empty (untracked report).