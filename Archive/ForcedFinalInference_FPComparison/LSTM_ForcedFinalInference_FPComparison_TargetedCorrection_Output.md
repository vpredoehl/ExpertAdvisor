---
title: "LSTM Forced Final Inference FP Comparison Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ForcedFinalInference_FPComparison_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Forced Final Inference FP Comparison Targeted Correction

Implemented the targeted forced-final-inference correction.

- Forced attempts now use `abs(threshold_logret - expected) <= 1e-7` for completion and reconciliation.
- Non-forced attempts retain exact threshold equality.
- All other identity fields remain exact.
- Regression coverage accepts a `5e-8` difference and rejects a `0.0002` difference.
- Phase 4A changes were preserved.
- No migrations, database mutations, scheduler/workers, or experiments were run.

Validation:

- Shell syntax: passed.
- Debug build: passed.
- `git diff --check`: passed.
- Release build: blocked by the expected clean-worktree provenance guard.
- Integration test execution skipped because it launches scheduler/workers and applies migrations, prohibited by the request.

Files:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:10205>)
- [SchedulerForcedFinalInferenceRerunIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerForcedFinalInferenceRerunIntegrationTests.sh:268>)
- [Implementation report](</Volumes/Developer SSD/ExpertAdvisor/LSTM_ForcedFinalInference_FPComparison_TargetedCorrection_Output.md>)