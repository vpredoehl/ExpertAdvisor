---
title: "Phase 22Z5 Thin Standalone Inference Worker Extraction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22Z5_ThinStandaloneInferenceWorkerExtraction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 22Z5 Thin Standalone Inference Worker Extraction

Phase 22Z5 extraction is implemented and documented: [report](/Volumes/Developer%20SSD/ExpertAdvisor/docs/Phase22/LSTM_Phase22Z5_ThinStandaloneInferenceWorkerExtraction_Output.md).

Changed:
- New `lstm-infer-worker` target and shared `LSTM Infer Worker` scheme.
- Thin worker main plus shared managed-inference argv adapter.
- `LSTM_Release` now uses the same adapter for managed final/checkpoint inference.
- Focused Phase 22Z5 parity/rejection/linkage test.

Validation passed:
- Debug builds for `LSTM Infer Worker` and `LSTM Debug`.
- Required Phase 22Z1/Z3/Z4/U/V/W, profitability, and semantic-admission tests.
- New worker final/checkpoint parity and registration-failure status `125`.
- `git diff --check`; project plist validation.

Not run: detached-materialization concurrency fixture test, because its required retained Release fixture binary is absent. No DB-backed recovery test was run: active production training workers were detected and left untouched.

Release provenance validation remains intentionally pending a commit and clean tree. The Phase 23A handoff notes a real registry prerequisite: current registry selection is layout-keyed, so role-aware inference publication must preserve the existing layout-7 training/reference artifact.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M LSTM/main.cpp
 M Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh
?? ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Infer Worker.xcscheme
?? LSTM/InferWorkerMain.cpp
?? Sources/ManagedInferenceWorkerCli.cpp
?? Sources/ManagedInferenceWorkerCli.hpp
?? Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh
?? docs/Phase22/LSTM_Phase22Z5_ThinStandaloneInferenceWorkerExtraction_Output.md
```

`git diff --stat` (tracked files):

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 206 +++++++++++++++++++++
 LSTM/main.cpp                            |  41 +---
 Tests/LSTMPhase22Z4...Tests.sh           |  11 +-
```

Phase 22Z5 thin standalone inference worker extraction: IMPLEMENTATION COMPLETE — PENDING POST-COMMIT RELEASE VALIDATION