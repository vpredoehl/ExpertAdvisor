---
title: "LSTM Causal Multi-Bar Range Pressure Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalMultiBarRangePressure_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Causal Multi-Bar Range Pressure Implementation

Implemented `CausalMultiBarRangePressure` at tensor slot 45. Current input width is now 50; width-49 models preserve their 45-column historical projection.

Report: [LSTM_CausalMultiBarRangePressure_Implementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalMultiBarRangePressure_Implementation_Output.md)

Changed production files include the new causal helper, Tensor integration, feature layout, ablation registry, and model-input compatibility contract. Added focused feature test/runner and updated explicit width assertions in affected feature tests.

Passed focused tests:

- `bash Tests/LSTMCausalMultiBarRangePressureTests.sh`
- Causal-feature regression runners, feature-vector parity, model-input compatibility, relative-volume, and session-phase tests
- `git diff --check`

Release build attempted:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

It was blocked by the existing build-provenance script requiring a clean source tree—not by compilation—and did not run `LSTM_Release`.

`git status --short`:

```text
M Headers/FeatureAblation.hpp
M Headers/FeatureLayout.hpp
M Headers/ModelInputContract.hpp
M Headers/Tensor.hpp
M LSTM/LSTM.cpp
M LSTM/Tensor.cpp
M Tests/DonchianTensorIntegrationTests.cpp
M Tests/LSTMCausalCloseLocationTests.cpp
M Tests/LSTMCausalDirectionalAdverseExcursionTests.cpp
M Tests/LSTMCausalDirectionalEfficiencyTests.cpp
M Tests/LSTMCausalDirectionalRangeTests.cpp
M Tests/LSTMCausalReturnDirectionImbalanceTests.cpp
M Tests/LSTMCausalReturnSignPersistenceTests.cpp
M Tests/LSTMCausalReturnSurpriseTests.cpp
M Tests/LSTMCausalVolatilityRegimeTests.cpp
M Tests/LSTMFeatureVectorParityTests.cpp
M Tests/LSTMModelInputCompatibilityTests.cpp
M Tests/LSTMRelativeTickVolumeTests.cpp
M Tests/LSTMTrueSessionPhaseTests.cpp
?? Headers/CausalMultiBarRangePressureFeatures.hpp
?? LSTM_CausalMultiBarRangePressure_Implementation_Output.md
?? Tests/LSTMCausalMultiBarRangePressureTests.cpp
?? Tests/LSTMCausalMultiBarRangePressureTests.sh
```

`git diff --stat`: 19 tracked files changed, 89 insertions, 46 deletions (new untracked files listed above are excluded by Git’s default diff stat).

Assessment: **PASS WITH CAVEAT** — implementation and focused tests pass; full Release build is gated by clean-worktree provenance enforcement.