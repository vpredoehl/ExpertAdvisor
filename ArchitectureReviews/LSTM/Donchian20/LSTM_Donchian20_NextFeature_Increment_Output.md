---
title: "LSTM Donchian20 Next Feature Increment"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Donchian20_NextFeature_Increment_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Donchian20 Next Feature Increment

Implemented the causal Donchian-20 increment. Full report: [LSTM_Donchian20_NextFeature_Increment_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Donchian20_NextFeature_Increment_Output.md).

Key result: active models are now 38-wide (34 base, including Donchian at 32–33, plus four returns); legacy 36-wide models explicitly project to the historical 32-base prefix plus returns.

Changed production files:

- `Headers/{FeatureLayout,DonchianFeatures,ModelInputContract}.hpp`
- `Headers/{Params,LSTM,PgModelIO,Tensor}.hpp`
- `LSTM/{Tensor,LSTM,main}.cpp`

Changed/added tests:

- `Tests/Donchian20FeatureIncrementTests.{cpp,sh}`
- `Tests/{LSTMFeatureVectorParityTests,LSTMStatefulFeatureWarmupTests,ExperimentRecommendationTests}.cpp`

Verification passed:

- `Tests/Donchian20FeatureIncrementTests.sh`
- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/LSTMStatefulFeatureWarmupTests.sh`
- Compiled and ran `ExperimentRecommendationTests.cpp`
- Requested `xcodebuild ... clean build` — **BUILD SUCCEEDED**

DB-backed tests were deferred to avoid modifying shared database state, as requested. No production scheduler/H4 state or DB rows were changed. No commit created.

`git status --short` still includes the pre-existing modified database backup files, plus this increment’s files. `git diff --stat` reports 12 tracked files changed, 386 insertions, 136 deletions; new untracked headers/tests/report are listed in status.