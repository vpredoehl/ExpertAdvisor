---
title: "LSTM Fixed Causal 32-Bar Relative Tick Volume Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FixedCausal32BarRelativeTickVolume_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Fixed Causal 32-Bar Relative Tick Volume Implementation

Implemented the fixed causal relative tick-volume increment.

- Formula: `log((vol_t + 1) / (mean(prior up-to-32 volumes) + 1))`. The current bar is numerator-only; reference is the prior 1–32 completed bars. Bootstrap is `0`; zero volumes are valid; invalid direct inputs sanitize to `0`; output is always finite.
- Column 36 is `relativeTickVolumeCol`; `feature_size = 37`.
- `n_in = 41` remains correct: 37 canonical tensor features + 4 existing return channels. Historical projections remain 36→32, 38→34, 40→36; new models use 41→37.
- Recommendation semantic default is now `v8`; v3–v7 canonical reconstruction remains explicit.

Changed:
- Feature ingestion/construction: [PricePoint.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PricePoint.hpp), [RelativeTickVolumeFeatures.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/RelativeTickVolumeFeatures.hpp), [Tensor.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp), [db_cursor.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp).
- Layout/model compatibility: [FeatureLayout.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureLayout.hpp), [ModelInputContract.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp).
- Provenance: [ExperimentRecommendation.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp) and version-aware campaign reconstruction sites.
- Tests/docs: new [LSTMRelativeTickVolumeTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMRelativeTickVolumeTests.cpp) plus existing feature, Donchian, model-input, session-phase, and recommendation tests; [Volume_II_Data_Pipeline.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/Volume_II_Data_Pipeline.md).

Passed:

- `Tests/LSTMRelativeTickVolumeTests.sh`
- `Tests/LSTMModelInputCompatibilityTests.sh`
- `Tests/LSTMTrueSessionPhaseTests.sh`
- `Tests/LSTMFeatureVectorParityTests.sh`
- Donchian feature and Tensor integration tests
- `ExperimentRecommendationTests`
- Full Debug build: succeeded.

Release build:

- Ran the required Release command.
- It failed before compilation (`exit 65`) because `GenerateBuildProvenance.py` requires a clean Git tree; the tree necessarily contains this implementation plus the pre-existing untracked review output. I did not bypass, stash, or weaken that provenance guard. Therefore, Release build success is unverified.

Residual risk: DB-backed recommendation workflow tests were not run; the production cursor and repositories did compile in the successful Debug build.

`git status --short`:

```text
 M Common/db_cursor.cpp
 M Headers/FeatureLayout.hpp
 M Headers/ModelInputContract.hpp
 M Headers/PricePoint.hpp
 M Headers/Tensor.hpp
 M LSTM/Tensor.cpp
 M LSTM/main.cpp
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
 M Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
 M Tests/DonchianFeatureTests.cpp
 M Tests/DonchianTensorIntegrationTests.cpp
 M Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
 M Tests/ExperimentRecommendationTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMModelInputCompatibilityTests.cpp
 M Tests/LSTMTrueSessionPhaseTests.cpp
 M docs/architecture/Volume_II_Data_Pipeline.md
?? Headers/RelativeTickVolumeFeatures.hpp
?? LSTM_FeaturePipeline_PostSessionPhase_NextIncrement_Review_Output.md
?? Tests/LSTMRelativeTickVolumeTests.cpp
?? Tests/LSTMRelativeTickVolumeTests.sh
```

`git diff --stat`: 19 tracked files changed, 129 insertions, 44 deletions (the three untracked files are not included in Git’s diff stat).