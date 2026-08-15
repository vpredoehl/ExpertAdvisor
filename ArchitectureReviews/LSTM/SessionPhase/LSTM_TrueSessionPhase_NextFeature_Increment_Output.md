---
title: "LSTM True Session Phase Next Feature Increment"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_TrueSessionPhase_NextFeature_Increment_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM True Session Phase Next Feature Increment

Implemented true UTC intraday/session phase.

- New base columns: `34=session_phase_sin`, `35=session_phase_cos`; current `n_in=40`.
- Historical projections remain explicit: `n_in=36` → base `0..31`; `n_in=38` → base `0..33`; `n_in=40` → base `0..35`.
- Historical timestamp channels remain unchanged.
- Recommendation semantic identity advanced from v6 to v7; persisted v3–v6 reconstruction remains version-specific.

Full report: [LSTM_TrueSessionPhase_NextFeature_Increment_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_TrueSessionPhase_NextFeature_Increment_Output.md)

Validation passed:

- `LSTMTrueSessionPhaseTests`
- `LSTMFeatureVectorParityTests`
- `LSTMStatefulFeatureWarmupTests`
- `Donchian20FeatureIncrementTests`
- `ExperimentRecommendationTests`
- All three requested isolated DB-backed provenance tests against `ea_warmup_provenance_test`
- Release build: `** BUILD SUCCEEDED **`

No scheduler state, production data, or commits were changed.

`git status --short`:

```text
 M Headers/FeatureLayout.hpp
 M Headers/ModelInputContract.hpp
 M LSTM/LSTM.cpp
 M LSTM/Tensor.cpp
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
 M Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
 M Sources/ExperimentRecommendationCampaignStatusRepository.cpp
 M Sources/ExperimentRecommendationConversionExecutionRepository.cpp
 M Sources/ExperimentRecommendationConversionRepository.cpp
 M Tests/Donchian20FeatureIncrementTests.cpp
 M Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
 M Tests/ExperimentRecommendationTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMStatefulFeatureWarmupTests.cpp
?? Headers/SessionPhaseFeatures.hpp
?? LSTM_TrueSessionPhase_NextFeature_Increment_Output.md
?? Tests/LSTMTrueSessionPhaseTests.cpp
?? Tests/LSTMTrueSessionPhaseTests.sh
```

Ready for final assurance.