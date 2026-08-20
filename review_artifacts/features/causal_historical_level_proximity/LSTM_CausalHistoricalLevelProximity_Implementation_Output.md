---
title: "LSTM Causal Historical-Level Proximity Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalHistoricalLevelProximity_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Causal Historical-Level Proximity Implementation

Implemented the causal historical-level proximity feature without touching scheduler lifecycle or recommendation-evaluation schema behavior.

### Feature contract

- Aggregates each symbol independently into UTC Monday-based weekly bars.
- Discovers strict five-week swing highs/lows only after two confirming weeks complete.
- Requires 26 consecutive weeks for volatility bandwidth and 104 completed weeks before emitting a value.
- Candidate bandwidth:
  \(h_i=\max(10^{-6},0.25\times median_{26}(\log(H/L)))\)
- Retains candidates for 260 weeks, tapering over the final 26.
- Repeated nearby pivots smoothly corroborate each candidate.
- Current proximity:
  \(1-\exp(-S_t/2)\), bounded to `[0,1)`.
- Invalid/insufficient history returns `0`.
- Small price changes produce smooth Gaussian changes; no hard-coded or round-number levels are used.

Candidate discovery uses only completed weeks preceding the observation. The current close participates solely in the proximity transform, so future mutations cannot affect an earlier row.

### Layout, provenance, and compatibility

- New column: `47 = historicalLevelProximityCol`
- `feature_size`: `47 → 48`
- Current `n_in`: `51 → 52`
- Historical `n_in=51` remains an exact 47-column prefix projection.
- Recommendation semantic version: `v16 → v17`
- Explicit v3–v16 reconstruction remains supported.
- Ablation token: `historical_level_proximity`
- Ablation preserves width and zeros only column 47.

### Files changed

Core implementation:

- [CausalHistoricalLevelProximityFeatures.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/CausalHistoricalLevelProximityFeatures.hpp>)
- [FeatureLayout.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureLayout.hpp>)
- [ModelInputContract.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputContract.hpp>)
- [FeatureAblation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureAblation.hpp>)
- [Tensor.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp>)
- [Tensor.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp>)

Provenance/version reconstruction:

- `Sources/ExperimentRecommendation.{hpp,cpp}`
- `Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp`
- `Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp`

Tests/documentation:

- [LSTMCausalHistoricalLevelProximityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMCausalHistoricalLevelProximityTests.cpp>)
- [Volume_II_Data_Pipeline.md](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_II_Data_Pipeline.md>)
- Existing feature, compatibility, parity, and recommendation tests updated for the appended width/version.

### Verification

Passed:

- New focused historical-level tests
- Model-input compatibility tests
- Feature-vector train/inference parity tests
- Donchian feature and Tensor integration tests
- All existing causal appended-feature tests
- Recommendation semantic and candidate-generator tests
- Strict compilation of all affected application translation units with `-Werror`, suppressing only documented pre-existing warning classes
- `git diff --check`, including separate checks for untracked new files

The prescribed Release build returned exit 65 solely because `GenerateBuildProvenance.py` requires a clean source tree. The guard was not bypassed. No production experiments were queued.

Remaining risks: a fully linked Release build remains unverified until the patch is committed/clean, and real-dataset runtime/empirical utility has not been benchmarked. Complexity is bounded to retained five-year candidates: linear per observation and quadratic only during weekly corroboration refresh.

### `git status --short`

```text
 M Headers/FeatureAblation.hpp
 M Headers/FeatureLayout.hpp
 M Headers/ModelInputContract.hpp
 M Headers/Tensor.hpp
 M LSTM/Tensor.cpp
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
 M Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
 M Tests/DonchianTensorIntegrationTests.cpp
 M Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
 M Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
 M Tests/ExperimentRecommendationTests.cpp
 M Tests/LSTMCausalCloseLocationTests.cpp
 M Tests/LSTMCausalDirectionalAdverseExcursionTests.cpp
 M Tests/LSTMCausalDirectionalEfficiencyTests.cpp
 M Tests/LSTMCausalDirectionalRangeTests.cpp
 M Tests/LSTMCausalMultiBarRangePressureTests.cpp
 M Tests/LSTMCausalReturnDirectionImbalanceTests.cpp
 M Tests/LSTMCausalReturnSignPersistenceTests.cpp
 M Tests/LSTMCausalReturnSurpriseTests.cpp
 M Tests/LSTMCausalRollingRangeExpansionTests.cpp
 M Tests/LSTMCausalVolatilityRegimeTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMModelInputCompatibilityTests.cpp
 M Tests/LSTMRelativeTickVolumeTests.cpp
 M Tests/LSTMTrueSessionPhaseTests.cpp
 M docs/architecture/Volume_II_Data_Pipeline.md
?? Headers/CausalHistoricalLevelProximityFeatures.hpp
?? Tests/LSTMCausalHistoricalLevelProximityTests.cpp
?? Tests/LSTMCausalHistoricalLevelProximityTests.sh
```

### `git diff --stat`

```text
28 tracked files changed, 174 insertions(+), 70 deletions(-)
```

The three untracked new files add 539 lines and are not included in Git’s default diff stat.