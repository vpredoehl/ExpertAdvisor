---
title: "LSTM Causal Rolling Return Autocorrelation Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalRollingReturnAutocorrelation_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Causal Rolling Return Autocorrelation Implementation

Implemented the causal rolling lag-1 return-autocorrelation increment.

### Behavior

- Appended Tensor column `48`: `returnAutocorrelationCol`.
- Tensor width: `48 → 49`.
- Effective model input width: `52 → 53` (`49` Tensor + `4` return suffix).
- Window: `N = 32` adjacent return pairs using 33 consecutive valid returns.
- At completed row `t`:
  - returns are `r_j = log(close_j / close_(j-1))`;
  - retained window is `r_(t-32)…r_t`;
  - Pearson correlation compares `r_(t-32)…r_(t-1)` with `r_(t-31)…r_t`.
- The current return is legal because `Tensor::Add` receives a completed candlestick; training and inference consume that same Tensor row.
- Uses two-pass double-precision, mean-centered accumulation.
- Outputs exactly `0` for insufficient history, invalid continuity, or variance sums `<= 1e-24`.
- Invalid returns clear the pair window to preserve true bar adjacency.
- Final output is finite, unscaled, and clamped to `[-1,1]`.

This differs from sign persistence/counts, direction imbalance, path efficiency, RMS volatility regime, and return surprise because it measures ordered, mean-centered linear dependence between adjacent signed return magnitudes.

### Compatibility and provenance

- Historical width `52` remains the exact 48-column historical-level-proximity prefix.
- Registered widths are now `36, 38, 40, 41…53`.
- Ablation token: `return_autocorrelation`; it zeros only column 48 without changing width.
- Recommendation semantic version: `v17 → v18`; historical versions remain reconstructible.
- Input semantic layout:
  - Generation 1 remains fixed at width 52.
  - Generation 2 appends width 53 with generation 1 as predecessor.
- Generic explicit `52 → 53` expansion:
  - preserves old Tensor weights;
  - zero-initializes the autocorrelation input row;
  - relocates the four return rows and recurrent rows unchanged;
  - leaves the new row trainable.
- No recommendation-evaluation schema/idempotency or scheduler lifecycle behavior changed.

### Files

Core changes include:

- [CausalReturnAutocorrelationFeatures.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/CausalReturnAutocorrelationFeatures.hpp>)
- [FeatureLayout.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureLayout.hpp>)
- [ModelInputContract.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputContract.hpp>)
- [ModelInputExpansion.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputExpansion.hpp>)
- [FeatureAblation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureAblation.hpp>)
- [Tensor.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp>)
- [Tensor.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp>)
- Recommendation semantic/provenance sources and affected tests.
- [LSTMCausalReturnAutocorrelationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMCausalReturnAutocorrelationTests.cpp>)
- [Volume_II_Data_Pipeline.md](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_II_Data_Pipeline.md>)
- [LSTMInputWidthExpansion.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/LSTMInputWidthExpansion.rst>)

### Verification

Passed:

- New focused autocorrelation suite covering all requested numerical, causality, boundary, parity, integration, ablation, compatibility, and expansion cases.
- Model-input compatibility.
- Generic input-width expansion.
- Real PostgreSQL expansion persistence and production BPTT trainability.
- Feature-vector train/inference parity.
- Thirteen existing causal/session feature regression executables.
- Recommendation semantic identity and candidate generation.
- Recommendation campaign materialization repository integration.
- Recommendation outcome-assessment provenance parsing.
- `git diff --check`.
- Full `LSTM Debug` build: `BUILD SUCCEEDED`.

The checked-in older causal shell wrappers reference the pre-existing obsolete `DerivedData/ExpertAdvisor-next-feature` directory; their equivalent strict commands passed using the current library location.

The prescribed Release build stopped with exit 65 at the expected clean-tree provenance guard:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

A canonical Release build remains for the post-commit clean-tree workflow. Empirical predictive value on real datasets remains unmeasured. No production experiments were queued.

### Repository state

```text
 M Headers/FeatureAblation.hpp
 M Headers/FeatureLayout.hpp
 M Headers/ModelInputContract.hpp
 M Headers/ModelInputExpansion.hpp
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
 M Tests/LSTMCausalHistoricalLevelProximityTests.cpp
 M Tests/LSTMCausalMultiBarRangePressureTests.cpp
 M Tests/LSTMCausalReturnDirectionImbalanceTests.cpp
 M Tests/LSTMCausalReturnSignPersistenceTests.cpp
 M Tests/LSTMCausalReturnSurpriseTests.cpp
 M Tests/LSTMCausalRollingRangeExpansionTests.cpp
 M Tests/LSTMCausalVolatilityRegimeTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMInputWidthExpansionPersistenceTests.cpp
 M Tests/LSTMInputWidthExpansionTests.cpp
 M Tests/LSTMModelInputCompatibilityTests.cpp
 M Tests/LSTMRelativeTickVolumeTests.cpp
 M Tests/LSTMTrueSessionPhaseTests.cpp
 M docs/LSTMInputWidthExpansion.rst
 M docs/architecture/Volume_II_Data_Pipeline.md
?? Headers/CausalReturnAutocorrelationFeatures.hpp
?? Tests/LSTMCausalReturnAutocorrelationTests.cpp
?? Tests/LSTMCausalReturnAutocorrelationTests.sh
```

`git diff --stat` for tracked files:

```text
33 files changed, 256 insertions(+), 144 deletions(-)
```

The three new untracked files are not included in Git’s tracked diff statistic.