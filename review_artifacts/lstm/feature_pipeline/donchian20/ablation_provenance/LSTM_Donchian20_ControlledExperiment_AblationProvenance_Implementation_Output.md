---
title: "LSTM Donchian-20 Controlled Experiment Ablation and Provenance Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Donchian20_ControlledExperiment_AblationProvenance_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Donchian-20 Controlled Experiment Ablation and Provenance Implementation

## Disposition

**PASS WITH FINDINGS**

Implemented and left uncommitted on `lstm-feature-development`.

1. **Files changed**

Added:

- `Headers/Donchian20Mode.hpp`
- `Database/migrations/060_donchian20_mode.sql`
- `docs/LSTMDonchian20Mode.rst`

Modified:

- `Headers/Tensor.hpp`
- `Headers/PgModelIO.hpp`
- `LSTM/Tensor.cpp`
- `LSTM/main.cpp`
- `Sources/ExperimentScheduler.cpp`
- Recommendation/campaign repository and canonicalization sources
- `Tests/DonchianFeatureTests.cpp`
- `Tests/DonchianTensorIntegrationTests.cpp`

2. **Mode values**

- `Donchian20Mode::Enabled` → `enabled`
- `Donchian20Mode::ZeroAblation` → `zero_ablation`

3. **Default**

`enabled` for new experiments, scheduler defaults, and legacy rows/models without mode metadata.

4. **CLI**

```text
--donchian20-mode=enabled
--donchian20-mode=zero_ablation
```

Both separated-value and `=` forms are accepted.

5. **Experiment persistence**

`experiment.donchian20_mode`, constrained to the two supported values. Mode is included in experiment uniqueness identity.

6. **Scheduler propagation**

Persisted mode is loaded on restart and passed to:

- Training
- Resume training
- Final inference
- Checkpoint inference

7. **Model provenance**

Every checkpoint and final model stores `donchian20_mode_meta` in the existing `matrix` metadata path.

8. **Compatibility enforcement**

Explicit runtime, scheduler, resume, and persisted-model modes are compared. Mismatches fail clearly. Infer-all rejects candidates with incompatible mode metadata.

9. **Feature parity**

Enabled writes causal Donchian values to columns 32/33. Zero-ablation writes exactly `0.0f`. Both retain 34 base columns and 38 effective model inputs.

10. **Training/inference parity**

The mode is selected before Tensor construction and is used consistently by training, checkpoint inference, final inference, resume, and infer-all compatibility checks.

11. **Resume behavior**

Resume restores the mode from the source model and rejects explicit mismatches.

12. **Campaign Operations**

Recommendation canonicalization was advanced to v4 and carries the mode. Conversion, campaign materialization, and campaign status paths preserve it. Legacy v3 invocations default to `enabled`.

Campaign-created paired arms can therefore inherit explicit source modes; arbitrary mode selection at a higher campaign UI/schema layer remains a possible small follow-up.

13. **Paired initialization**

Existing initialization uses a fixed thread-local seed (`42`) and does not branch on Donchian mode. Since width and geometry remain unchanged, otherwise identical fresh arms receive identical initialization behavior.

14. **Tests**

Passed:

```text
clang++ ... Tests/DonchianFeatureTests.cpp && /tmp/DonchianFeatureTests
```

Passed Tensor integration syntax check:

```text
clang++ -std=c++20 -fsyntax-only ... Tests/DonchianTensorIntegrationTests.cpp LSTM/Tensor.cpp
```

Invalid CLI value correctly rejected.

15. **Release build**

Passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build
```

Existing compiler warnings remain, primarily libpqxx deprecation warnings.

16. **Diff check**

Passed:

```text
git diff --check
```

17. **Migration**

Added migration `060_donchian20_mode.sql`. Migration script syntax and checksum generation were verified. No production database was modified; no live migration integration test was run.

18. **Residual findings**

- Run migration/schema integration tests against a disposable database before campaign launch.
- Legacy models without mode metadata are interpreted as `enabled`.
- Standalone legacy diagnostic commands are not model-lineage inference paths and retain their existing enabled default.

19. **Scope**

No Donchian formula, lookback, feature width, architecture, optimizer, labeling, scheduler capacity, continuation policy, or historical weights were changed. No commit, merge, push, or branch switch was performed.

`git status --short`:

```text
 M Headers/PgModelIO.hpp
 M Headers/Tensor.hpp
 M LSTM/Tensor.cpp
 M LSTM/main.cpp
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
 M Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
 M Sources/ExperimentRecommendationCampaignStatusRepository.cpp
 M Sources/ExperimentRecommendationConversionExecutionRepository.cpp
 M Sources/ExperimentRecommendationRepository.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/DonchianFeatureTests.cpp
 M Tests/DonchianTensorIntegrationTests.cpp
?? Database/migrations/060_donchian20_mode.sql
?? Headers/Donchian20Mode.hpp
?? docs/LSTMDonchian20Mode.rst
```

`git diff --stat`: 14 tracked files changed, 337 insertions, 62 deletions; three new files are untracked as shown above.