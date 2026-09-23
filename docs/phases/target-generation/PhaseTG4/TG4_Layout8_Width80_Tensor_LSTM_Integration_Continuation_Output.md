---
title: "TG4 Layout 8 Width 80 Tensor LSTM Integration Continuation"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Layout8_Width80_Tensor_LSTM_Integration_Continuation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Layout 8 Width 80 Tensor LSTM Integration Continuation

Implemented and validated the dirty continuation work without resetting, committing, publishing, or touching production data.

Starting/current HEAD: `b3e07fd2ca96203bb1c1219b36181d18d120836b`.

Layout 8 contract is now:

- Tensor columns `0..72`: unchanged layout-7 semantic prefix.
- `73`: `tg4_inner_break_any`
- `74`: `tg4_source_tg3_structurally_eligible`
- `75`: `tg4_source_tg3_confluent`
- Model return suffix `76..79`: 1/4/8/16-bar scaled log returns.
- Current identity: width `80`, layout `8`.
- Historical layout 7 remains width `77`, projects Tensor `0..72`, then writes returns at `73..76`; TG4 cannot alias that suffix.

Production path is `canonical Feature bars -> ModelInputPreparation -> Tensor::Add -> owned TG4 ProductionStreamingAdapter -> Tensor columns 73..75 -> model-input projection + returns`. Tensor calls the adapter once per accepted bar, requires exact pulse/bar timestamp equality, validates `confluent <= eligible <= inner_break_any`, and writes exact `0.0f`/`1.0f` bits in both first-row and normal-row branches. TG4 remains UTL-only; DTL structures remain eligible only for `[1,0,0]`.

Expansion is semantic, not a physical 77-row prefix copy:

- Old Tensor rows `0..72` retain weights.
- New TG4 rows `73..75` zero-initialize.
- Old return rows `73..76` move to `76..79`.
- Recurrent rows move after the complete expanded input block.
- Biases and heads remain unchanged.

`LSTMInputWidthExpansionTests` has sentinel proof for all of the above. The new integration test also has distinct TG4/return sentinels proving historical layout-7 projection cannot copy or alias TG4 data into its return suffix.

Feature-ablation audit: masks remain name-enumerated and retain their historical meanings. No TG4 ablation syntax or broad prefix rule was added; an explicit regression confirms existing economic-event masking leaves all TG4 columns unchanged. A TG4 ablation remains a separate follow-on design.

Migration 089 already accepts arbitrary positive width/layout pairs and preserves immutable identity, so no migration was added or executed against production.

Files changed:

```text
Headers/FeatureLayout.hpp
Headers/ModelInputContract.hpp
Headers/ModelInputExpansion.hpp
Headers/ModelInputFeatureSemantics.hpp
Headers/Tensor.hpp
LSTM/Tensor.cpp
Sources/FeatureAblationReplicationEvaluation.cpp
Tests/CausalSurpriseObservabilityRepositoryTests.cpp
Tests/CausalSurpriseObservabilityTests.cpp
Tests/DonchianTensorIntegrationTests.cpp
Tests/EconomicEventTensorIntegrationTests.cpp
Tests/LSTMCausalCloseLocationTests.cpp
Tests/LSTMCausalDirectionalAdverseExcursionTests.cpp
Tests/LSTMCausalDirectionalEfficiencyTests.cpp
Tests/LSTMCausalDirectionalRangeTests.cpp
Tests/LSTMCausalHistoricalLevelProximityTests.cpp
Tests/LSTMCausalMultiBarRangePressureTests.cpp
Tests/LSTMCausalReturnAutocorrelationTests.cpp
Tests/LSTMCausalReturnDirectionImbalanceTests.cpp
Tests/LSTMCausalReturnSignPersistenceTests.cpp
Tests/LSTMCausalReturnSurpriseTests.cpp
Tests/LSTMCausalRollingRangeExpansionTests.cpp
Tests/LSTMCausalVolatilityRegimeTests.cpp
Tests/LSTMFeatureVectorParityTests.cpp
Tests/LSTMInputWidthExpansionPersistenceTests.cpp
Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh
Tests/LSTMInputWidthExpansionTests.cpp
Tests/LSTMModelInputCompatibilityTests.cpp
Tests/LSTMModelInputIdentityMigrationTests.sql
Tests/LSTMRelativeTickVolumeTests.cpp
Tests/LSTMTrueSessionPhaseTests.cpp
Tests/SchedulerSemanticAdmissionTests.cpp
Tests/TG4TensorModelInputIntegrationTests.cpp
Tests/TG4TensorModelInputIntegrationTests.sh
docs/phases/target-generation/PhaseTG4/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md
docs/phases/target-generation/PhaseTG4/TG4_TENSOR_LSTM_LAYOUT8_INTEGRATION.md
```

Passed validations included:

- TG4 adapter and canonical replay tests.
- New TG4 Tensor/model-input integration test.
- Feature-vector parity, model-input compatibility, and width-expansion tests.
- Width-expansion persistence against a disposable cloned DB.
- Model-input identity migration against a disposable cloned DB.
- Causal-surprise observability unit/repository tests.
- Scheduler semantic-admission and semantic-worker registry tests.
- `git diff --check`.

Release build result: the ordinary non-publishing Release build was attempted using stable `DerivedData/ExpertAdvisor`, but stopped at the expected dirty-tree provenance guard before compiling the `LSTM Release` target:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

No canonical or semantic worker was published. The scheduler integration harness could not run against a matching binary: its existing product failed the embedded-source-commit check. After a clean commit, run:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
bash Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
```

The live scheduler remained untouched (PID `68183`, inference-only); no experiment was started and production LSTM was not mutated. Semantic-worker registry/current symlink timestamps were unchanged, no `layout8` worker directory exists, and immutable layout-7 train/reference and inference artifacts remain retained.

Stale-current-semantics audit: production defaults and current assertions now resolve through `80/8/76`. Remaining `77/7` references are historical layout fixtures, layout-6/7 migration coverage, retained worker tests/artifacts, causal-surprise historical studies, or phase-history documents. The old TG4 design document was explicitly marked as superseded rather than left claiming 77/7 is current.

Final `git status --short`: 33 modified tracked files and 3 untracked new files, as listed above. `git diff --stat`: 33 tracked files, `282 insertions`, `151 deletions` (untracked files excluded by Git’s stat).

Remaining prerequisite before the first controlled TG4 ablation: a separately approved TG4 mask/provenance experiment design, plus a clean committed Release build and explicit layout-8 semantic-worker publication.

Recommended commit message:

```text
Integrate TG4 pulses into Tensor layout 8
```