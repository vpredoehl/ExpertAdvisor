---
title: "LSTM Append-Only Input-Width Expansion Continuation Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_AppendOnlyInputWidthExpansion_Continuation_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Append-Only Input-Width Expansion Continuation Implementation

This document records the explicit append-only expansion mechanism implemented
on branch `lstm-feature-development`. Production migration and experiment
cutover remain separate operator actions; this implementation does not perform
either action and is not pushed by the final correction task.

## Previous behavior and investigation

- Authoritative width metadata is `model_meta` (`1x3`: schema, `n_in`, hidden size), cross-checked against the persisted `param` shape.
- Current Tensor width is 48; four multi-horizon return features make the current model width 52.
- Historical model widths are `36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52`, corresponding to Tensor prefixes `32, 34, 36, 37…48`.
- LSTM parameters use one fused row-major matrix:
  - Shape: `(n_in + H) x 4H`
  - Rows: Tensor inputs, four return inputs, then recurrent hidden state
  - Gate columns: `[input | forget | candidate | output]`
- Other parameters are:
  - Bias: `1 x 4H`
  - Regression head: `H x 1`, `1 x 1`
  - Direction head: `H x 3`, `1 x 3`
- SGD persists only its update count. There are no moment tensors to widen.
- Previously, ordinary resume constructed the model at the historical width and projected current Tensor rows to that prefix. It did not expand the model.

## Implemented behavior

The new opt-in is:

```text
--resume-model-id=MODEL_ID --resume-expand-input-width
```

Ordinary `--resume-model-id` remains unchanged.

Expansion from `W_old` to 52:

1. Copies historical Tensor rows `[0, T_old)` exactly.
2. Inserts zero input-weight rows `[T_old, 48)`.
3. Relocates the four return-feature rows from `[T_old, T_old+4)` to `[48, 52)`.
4. Relocates all recurrent rows from `[W_old, W_old+H)` to `[52, 52+H)`.
5. Loads biases and both heads unchanged.

Relocating the return-feature suffix is essential: merely appending rows would reinterpret historical return weights as newly added Tensor features.

Zero initialization is safe because production BPTT computes input gradients as `[x|h]^T × d_gates`; the gradient does not depend on the existing weight value. The production training test confirmed that a zero-initialized new input weight becomes nonzero after one deterministic update.

## Safety and provenance

Expansion rejects:

- Wider-than-current, current-width initial expansion, or unknown widths.
- Non-current targets or layouts not represented by the append-only registry.
- Incompatible semantic-layout metadata.
- Missing or malformed parameter, bias, target, or head tensors.
- Hidden-size or fused-matrix shape mismatches.
- Noncanonical or inconsistent expansion provenance.

Historical models without the new semantic marker remain eligible only through the existing known-width prefix registry. New models persist `model_input_semantics_meta`.

Expanded descendants persist:

- Existing `model.parent_model_id` and `model.experiment_id` lineage.
- Original expansion source model ID and width.
- Expanded width.
- New Tensor column interval and semantic names.
- Zero initialization policy.
- Semantic-layout version.
- Current-width `model_meta`.
- Existing epoch and optimizer update accounting.

A current-width scheduler retry is accepted only if it belongs to the same scheduler experiment, contains valid expansion provenance, and its parent chain reaches the original expansion source.

Expansion provenance is immutable event-time evidence. Its
`semantic_layout` value is accepted when it is a registered ancestor of the
current append-only generation and its registered width matches that event's
expanded width; it is not required to equal the currently compiled generation.
Historical provenance is validated whenever a marker-bearing model is used as
a later expansion source or retry checkpoint. The persisted model/source
widths and parent ancestry must agree with the marker. A later re-expansion
creates new provenance for the new event while leaving its source model's prior
marker unchanged.

Migration `071` adds `experiment.resume_expand_input_width`, includes it in experiment identity, and requires a resume source. Recommendation evaluation schema/idempotency was not changed. Expanded continuations are explicitly excluded as recommendation sources so they are not conflated with fresh experiments.

Feature ablation retains current width. Source ablations cannot be removed; newly appended features may be additionally ablated and receive the standard zero value.

## Files changed

Core:

- [ModelInputExpansion.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputExpansion.hpp>)
- [PgModelIO.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp>)
- [main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [ExperimentRecommendationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationRepository.cpp>)

Schema/documentation:

- [071_resume_input_width_expansion.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/071_resume_input_width_expansion.sql>)
- [LSTMInputWidthExpansion.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/LSTMInputWidthExpansion.rst>)

Tests:

- New unit, persistence, scheduler, and migration tests under `Tests/LSTMInputWidthExpansion*` and `Tests/InputWidthExpansionMigrationTests.sql`.
- Updated recommendation repository test.
- Updated seven disposable scheduler integration harnesses to apply migration 071.

## Verification

Passed:

- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/LSTMInputWidthExpansionPersistenceTests.sh`
  - Real PostgreSQL save/load.
  - Real production forward/BPTT/SGD path.
  - Prediction preservation with nonzero current-only features.
  - New-weight trainability.
  - Checkpoint/provenance/optimizer round-trip.
  - Source immutability.
- `bash Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh .../LSTM_Debug`
- `bash Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh .../LSTM_Debug`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- `bash Tests/LSTMFeatureVectorParityTests.sh`
- All 13 relevant causal feature tests, using the isolated Debug library path.
- Recommendation repository integration test against a disposable database.
- Migration contract test.
- Shell syntax checks.
- `git diff --check`
- Isolated Debug build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath DerivedData/ExpertAdvisorExpansion build
```

Result: `BUILD SUCCEEDED`.

The prescribed Release build was attempted without bypassing provenance:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor build
```

Result: exit 65, correctly stopped by:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

Remaining runtime verification is a clean-tree Release build after commit and a later explicitly selected historical-model run. Migration 071 has not been applied to production, and no production experiment was queued.

## Example queue command

After migration 071 and a clean Release build:

```bash
DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  --queue-experiment \
  --resume-model-id=MODEL_ID \
  --resume-expand-input-width \
  --target-epochs=240
```

Optionally ablate a newly introduced feature without shrinking:

```bash
  --ablate-features=historical_level_proximity
```

The pre-commit status and diff statistics in the original implementation run
were superseded by the final provenance correction and its focused regression
coverage. The final task report and Git commit are authoritative for the
delivered file set.
