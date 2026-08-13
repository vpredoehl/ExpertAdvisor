---
title: "LSTM Legacy Model Feature Width Compatibility Scoped Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_LegacyModel_FeatureWidthCompatibility_ScopedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Legacy Model Feature Width Compatibility Scoped Correction

# LSTM Legacy/Pre-Donchian Model Feature-Width Compatibility — Scoped Correction

## Root Cause

`FeatureLayout.hpp` establishes:

- Legacy base width: 32
- Current base width: 34 (`32 + 2` Donchian columns)
- Four appended return features
- Legacy effective width: 36
- Current effective width: 38

`PgModelIO::saveModelMeta()` derives `n_in` from the persisted parameter matrix: `param.rows - hidden_size`. Therefore `model_meta.n_in` and the parameter shape are structural authority.

The runtime previously computed `34 + 4 = 38` unconditionally and rejected persisted 36-wide models.

## Design

Added [`ModelInputContract.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputContract.hpp>):

- `n_in=36` → copy 32 tensor columns + 4 returns
- `n_in=38` → copy 34 tensor columns + 4 returns
- Other widths → deterministic fail-closed diagnostic

The projection is structural; Donchian columns are not copied into legacy model inputs.

`PgModelIO` now validates `model_meta` against the persisted `param` matrix and rejects parameter-shape mismatches.

## Files Changed

- [`Headers/ModelInputContract.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/ModelInputContract.hpp>) — centralized width contracts and projection.
- [`Headers/LSTM.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/LSTM.hpp>) — optional persisted input width and width accessor.
- [`Headers/PgModelIO.hpp`](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp>) — structural `model_meta`/parameter validation and load-shape checks.
- [`LSTM/LSTM.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp>) — contract-aware constructor and all model-input assembly paths.
- [`LSTM/main.cpp`](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp>) — resume, direct inference, scheduler inference, and infer-all compatibility handling.
- [`Tests/LSTMModelInputCompatibilityTests.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMModelInputCompatibilityTests.cpp>) and its shell runner — focused regressions.
- [`Tests/DonchianTensorIntegrationTests.cpp`](</Volumes/Developer SSD/ExpertAdvisor/Tests/DonchianTensorIntegrationTests.cpp>) — standalone test linker stub only.

No schema migration was introduced.

`Sources/ExperimentScheduler.cpp` and the other recovery artifacts shown by `git status` were pre-existing changes and were not modified for this correction.

## Compatibility Semantics

- `n_in=36`: physical 34-column tensor is projected to its first 32 columns plus four returns.
- `n_in=38`: all 34 tensor columns, including Donchian columns, plus four returns are used.
- Unsupported widths: fail closed with `MODEL_INPUT_WIDTH_UNSUPPORTED`.

Historical parameter matrices and `model_meta` values are never rewritten.

## Resume Path

Resume metadata now validates `model_meta` against the persisted parameter matrix. The LSTM is constructed with the persisted width before loading parameters, allowing valid 36-wide checkpoints to resume without the former 36-vs-38 mismatch.

All existing hidden-size, symbol, horizon, threshold, date-range, target, optimizer, and other compatibility checks remain intact.

## Inference Path

The same contract applies to:

- Direct `--infer`
- Scheduler inference
- Checkpoint inference
- `--infer-all`

Infer-all accepts legacy width only when the persisted model structure is supported. Unsupported widths are skipped with a deterministic incompatibility diagnostic.

## LSTM Input Assembly

Audited and corrected:

- `CalculateBatch`
- `PredictNextDirectionProbs`
- `PredictNextReturn`
- `PredictNextRelativeMove`
- Prebuilt batch-row copying

Baseline and diagnostic builders remain current-runtime diagnostic paths and do not bind persisted legacy model weights.

## Regression Tests

- Legacy contract: 34 physical columns → 32 copied tensor columns + 4 returns.
- Current contract: all 34 tensor columns copied.
- Unsupported width: `n_in=37` rejected deterministically.
- Parameter shape remains unchanged.
- Existing Donchian causal-feature test passed.
- Existing Donchian tensor integration test passed.

## Validation

Passed:

```text
Tests/LSTMModelInputCompatibilityTests.sh
DonchianFeatureTests
DonchianTensorIntegrationTests
git diff --check
```

Release build passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor-legacy-width \
  build
```

Result: `BUILD SUCCEEDED`

## Live-System Safety

A live scheduler and training worker were observed. No workers were launched, stopped, or signaled. No live database rows were changed, and no production experiment was used for testing. The build used isolated DerivedData.

## Diff Hygiene

`git diff --check`: passed.

Current status:

```text
 M Headers/LSTM.hpp
 M Headers/PgModelIO.hpp
 M LSTM/LSTM.cpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/DonchianTensorIntegrationTests.cpp
?? Headers/ModelInputContract.hpp
?? Tests/LSTMModelInputCompatibilityTests.cpp
?? Tests/LSTMModelInputCompatibilityTests.sh
```

Tracked diff stat:

```text
6 files changed, 295 insertions(+), 147 deletions(-)
```

## Residual Risks

End-to-end execution against model IDs 1562 and 1601 was deliberately not run because live scheduler/workers are active.

## Final Disposition

The scoped compatibility correction is complete and ready for independent reverification.