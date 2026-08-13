# LSTM Legacy/Pre-Donchian Model Feature-Width Compatibility — Independent Reverification

## Disposition

**PASS WITH ONE NON-BLOCKING VALIDATION LIMITATION**

The scoped correction is structurally correct and fixes the observed `MODEL_CONFIG_MISMATCH,field=feature_count,model=36,runtime=38` failure at the correct boundary: the persisted model input width governs how the current physical tensor is projected into the model.

No blocking source defect was found.

The remaining limitation is validation depth: the supplied package does not include an end-to-end run of the actual affected persisted model IDs 1562 and 1601 through the corrected executable. The focused regression validates the model-input contract and projection helper rather than exercising a real PostgreSQL-backed 36-wide model through the complete resume/inference path.

## Basis reviewed

Packaged state:

- branch: `lstm-feature-development`
- HEAD: `61af0dc620e0eddee268df928d0f7579a758f280`
- legacy tensor width: 32
- current tensor width: 34
- appended return features: 4
- legacy persisted effective width: 36
- current persisted effective width: 38

Reported validation evidence:

- `Tests/LSTMModelInputCompatibilityTests.sh`: PASS
- `DonchianFeatureTests`: PASS
- `DonchianTensorIntegrationTests`: PASS
- `git diff --check`: PASS
- Release Xcode build: `BUILD SUCCEEDED`

## 1. Root cause

**PASS**

The original runtime always built the current effective input width as `34 + 4 = 38`. Historical/pre-Donchian models were trained and persisted with `32 + 4 = 36`. Rejecting a persisted 36-wide model solely because the current physical tensor is now 34 columns was therefore an inappropriate compatibility check.

The correction correctly treats persisted model geometry as authoritative when loading an existing model.

## 2. Explicit model-input contracts

**PASS**

`Headers/ModelInputContract.hpp` recognizes exactly two persisted contracts:

- `n_in=36` -> 32 tensor columns + 4 return features
- `n_in=38` -> 34 tensor columns + 4 return features

Other widths fail closed with `MODEL_INPUT_WIDTH_UNSUPPORTED`.

This is preferable to generic truncation because only the two known historical layouts are accepted.

## 3. Legacy projection semantics

**PASS**

For a 36-wide model, `CopyTensorFeaturesForModelInput()` copies exactly the first 32 columns of the current 34-column tensor. Donchian columns 32 and 33 are therefore excluded, and the four return features are appended beginning at destination column 32.

That preserves the exact historical learned layout.

This is materially important: copying 34 columns and merely zeroing the two Donchian values would still shift the four return features by two positions and would be incompatible with the old weight matrix. The implementation does not make that mistake.

For a 38-wide model, all 34 tensor columns are copied and the return features occupy columns 34-37.

## 4. LSTM construction and parameter loading

**PASS**

The LSTM constructor now accepts an optional persisted model input width and resolves `n_in` before allocating width-dependent matrices.

That means a legacy model is instantiated as `n_in=36` before its parameters are loaded.

`PgModelIO::loadAll()` then verifies that the loaded `param` matrix implies the same input width as `lstm.InputFeatureCount()`. This prevents silently loading a 36-wide gate matrix into an object whose internal model width is still 38.

## 5. Persisted metadata / parameter-shape authority

**PASS**

`PgModelIO::loadRequiredModelMeta()` cross-validates `model_meta` against the persisted gate matrix.

It checks:

- `model_meta` shape is `1x3`;
- schema and dimensions are valid finite positive integers;
- persisted input width is supported;
- `param` has valid LSTM gate geometry;
- hidden size derived from `param` matches metadata;
- input width derived from `param` matches metadata.

Thus `model_meta.n_in=36` is not trusted on its own; the persisted parameter shape must independently prove the same model geometry.

Historical model matrices and metadata are not rewritten.

## 6. Resume path

**PASS BY SOURCE INSPECTION**

`LoadResumeCheckpointConfig()` now resolves the persisted width through `loadRequiredModelMeta()`. The main runtime passes that width into `CreateLstmForRuntimeLogLevel()` before `PgModelIO::loadAll()`.

This directly removes the former 36-vs-38 rejection for a valid pre-Donchian checkpoint and is the correct architecture for the failed resume from model 1562.

## 7. Direct/scheduler inference path

**PASS BY SOURCE INSPECTION**

`LoadPersistedInferenceConfig()` likewise resolves the persisted width before LSTM construction. A scheduler-launched `--infer --model=<legacy-model>` can therefore create an `n_in=36` LSTM, load its 36-wide persisted parameters, and project the current tensor to the old 32-column base layout.

This directly addresses the failure observed for model 1601.

Checkpoint inference uses the same persisted-model compatibility path.

## 8. Infer-all

**PASS**

Infer-all now carries a candidate-specific `modelInputWidth`.

With `model_meta` present, metadata and parameter shape are cross-validated. For older candidates without `model_meta`, the implementation derives `n_in` from the gate-matrix shape and still requires it to resolve to one of the supported contracts.

The LSTM is then constructed using that candidate width before parameter loading.

## 9. Model input assembly

**PASS BY SOURCE INSPECTION**

The compatibility contract is propagated through the relevant LSTM assembly paths, including:

- `CalculateBatch`
- `PredictNextDirectionProbs`
- `PredictNextReturn`
- `PredictNextRelativeMove`

Each path distinguishes:

- physical current tensor width;
- tensor width required by the persisted model contract;
- total persisted model input width.

The four return features are appended at the persisted model's tensor-width boundary, not the current physical tensor width.

## 10. Current Donchian behavior

**PASS**

Current `n_in=38` models still consume all 34 physical tensor columns, including Donchian columns, plus four return features.

The existing Donchian feature and integration tests passed after the correction.

Zero-ablation remains a 34-column physical tensor with zero-valued Donchian columns, which preserves current 38-wide paired-arm geometry.

## 11. Unsupported/corrupt geometry

**PASS**

The correction fails closed for unsupported model widths, too-narrow tensors, malformed metadata, invalid gate-matrix geometry, metadata/parameter disagreement, and loaded-parameter/runtime-width disagreement.

This is a substantive improvement over simply weakening the feature-count check.

## 12. Regression evidence

**PASS WITH LIMITED DEPTH**

The focused regression verifies:

- legacy base width 32;
- current base width 34;
- effective widths 36 and 38;
- legacy projection copies only columns 0-31;
- Donchian columns are excluded from legacy input;
- current projection includes Donchian columns;
- width 37 is rejected;
- contract resolution does not mutate persisted parameter shape.

The Release target also builds successfully.

### Validation limitation LFWC-RV-001

**Severity: LOW / evidence only**

The package does not include a real end-to-end load/resume/infer using model 1562 or 1601.

Recommended final closure when operationally safe:

1. Exercise model 1562 far enough through resume to prove the old `model=36,runtime=38` failure no longer occurs and the loaded LSTM has `n_in=36`.
2. Exercise model 1601 through direct or scheduler inference far enough to prove successful 36-wide parameter loading and model-input assembly.

No retraining or rewriting of those models should be necessary.

## Interaction with scheduler orphan-recovery correction

**NO CONFLICT FOUND**

The coexisting `Sources/ExperimentScheduler.cpp` changes concern orphan checkpoint/final-model recovery and exact-attempt fencing. They are orthogonal to the feature-width compatibility contract. No source-level conflict was found in the packaged state.

## Final conclusion

**PASS WITH ONE NON-BLOCKING VALIDATION LIMITATION**

The legacy/pre-Donchian feature-width compatibility correction is technically sound and ready for integration from a source-correctness standpoint.

It preserves the semantics historical 36-wide models were trained against, retains the current 38-wide Donchian layout, strengthens metadata/parameter validation, and rejects unknown layouts rather than silently adapting them.

The only remaining work for complete empirical closure is a live-safe end-to-end validation with one or both affected 36-wide persisted models.
