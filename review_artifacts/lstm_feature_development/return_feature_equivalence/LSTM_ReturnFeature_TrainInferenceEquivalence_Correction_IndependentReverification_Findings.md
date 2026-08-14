# LSTM Return Feature Train/Inference Equivalence Correction — Independent Reverification Findings

**Date:** 2026-08-14
**Branch:** `lstm-feature-development`
**Baseline commit:** `b05387a0586e6f1cca5a12ca1e78ac59a92ce7c4`
**Review input:** `ReturnFeature_TrainInferenceEquivalence_IndependentReverification_Input.tar.gz`
**Disposition:** **CORRECTION VERIFIED; RELEASE BUILD PROVENANCE REMAINS A SEPARATE POST-COMMIT GATE**

## 1. Scope

This reverification independently reviewed the focused correction for the appended multi-horizon return-feature train/inference mismatch identified in the current feature-pipeline baseline review.

The review covered:

- the worktree patch against the recorded baseline commit,
- `Headers/ReturnFeatureHistory.hpp`,
- `Headers/LSTM.hpp`,
- `LSTM/LSTM.cpp`,
- `LSTM/main.cpp`,
- `Tests/LSTMFeatureVectorParityTests.cpp`,
- `Tests/LSTMFeatureVectorParityTests.sh`,
- the prior baseline review,
- the implementation report, and
- the packaged Git-state/provenance metadata.

The question under review was narrow: whether return-1/4/8/16 features are now anchored to the same Tensor-global causal history coordinate in training, inference, and baseline diagnostics, rather than to batch/window-local row offsets.

## 2. Root-Cause Correction

The original defect was correctly identified as a coordinate-system mismatch.

Previously, return history was computed from a supplied `Window` plus a local row index. A training row could therefore access history preceding the beginning of its local training window because its row index was relative to the enclosing batch, while inference row zero was relative to the inference window and returned zero for all unavailable local lookbacks.

The correction removes that ambiguity.

`ComputeLookbackLogReturn` and `AppendMultiHorizonReturnFeatures` now accept a Tensor-global position. The shared helper in `Headers/ReturnFeatureHistory.hpp` computes:

```text
log(close[currentGlobalPosition] /
    close[currentGlobalPosition - lookbackBars])
```

when the global history exists, and zero otherwise.

The helper reads no row after `currentGlobalPosition`.

**Finding:** **PASS**

## 3. Training Path

`CalculateBatch` establishes:

```text
batchGlobalStartIdx = batch.begin() - t.begin()
```

and appends return features at:

```text
batchGlobalStartIdx + r
```

for each prebuilt training row.

This directly converts the enclosing-batch-local row to the Tensor-global observation coordinate before any return lookup.

The prior local-batch-history behavior is removed.

**Finding:** **PASS**

## 4. Inference Paths

All three reviewed inference paths now establish a global window origin:

```text
windowGlobalStartIdx = w.begin() - t.begin()
```

and append return features at:

```text
windowGlobalStartIdx + rowIdx
```

The reviewed sites are:

- `PredictNextDirectionProbs`
- `PredictNextReturn`
- `PredictNextRelativeMove`

All three therefore use the same Tensor-global return-history definition as training.

No remaining production `AppendMultiHorizonReturnFeatures` call in the supplied `LSTM/LSTM.cpp` passes a batch/window object or a purely local row coordinate.

**Finding:** **PASS**

## 5. Baseline / Diagnostic Path

`BaselineFeatureAt` now derives:

```text
currentGlobalPosition = ex.globalStart + windowRow
```

and `BaselineLookbackLogReturn` uses the same shared global-position helper.

`BuildBaselineExamples` establishes `ex.globalStart` as:

```text
batchStart + localStart
```

so the diagnostic path is aligned with the corrected production coordinate system.

**Finding:** **PASS**

## 6. Return Semantics Preserved

The correction preserves the intended feature semantics:

- lookbacks remain `1, 4, 8, 16`,
- output order remains return-1, return-4, return-8, return-16,
- scaling remains `EA::LSTM::kFeatScale`,
- startup history remains zero when `currentGlobalPosition < lookbackBars`,
- non-finite or non-positive close values still produce zero,
- the helper itself remains unclamped, and
- the existing downstream model-input clamp remains responsible for `[-10, 10]`.

The fixed four-lookback helper is consistent with the supplied production contract: both `LSTM/LSTM.cpp` and `LSTM/main.cpp` statically require the configured return-feature count to equal `EA::kModelReturnFeatureCount` (four).

**Finding:** **PASS**

## 7. Causality / No-Look-Ahead

The shared helper accesses only:

```text
currentGlobalPosition
currentGlobalPosition - lookbackBars
```

and explicitly rejects a lookback that precedes Tensor position zero.

A future raw-close mutation therefore cannot affect a return feature for an earlier current position.

The supplied regression test covers this property, and an independent standalone compile/run of the packaged `ReturnFeatureHistory.hpp` also verified:

- startup boundaries at 1/4/8/16,
- nonzero complete-history returns,
- ordered geometric-return magnitudes,
- and invariance to future-close mutation.

The independent standalone helper test compiled with:

```text
-std=c++20 -Wall -Wextra -Werror
```

and exited successfully.

**Finding:** **PASS**

## 8. Focused Regression Test Review

`Tests/LSTMFeatureVectorParityTests.cpp` is appropriately targeted to the reported defect.

It covers:

- the previously failing global offset 16 case,
- byte-identical synthetic training-style and inference-style rows,
- boundaries immediately before and at 1/4/8/16 bars,
- a later complete-history position,
- fixed appended feature order,
- future-data independence,
- unclamped helper / downstream clamp behavior,
- current 38-wide model input, and
- legacy 36-wide model-input projection.

The test is partly structural rather than a direct invocation of `CalculateBatch` versus each production inference method; however, static inspection of the supplied production source confirms that those production sites now feed the same global-coordinate helper. The test therefore provides useful regression coverage rather than being the sole basis for the equivalence finding.

**Finding:** **PASS**

## 9. Compatibility

The correction does not change:

- model input widths,
- persisted model metadata shape,
- base tensor feature count,
- return-feature count,
- return-feature ordering,
- scaling,
- clamp semantics,
- schema,
- database migration state,
- Campaign Operations behavior, or
- H4 deployment state.

The change is semantic only in the history coordinate used by the four existing return columns.

Existing 36-wide and 38-wide contracts remain structurally unchanged in the reviewed material.

**Finding:** **PASS**

## 10. Separate Query-Start Prehistory Limitation

The prior baseline review also identified a different issue: a Tensor loaded from a later query `fromDate` starts EMA/ATR/rolling/Donchian and raw-history state at that query boundary unless prior rows are explicitly loaded.

This correction does **not** change that behavior, and the implementation report correctly leaves it as a separate limitation.

That limitation does not invalidate this specific correction: for observations within one bound Tensor, training, inference, and diagnostic return features now use the same Tensor-global history coordinate.

**Finding:** **OUT OF SCOPE / UNCHANGED**

## 11. Build and Test Evidence

The supplied implementation report states that these checks passed:

- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/LSTMModelInputCompatibilityTests.sh`
- standalone `DonchianFeatureTests.cpp`
- Debug Xcode build of the changed production sources

The package does not include the complete repository dependency set or raw build/test transcripts, so those reported executions cannot be independently replayed in full from the archive alone.

The reported Release build was blocked before compilation by the repository's clean-source-tree provenance requirement. That is expected for an intentionally uncommitted implementation and is not evidence of a source defect.

A clean Release provenance build therefore remains an appropriate **post-commit** gate.

**Finding:** **SOURCE CORRECTION VERIFIED; CLEAN RELEASE BUILD STILL REQUIRED AFTER COMMIT**

## 12. Package Integrity Note

All packaged files listed in `SHA256SUMS.txt` were independently rehashed.

Every non-manifest file matched its recorded SHA-256.

The sole mismatch was the `SHA256SUMS.txt` file's checksum of itself. The packaging script redirects output into `SHA256SUMS.txt` while also including that file in the `find` input set, so a stable self-hash cannot be produced by that procedure.

This is a packaging-manifest defect only. It does not affect the integrity verification of the source, patch, tests, review outputs, Git baseline, or branch metadata.

For future packages, exclude `SHA256SUMS.txt` from its own manifest.

**Finding:** **NON-BLOCKING PACKAGING ISSUE**

## 13. Final Determination

No blocking defect was found in the return-feature equivalence correction.

The supplied source establishes that the same market observation now resolves return-1/4/8/16 history through one Tensor-global causal coordinate in:

- training prebuild,
- direction inference,
- return regression inference,
- relative-move inference, and
- baseline diagnostics.

The original batch-local versus window-local mismatch is therefore corrected.

```text
RETURN_FEATURE_ROOT_CAUSE=CONFIRMED
RETURN_FEATURE_GLOBAL_COORDINATE_HELPER=PASS
TRAINING_GLOBAL_HISTORY_COORDINATE=PASS
DIRECTION_INFERENCE_GLOBAL_HISTORY_COORDINATE=PASS
RETURN_INFERENCE_GLOBAL_HISTORY_COORDINATE=PASS
RELATIVE_MOVE_INFERENCE_GLOBAL_HISTORY_COORDINATE=PASS
BASELINE_DIAGNOSTIC_GLOBAL_HISTORY_COORDINATE=PASS
RETURN_ORDER_SCALE_STARTUP_SEMANTICS=PASS
NO_LOOKAHEAD=PASS
LEGACY_36_CURRENT_38_STRUCTURAL_COMPATIBILITY=PASS
FOCUSED_PARITY_REGRESSION=PASS
RETURN_FEATURE_TRAIN_INFERENCE_EQUIVALENCE_CORRECTION=VERIFIED
CLEAN_RELEASE_PROVENANCE_BUILD=REQUIRED_POST_COMMIT
PACKAGE_SHA256_SELF_ENTRY=NON_BLOCKING_MANIFEST_DEFECT
```

## 14. Recommended Next Boundary

Proceed with the normal source-control closure for this focused correction, then run the clean Release build from the committed clean tree.

Do not begin the proposed new volume feature until that clean Release build succeeds.

No database backup is indicated solely for this correction because no schema change is present in the reviewed material.
