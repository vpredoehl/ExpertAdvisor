# LSTM Train / Inference Preprocessing Clamp Mismatch — Independent Review

**Repository:** `vpredoehl/ExpertAdvisor`
**Branch reviewed:** `campaign-operations`
**Review type:** Independent source review
**Disposition:** **CONFIRMED DEFECT**
**Scope:** 3-class classification input preprocessing immediately before the LSTM

## Executive Finding

The active training and 3-class classification inference paths do not apply identical final preprocessing to the model input tensor.

Training explicitly clamps every assembled model input feature to the interval `[-10.0, 10.0]` before the data is packed into timestep matrices and presented to the LSTM.

The classification inference path `EA::LSTM::PredictNextDirectionProbs()` assembles the corresponding model-width feature row and validates that each feature is finite, but it does not apply the same clamp before calling `forwardStep()`.

The source additionally emits a diagnostic that explicitly reports this mismatch:

```text
DIAG_NORM_WARN
kind=classification_inference_prelstm_unsanitized
training_clamp=10
classification_inference_clamp=0
```

This confirms that the difference is present in the current code path rather than being an ambiguity in the review.

## Training Path Evidence

In `LSTM/LSTM.cpp`, the training path builds `prebuilt_rows`, verifies each feature with `std::isfinite()`, and then applies:

```cpp
for (size_t i = 0; i < batchRows * featureCount; ++i)
    dst[i] = std::clamp(dst[i], -10.0f, 10.0f);
```

The clamp occurs before the packed timestep matrices are consumed by the LSTM.

Therefore, checkpoints trained through this path were optimized using a pre-LSTM feature domain bounded to `[-10, 10]`.

## Classification Inference Path Evidence

`EA::LSTM::PredictNextDirectionProbs()`:

1. copies the base feature channels into `model_row`;
2. appends the enabled multi-horizon return channels;
3. validates every resulting value with `std::isfinite()`;
4. calls `forwardStep(model_row, ...)`.

The current path does not clamp `dst[c]` before `forwardStep()`.

The same function emits:

```text
training_clamp=10
classification_inference_clamp=0
```

which directly corroborates the observed source behavior.

## Impact

This is a **train/inference preprocessing mismatch**, not a label leakage issue.

For values already inside `[-10, 10]`, training and inference receive the same value.

For any feature with magnitude greater than 10:

- training presents `+10` or `-10` to the LSTM;
- classification inference presents the raw out-of-range value.

Accordingly, inference can enter feature regions that the trained checkpoint was never exposed to through the training preprocessing path.

The mismatch is especially relevant to scaled return-like, volatility/range, ATR-normalized, and other feature channels that can produce large magnitudes under unusual market conditions.

## Recommended Correction

Preserve the existing training behavior and make classification inference apply the identical clamp.

The inference preprocessing loop should retain the non-finite check first, then clamp:

```cpp
for (size_t c = 0; c < modelFeatureCount; ++c)
{
    const float v = dst[c];

    if (!std::isfinite(v))
    {
        // existing diagnostic/assert path
    }

    dst[c] = std::clamp(v, -10.0f, 10.0f);
}
```

The non-finite check should remain before clamping so that NaN/Inf values are detected rather than hidden.

## Compatibility Assessment

The recommended fix preserves compatibility with existing checkpoints.

Existing checkpoints were trained with the `[-10, 10]` input clamp. Making inference obey the same clamp aligns deployment behavior with the domain on which those checkpoints were trained.

Removing the training clamp instead would change the training distribution for future models and would not restore parity for already-trained checkpoints.

## Verification Requirements

At minimum, verification should establish:

- values below `-10` become `-10` in both training and classification inference;
- in-range values are unchanged;
- values above `+10` become `+10` in both paths;
- NaN/Inf values continue to trigger the existing failure diagnostics before clamping;
- the classification inference diagnostic no longer reports `classification_inference_clamp=0`;
- the Release target builds successfully.

A focused regression test that directly compares pre-LSTM training and inference preprocessing would be preferable before proceeding with new feature experiments.

## Project Sequencing Recommendation

Resolve and independently verify this preprocessing parity defect before introducing the proposed Donchian feature increment.

Feature A/B testing should begin from a baseline where training and inference use identical final preprocessing; otherwise measured differences could be confounded by the existing mismatch.

## Final Disposition

**CONFIRMED — correction recommended before feature-expansion experiments.**
