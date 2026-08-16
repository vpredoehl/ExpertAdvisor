# LSTM Causal 8x32 RMS Volatility Regime — Independent Reverification Findings

## Verdict

**PASS WITH ONE MINOR CONTRACT-HARDENING ITEM**

The packaged implementation is structurally sound and matches the intended append-only feature design:

- column 38 is the causal 8-over-32 prior-return RMS volatility-regime feature;
- `feature_size = 39`;
- current model width is `n_in = 43`;
- historical persisted widths through `n_in = 42` preserve their exact tensor prefixes and do not consume column 38;
- the rolling state is causal and bounded to 32 prior returns;
- the 8-return short RMS bookkeeping is correct, including eviction after the 32-return long window is full;
- the current bar's return is retained only after the current bar's regime value is computed;
- recommendation semantic configuration v10 is recognized explicitly while v3-v9 remain distinct historical reconstruction versions.

No defect was found that requires redesign of the feature, its 8/32 definition, its column placement, or its historical model-width projection.

## Independent source review

### 1. Mathematical definition

`CausalVolatilityRegime8x32::AddCompletedClose()` computes, from already-retained prior returns:

- `longRms = sqrt(sum(last up-to-32 r^2) / longCount)`
- `shortRms = sqrt(sum(last up-to-8 r^2) / shortCount)`
- `regime = shortRms / longRms - 1`

The function computes this value before deriving and retaining the return from `previousClose` to the supplied `close`.

This satisfies the requested causal interpretation:

- approximately 0: short volatility resembles the longer regime;
- positive: short volatility is elevated;
- negative: short volatility is compressed.

### 2. Present-bar causality

The implementation computes `result` before `currentReturn` is formed and before `RetainReturn(currentReturn)` is called.

Therefore changing the supplied current close cannot affect the value emitted for that same bar, except that it changes the state available to later bars.

The focused test independently exercises this by advancing two otherwise identical states with sharply different current returns and asserting equal emitted regime values.

### 3. Bootstrap semantics

With no retained prior return, the emitted value is 0.

For 1 through 8 prior returns:

- `longCount == shortCount`;
- the short and long sums cover the same retained returns;
- therefore `shortRms == longRms`;
- the emitted regime is 0 whenever the RMS is meaningful.

This matches the specified deterministic bootstrap behavior.

### 4. Rolling-window bookkeeping and eviction

The implementation retains at most 32 returns.

When a 33rd return is about to be retained:

1. the oldest retained return is removed from the 32-return long square sum;
2. that oldest value is popped from the deque;
3. the new return is pushed and added to both sums;
4. the value that has just moved outside the newest 8 returns is removed from the short square sum.

The short-window subtraction index

`priorReturns[priorReturns.size() - 8 - 1]`

is correct after the new return is pushed.

No off-by-one error was found in either the 8-return or 32-return windows.

### 5. Invalid-close continuity

A return is retained only when both the previous and current close are finite and positive.

An invalid close becomes the new `previousClose`, so the next bar also cannot form a return across the invalid bar. This avoids manufacturing a synthetic close-to-close return across a bad observation.

All emitted feature values are forced finite.

### 6. Tensor placement

`FeatureLayout.hpp` preserves the historical append-only layout and defines:

- legacy prefix: columns 0..31
- Donchian: 32..33
- session phase: 34..35
- relative tick volume: 36
- causal RMS return surprise: 37
- causal 8x32 RMS volatility regime: 38
- `feature_size = 39`

`Tensor::Add` computes the volatility-regime value and writes it only to `causalVolatilityRegimeCol`.

No reinterpretation of columns 0..37 was found in the packaged changes.

### 7. Historical model-input compatibility

`ModelInputContract.hpp` resolves the following append-only contracts:

- `n_in=36` -> 32 tensor features + 4 return features
- `n_in=38` -> 34 tensor features + 4 return features
- `n_in=40` -> 36 tensor features + 4 return features
- `n_in=41` -> 37 tensor features + 4 return features
- `n_in=42` -> 38 tensor features + 4 return features
- `n_in=43` -> 39 tensor features + 4 return features

The projection copies exactly `contract.tensorFeatureCount` values.

Accordingly, an existing `n_in=42` model stops at column 37 and cannot consume the new column 38. Only the new current `n_in=43` contract consumes it.

This is the desired persisted-model compatibility behavior.

### 8. Recommendation semantic v10 transition

The packaged recommendation code explicitly adds `RecommendationSemanticConfigurationVersion::v10`.

The parser recognizes canonical prefixes v3 through v10 separately.

The reviewed reconstruction logic maps:

- v3 -> 3
- v4 -> 4
- v5 -> 5
- v6 -> 6
- v7 -> 7
- v8 -> 8
- v9 -> 9
- v10 -> 10

Existing historical interpretation branches remain version-dependent rather than silently upgrading old canonical records to v10.

The materialization and outcome-assessment reconstruction paths include v10 only where later-version fields already apply, while retaining explicit v3/v4/v5 historical handling.

No historical semantic-version collapse was found in the packaged code.

## Minor contract-hardening item

The prompt requires:

> If `long_rms` is non-finite or effectively zero, emit 0.

The implementation currently tests:

`std::isfinite(longRms) && longRms > 0.0`

This handles exact zero but does not define an explicit nonzero threshold for "effectively zero."

This is a **minor specification mismatch**, not evidence of a practical feature failure in the reviewed FX data path. The implementation also suppresses nonfinite final ratios.

Recommendation:

- either define and justify a numerically meaningful lower bound for `longRms`, with a focused test;
- or explicitly document that, given the float close representation and this feature's numerical domain, the intended degeneracy rule is exact zero and revise the written contract accordingly.

Do not introduce an arbitrary clipping or epsilon constant merely to satisfy wording.

## Test/build evidence

The packaged implementation output records successful execution of:

- `Tests/LSTMCausalVolatilityRegimeTests.sh`
- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/LSTMModelInputCompatibilityTests.sh`
- `Tests/LSTMCausalReturnSurpriseTests.sh`
- `Tests/LSTMRelativeTickVolumeTests.sh`
- `Tests/LSTMTrueSessionPhaseTests.sh`
- direct warning-as-error `ExperimentRecommendationTests`
- `git diff --check`

The package does not contain the complete repository/dependency tree needed to independently rerun every maintained test suite in isolation, so these successful executions are treated as implementation-run evidence rather than re-executed evidence.

The implementation output also correctly states that the Debug source compilation reached the existing provenance step and then failed on a DerivedData filesystem permission error. No Release qualification was claimed.

A clean-tree Release build remains required before final integration/experimental qualification.

## Package integrity observation

The package's `SHA256SUMS.txt` verifies every packaged file **except itself**.

This is caused by the packaging script including `SHA256SUMS.txt` in the `find` input while simultaneously writing that same file. Its own recorded digest therefore cannot match the finalized manifest.

This is a packaging-script issue only; it does not indicate corruption of the implementation files.

Future packaging should exclude `SHA256SUMS.txt` from the hash input, for example:

`find "${PKG}" -type f ! -name SHA256SUMS.txt -print0`

All other files listed in the manifest verified successfully.

## Closure recommendation

The feature is suitable to proceed to final qualification after:

1. resolving or formally closing the "effectively zero" wording/threshold issue;
2. running the clean Release build under the repository's provenance requirements;
3. archiving the successful Release result.

No broader implementation correction is recommended.
