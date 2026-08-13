# LSTM Donchian-20 Minimal Increment — Independent Reverification

## Disposition

**PASS WITH ONE TEST-COVERAGE FINDING**

The production Donchian-20 implementation is correct with respect to the requested feature definitions, causal boundary, channel layout, startup behavior, model-width propagation, and persisted-model feature-count compatibility.

The only remaining finding is test coverage: the focused regression test proves the helper semantics, but it does **not** execute `Tensor::Add` twice with different current-bar high/low values and compare the resulting Donchian tensor columns. The production code itself is structurally causal, so this is not a production-code defect. A minimal integration-level regression should be added before commit if executable proof at the `Tensor::Add` boundary is desired.

## Reverification Basis

Reviewed the packaged repository state on branch:

`lstm-feature-development`

at base HEAD:

`7e3d6abfbd934c889765abd992361038b84a8cd1`

The package contains the modified/new feature-layout and Donchian sources, the production `Tensor::Add` implementation, the relevant LSTM training/inference assembly code, the focused regression test, and the implementation report.

One packaging defect was observed: the requested `LSTM/Tensor.hpp` path does not exist; the repository uses `Headers/Tensor.hpp`. This omission does not block the conclusions below because the production `Tensor.cpp`, feature matrices, LSTM constructor, training path, inference path, and compatibility logic needed for this review are present in the package.

## 1. Feature Width

**PASS**

`FeatureLayout.hpp` defines:

```cpp
inline constexpr std::size_t legacy_feature_size = 32;
inline constexpr std::size_t donchianUpCol = legacy_feature_size;
inline constexpr std::size_t donchianDownCol = legacy_feature_size + 1;
inline constexpr std::size_t feature_size = legacy_feature_size + 2;
```

Therefore:

- legacy base width = 32
- new base width = 34
- Donchian upper column = 32
- Donchian lower column = 33

The LSTM still appends four existing multi-horizon return features, so the effective model input width becomes:

`34 + 4 = 38`

The LSTM constructor derives `n_in` from the actual tensor width plus `kReturnFeatureCount`, so the new width propagates from the tensor instead of relying on an unrelated hard-coded LSTM input constant.

## 2. Exact Donchian Definitions

**PASS**

`ComputeCausalDonchian20` computes the usable history length from prior high/low vectors, limits the starting index to the most recent 20 prior rows, then derives:

```cpp
upper =
    log(currentClose / priorMaximumHigh) * featureScale;

lower =
    log(currentClose / priorMinimumLow) * featureScale;
```

This matches the requested definitions:

```text
log(close_t / max(high[t-20 ... t-1])) * kFeatureScale

log(close_t / min(low[t-20 ... t-1])) * kFeatureScale
```

No current high or low is accepted as a helper argument.

The production call passes the existing `kFeatureScale`, so the new channels use the same feature scaling convention as the existing price-derived features.

## 3. Production Causality / Current-Bar Exclusion

**PASS**

The production `Tensor::Add` integration is causal.

The Donchian calculation occurs as:

```cpp
const auto [donchianUp, donchianDown] = ComputeCausalDonchian20(
    raw_high, raw_low, f.close, kFeatureScale);

p[donchianUpCol] = donchianUp;
p[donchianDownCol] = donchianDown;
```

At this point, `raw_high` and `raw_low` contain only previously added rows.

The current row is appended only near the end of `Tensor::Add`:

```cpp
raw_high.push_back(f.high);
raw_low.push_back(f.low);
```

Consequently, `f.high` and `f.low` cannot participate in the extrema used for the current row's Donchian channels.

The current close is intentionally used, as required.

No lookahead path was found in the implementation.

## 4. Existing Current-Bar Causality Regression

**PASS AT HELPER LEVEL; INTEGRATION TEST MISSING**

The focused test contains the following intended proof:

```cpp
const auto baseline =
    ComputeCausalDonchian20(highs, lows, 100.0f, ...);

const auto unchanged =
    ComputeCausalDonchian20(highs, lows, 100.0f, ...);

assert(unchanged == baseline);
```

The accompanying comment explains that current high/low are not helper inputs.

This correctly verifies the helper's interface-level causality, but it does not actually vary current-bar high/low and route both bars through `Tensor::Add`.

Therefore, it cannot catch a future integration regression in which `Tensor::Add` accidentally appends the current high/low before invoking the helper, or passes a different history vector.

### Minimal recommended integration regression

Construct identical prior history in two `Tensor` instances, then add final bars with:

- identical current close;
- identical open/time as needed;
- drastically different current high;
- drastically different current low.

Read the final feature rows and assert:

```text
tensorA[last][32] == tensorB[last][32]
tensorA[last][33] == tensorB[last][33]
```

within a small floating-point tolerance.

This test should use the real `Tensor::Add` path.

This finding is **test coverage only**. The reviewed production ordering is correct.

## 5. Rolling Window / Off-by-One

**PASS**

The helper computes:

```cpp
available = min(priorHighs.size(), priorLows.size());

start =
    available > donchian_lookback
        ? available - donchian_lookback
        : 0;
```

and iterates:

```cpp
for (size_t i = start; i < available; ++i)
```

For 21 prior observations and lookback 20, index 0 is excluded and indices 1 through 20 are included.

The regression test explicitly places extreme values in the first element of 21-element prior vectors and verifies they do not influence the result.

This correctly proves exclusion of the 21st prior row.

## 6. Startup Behavior

**PASS**

For the first tensor row, `Tensor::Add` zero-fills the entire feature matrix and returns after recording raw history. Thus the Donchian columns are neutral zero on row 0.

For later rows with fewer than 20 prior observations, the helper starts at index 0 and consumes only the available prior rows.

No future filling, duplicated bars, synthetic history, or incomplete-window suppression is introduced.

The focused test verifies:

- empty history -> `(0, 0)`;
- one prior row -> formula based on that row.

This is deterministic and causal.

## 7. Channel Layout

**PASS**

The new columns are appended after the existing legacy 32 base columns:

- column 32: Donchian upper-envelope distance
- column 33: Donchian lower-envelope distance

Existing legacy column indices 0 through 31 remain unchanged.

No existing feature was moved or overwritten.

## 8. Training / Classification Inference Parity

**PASS**

Both paths derive their base feature count from the tensor matrix width.

### Training

`CalculateBatch` determines:

```cpp
baseFeatureCount = (*batch.begin()).Shape()[1];
modelFeatureCount = n_in;
```

then copies all `baseFeatureCount` tensor columns into the model row before appending the existing return features.

With the new tensor layout, all 34 base columns, including columns 32 and 33, are copied in order.

### Classification inference

`PredictNextDirectionProbs` likewise determines:

```cpp
baseFeatureCount = (*w.begin()).Shape()[1];
modelFeatureCount = n_in;
```

and performs:

```cpp
memcpy(dst, src, baseFeatureCount * sizeof(float));
```

before appending the same return channels.

Both paths therefore consume the same 34 base columns in the same order and form a 38-column effective model row.

Both paths also apply the existing pre-LSTM finite checks and clamp policy to the completed model input.

No Donchian-specific training/inference branch exists.

## 9. Persisted 36-Channel Model Compatibility

**PASS**

Persisted `model_meta` stores the model input width (`n_in`).

The runtime compatibility logic compares persisted feature count against `RuntimeModelInputWidth(tensor)` and reports/rejects a `feature_count` mismatch.

With this feature increment:

- old persisted model input width = 36;
- new runtime input width = 38.

Therefore a normal old 36-channel model with valid `model_meta` cannot be silently interpreted as a 38-channel model.

The repository also treats `feature_count` as required minimum compatibility metadata in the reviewed model-validation path. No migration or weight-padding logic was added.

## 10. Invalid / Non-Finite Inputs

**PASS WITH ACCEPTABLE DEFENSIVE BEHAVIOR**

The helper:

- rejects non-finite or non-positive current close by returning neutral zeros;
- skips non-finite or non-positive prior highs/lows;
- returns zero for a side with no usable extrema;
- replaces a non-finite computed output with zero.

For valid FX price data, the requested formula is preserved exactly.

No inconsistent secondary scaling or normalization was introduced.

## 11. Verification Evidence

The implementation report records:

- focused Donchian test: passed;
- Release build: `** BUILD SUCCEEDED **`;
- `git diff --check`: passed;
- remaining Release warnings identified as pre-existing/unrelated.

The implementation transcript shows that an initial direct standalone compilation attempt encountered MetaNN include/warning issues. The implementation was subsequently reorganized to isolate the lightweight feature-layout/helper dependencies, and the final implementation report records the focused test as passing with `-Wall -Wextra -Werror`.

The final disposition relies on the final packaged source plus the reported final verification state, not on the failed intermediate compilation attempt.

## 12. Scope Discipline

**PASS**

The reviewed working-tree change is limited to:

- moving the feature-width/layout constants into `FeatureLayout.hpp`;
- adding `DonchianFeatures.hpp`;
- increasing base width from 32 to 34;
- populating two Donchian columns in `Tensor::Add`;
- adding focused Donchian regression coverage.

No additional indicator, architecture, hidden-size, layer-count, optimizer, scheduler, hyperparameter, database-schema, or persisted-weight migration change was introduced by this increment.

## Findings Summary

### Production findings

**None.**

The reviewed production Donchian implementation satisfies the requested semantics and causal boundary.

### Test-coverage finding

**D20-REV-001 — Low severity**

The focused test proves helper-level causality but does not execute the actual `Tensor::Add` integration with two different current-bar high/low values.

Recommended resolution before commit:

Add one small `Tensor::Add` integration regression proving that changing only current `high_t` / `low_t`, while preserving prior history and `close_t`, leaves feature columns 32 and 33 unchanged.

This does not require a production-code change.

## Final Conclusion

**PASS WITH ONE TEST-COVERAGE FINDING**

The Donchian-20 minimal increment is production-code correct and remains within the intended scope.

The implementation can proceed to closure after adding the small `Tensor::Add` causality regression described above and rerunning the focused test plus Release build. No Codex-sized correction pass is required; this is a narrow regression-coverage addition.
