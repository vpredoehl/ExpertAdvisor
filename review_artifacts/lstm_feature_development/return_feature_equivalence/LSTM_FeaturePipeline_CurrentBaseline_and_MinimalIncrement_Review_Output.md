---
title: "LSTM Feature Pipeline Current Baseline and Minimal Increment Review"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeaturePipeline_CurrentBaseline_and_MinimalIncrement_Review_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Feature Pipeline Current Baseline and Minimal Increment Review

# LSTM Feature Pipeline — Current Baseline and Minimal Increment Review

## 1. Executive Determination

**BLOCKED_BY_CURRENT_PIPELINE_DEFECT**

The current model input is 38-wide: 34 tensor features plus four appended causal return features. The base 34 features are shared between training and inference, but the appended return features are not semantically equivalent:

- Training computes them relative to the enclosing 256-row batch.
- Inference computes them relative to the 64-row model window.

For a window beginning at batch offset 16 or later, its first timestep has nonzero return values in training but all four return columns are zero in inference. Evidence: [LSTM.cpp:3342](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3342), [LSTM.cpp:3555](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3555), [LSTM.cpp:1553](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1553), [LSTM.cpp:1939](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1939).

No feature increment should be implemented until this equivalence defect is repaired and regression-tested.

## 2. `input15m.txt` Source Contract

`input15m.txt` is a static PostgreSQL-style result dump, not SQL and not a runtime configuration file. It has 369,855 displayed data rows, ordered from `2010-01-03 17:00:00` through `2024-12-31 16:45:00`.

Its columns are:

| Column | Status in active pipeline |
|---|---|
| `dt` | Used |
| `open` | Used |
| `close` | Used |
| `high` | Used |
| `low` | Used |
| `vol` | Retrieved by the query, currently ignored by the parser/model |
| `target` | Retrieved by the query, currently ignored by the parser/model |

There are no active-code references to `input15m.txt`; it is a representative source snapshot only.

The active source is:

```sql
select * from candlestick('<symbol>', 15, 'minute', '<fromDate>', '<toDate>')
order by dt;
```

at [main.cpp:7268](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7268).

The repository’s `cst` return type is `(dt, open, close, high, low, vol, target)` and `target` is `(high + low + close) / 3`; see [candlestick.plpgsql:2](/Volumes/Developer%20SSD/ExpertAdvisor/candlestick.plpgsql:2) and [candlestick.plpgsql:22](/Volumes/Developer%20SSD/ExpertAdvisor/candlestick.plpgsql:22).

`db_input_iterator<Feature>` reads only `dt/open/close/high/low` by name; it does not read `vol` or `target` ([db_cursor.cpp:213](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp:213)).

Timestamp parsing correctly applies deterministic New York civil-time conversion and rejects nonexistent/ambiguous DST hours ([db_cursor.cpp:119](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp:119), [db_cursor.cpp:138](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp:138)).

## 3. Current Ordered Model Feature Contract

Base tensor construction is centralized in [`Tensor::Add`](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:78). Training copies the tensor prefix at [LSTM.cpp:3338](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3338); inference does the same at [LSTM.cpp:1550](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1550), [LSTM.cpp:7306](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:7306), and [LSTM.cpp:7399](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:7399).

All 38 model values are clamped to `[-10, 10]` immediately before LSTM evaluation/training.

| Index | Feature | Transformation / source |
|---:|---|---|
| 0 | open return | `1000 * log(open_t / close_t-1)` |
| 1 | close return | `1000 * log(close_t / close_t-1)` |
| 2 | high return | `1000 * log(high_t / close_t-1)` |
| 3 | low return | `1000 * log(low_t / close_t-1)` |
| 4 | candle body | `close_return - open_return` |
| 5 | log range | `high_return - low_return` |
| 6 | rolling volatility | population stddev of up to 32 close log returns, including current, scaled by 1000 |
| 7 | rolling return | cumulative up-to-32 close log return, including current, scaled by 1000 |
| 8 | intrabar time sine | `sin(2π * epoch_seconds mod 900 / 900)` |
| 9 | intrabar time cosine | cosine counterpart |
| 10 | UTC weekday sine | sine of UTC second-of-week |
| 11 | UTC weekday cosine | cosine counterpart |
| 12 | upper wick | `(high_return - max(open_return, close_return)) / max(range, 1e-6)` |
| 13 | lower wick | `(min(open_return, close_return) - low_return) / max(range, 1e-6)` |
| 14 | EMA-8 log distance | `(close_return - EMA8_return) / guarded_range` |
| 15 | EMA-21 log distance | `(close_return - EMA21_return) / guarded_range` |
| 16 | EMA-50 log distance | `(close_return - EMA50_return) / guarded_range` |
| 17 | EMA 8/21 log spread | `(EMA8_return - EMA21_return) / guarded_range` |
| 18 | EMA 21/50 log spread | `(EMA21_return - EMA50_return) / guarded_range` |
| 19 | EMA-8 ATR distance | `(close - EMA8) / ATR14` |
| 20 | EMA-21 ATR distance | `(close - EMA21) / ATR14` |
| 21 | EMA-50 ATR distance | `(close - EMA50) / ATR14` |
| 22 | EMA 8/21 ATR spread | `(EMA8 - EMA21) / ATR14` |
| 23 | EMA 21/50 ATR spread | `(EMA21 - EMA50) / ATR14` |
| 24 | EMA-8 log slope | `log(EMA8_t / EMA8_t-1) * 1000 / guarded_range` |
| 25 | EMA-21 log slope | equivalent EMA-21 calculation |
| 26 | EMA-50 log slope | equivalent EMA-50 calculation |
| 27 | EMA-8 ATR slope | `(EMA8_t - EMA8_t-1) / ATR14` |
| 28 | EMA-21 ATR slope | equivalent EMA-21 calculation |
| 29 | EMA-50 ATR slope | equivalent EMA-50 calculation |
| 30 | body strength | `(close_return - open_return) / range`, or zero if range is zero |
| 31 | range expansion | current raw range / causal running mean raw range |
| 32 | Donchian-20 upper distance | `1000 * log(close_t / max(prior 20 highs))` |
| 33 | Donchian-20 lower distance | `1000 * log(close_t / min(prior 20 lows))` |
| 34 | return-1 | `1000 * log(close_t / close_t-1)` |
| 35 | return-4 | `1000 * log(close_t / close_t-4)` |
| 36 | return-8 | `1000 * log(close_t / close_t-8)` |
| 37 | return-16 | `1000 * log(close_t / close_t-16)` |

Feature assignments 0–31 are in [Tensor.cpp:116](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:116) through [Tensor.cpp:351](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:351). Donchian is calculated from prior highs/lows only at [Tensor.cpp:134](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:134). Appended returns are ordered at [LSTM.cpp:1961](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1961).

The first source row is a special bootstrap row: all 34 base tensor features are zero ([Tensor.cpp:94](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:94)). Invalid parsed timestamp/OHLC input aborts loading; parsed non-finite values are rejected ([db_cursor.cpp:219](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp:219)). Price sanity violations are diagnosed but not rejected.

Two current contracts coexist:

- Legacy models: 32 base + 4 returns = width 36.
- Current models: 34 base + 4 returns = width 38.

A `zero_ablation` Donchian run remains width 38 but places zeros in columns 32–33.

## 4. End-to-End Feature Data Flow

```text
candlestick(...) ORDER BY dt
  -> Feature parser: dt, open, close, high, low
  -> Tensor::Add: 34 base features + raw OHLC/history
  -> model-input projection: 32 or 34 base columns
  -> append four close-return columns
  -> finite check and clamp [-10, 10]
  -> Metal-backed LSTM input matrix
```

Training uses 256-row source batches and creates all valid overlapping 64-step windows within each batch ([LSTM.cpp:3584](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3584)). The default window is 64, horizon is 6, hidden width is 64 ([Params.hpp:62](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/Params.hpp:62)).

Labels are not `input15m.txt`’s `target` column. Classification labels use future raw highs/lows over the configured horizon ([TargetLabel.hpp:28](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/TargetLabel.hpp:28)). This is intentional target look-ahead, not an input feature.

There is no learned input normalization or feature z-score. `normalization_version` is persisted configuration metadata; it does not transform input columns. The only active input scaling is feature-local formulas and the final clamp.

## 5. Training / Inference Equivalence

**Classification: MISMATCH_FOUND**

Base tensor features are equivalent only when both paths load the same ordered query interval. They use the same `Tensor::Add` implementation, same timestamp conversion, float type, model-prefix projection, and `[-10, 10]` clamp.

The appended return columns are not equivalent:

1. Training prebuilds return features over the full enclosing batch:
   `AppendMultiHorizonReturnFeatures(batch, r, ...)` at [LSTM.cpp:3342](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3342).
2. It later puts those batch-relative values into each training window at [LSTM.cpp:3555](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3555).
3. Inference calls the same helper over the 64-row window:
   [LSTM.cpp:1553](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1553).
4. The helper returns zero whenever its supplied local row index is less than the requested lookback:
   [LSTM.cpp:1939](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1939).

Thus, for a training window starting at local batch position 16:

- training timestep 0 gets a valid return-16 from rows `16` and `0` of the batch;
- inference timestep 0 gets zero for return-1/4/8/16 because its local index is zero.

This is a material input-vector mismatch, despite both paths being causal.

A second identified precondition is query-start history. `Tensor` initializes EMA, ATR, rolling state, and Donchian history at the first retrieved row. Inference over a later `fromDate` does not preload prior rows, so the same market observation can have different base features than it had in training until the respective histories warm up. No persisted feature state or preload query exists.

CPU and Metal do not independently pack feature vectors: both consume the same packed matrices. Gate-state CPU-reference validation exists for build mode 1 ([LSTM.cpp:2027](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:2027)); the default is fused-Metal-only mode 2 ([BuildConfig.hpp:25](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/BuildConfig.hpp:25)).

## 6. Input-Dimension and Compatibility Audit

Authoritative dimensions:

- `legacy_feature_size = 32`, `feature_size = 34`: [FeatureLayout.hpp:6](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureLayout.hpp:6)
- appended return count = 4: [ModelInputContract.hpp:14](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:14)
- accepted model widths = 36 and 38: [ModelInputContract.hpp:27](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:27)

Structural use sites include:

- Tensor allocation: [Tensor.cpp:92](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:92)
- training packed matrices: [LSTM.cpp:3315](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3315)
- inference input rows: [LSTM.cpp:1535](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1535)
- LSTM gate matrix shape: `(n_in + hidden_size) × 4*hidden_size`, [LSTM.cpp:3074](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3074)
- persistence `model_meta = [schemaVersion, n_in, hidden_size]`: [PgModelIO.hpp:222](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp:222)
- persisted shape validation: [PgModelIO.hpp:409](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp:409)
- resume/inference construction from persisted width: [main.cpp:7276](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7276)

Existing 36-wide and 38-wide models can currently load against a 34-column tensor. The projection intentionally copies only the prefix learned by the model ([ModelInputContract.hpp:63](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:63)).

If a new base feature makes the physical tensor 35-wide and new models 39-wide, current 38-wide models will load safely only if width 38 remains an accepted explicit projection to the first 34 columns. Existing models must not be “upgraded” by adding a weight row.

No database schema or migration change is required for an append-only 39-wide model: `model_meta` and parameter shape already persist and validate `n_in`. The compatibility risk is code-only: the width switch currently accepts exactly 36 and 38, and has no feature-layout hash/version beyond width. A semantic change that preserves width would be unsafe; an append-only 39-width contract is distinguishable.

## 7. Minimal Feature Increment Recommendation

No feature increment is approved while the vector-equivalence defect remains.

Conditional on repairing it, the single smallest justified next feature is:

`causal_log_relative_tick_volume`

```text
log((volume_t + 1) / (mean(previous up to 32 volumes) + 1))
```

It uses the already returned `vol` source field, has a plausible independent market-activity contribution, handles zero volume deterministically, requires no learned statistics, and can occupy appended base column 34.

Rejected alternatives:

- Close-location value: already derivable from current OHLC-derived columns and would add little independent information.
- More Donchian horizons: expands the feature family and compatibility/testing surface more than a single activity-state feature.

## 8. Leakage and Temporal-Semantics Review

For the conditional volume feature:

- It uses current completed-bar volume and at most 32 prior volumes.
- It uses no future row or candle.
- The first row is zero; later startup rows use the available prior subset.
- `+1` makes zero volume defined.
- Final model-input clamp remains `[-10, 10]`.
- No global mean/std statistic exists, so no train/validation leakage occurs through learned preprocessing.

Training and inference would still need the same history policy. The preferred policy is to calculate all history-dependent inputs against the full retrieved tensor and require/preload the maximum necessary prior history for an inference range. That same policy repairs the existing appended-return defect.

## 9. Exact Implementation Sites

Required before any feature work:

| File / site | Required change |
|---|---|
| [LSTM.cpp:1935](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1935) | Make return lookbacks use the Tensor-global current position, rather than the local batch/window position. |
| [LSTM.cpp:3342](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3342) | Pass the global position while prebuilding training rows. |
| [LSTM.cpp:1553](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:1553), [LSTM.cpp:7309](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:7309), [LSTM.cpp:7402](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:7402) | Pass the same global position in direction and regression inference. |
| [main.cpp:872](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:872) | Align diagnostic/baseline feature construction with the repaired return-history definition. |

Deferred, conditional volume-feature sites:

| File / site | Required change |
|---|---|
| [PricePoint.hpp:31](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PricePoint.hpp:31), [db_cursor.cpp:213](/Volumes/Developer%20SSD/ExpertAdvisor/Common/db_cursor.cpp:213) | Add and strictly parse `Feature::volume` from `vol`. |
| [FeatureLayout.hpp:6](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureLayout.hpp:6) | Preserve 32 and 34 historical base widths; add column 34 and make current base width 35. |
| [Tensor.hpp:51](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/Tensor.hpp:51), [Tensor.cpp:78](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp:78) | Retain prior volumes and compute the causal relative-volume value without altering columns 0–33. |
| [ModelInputContract.hpp:14](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:14) | Accept 36→32, 38→34, and 39→35 projections; create new models at width 39. |
| [Tests/LSTMModelInputCompatibilityTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMModelInputCompatibilityTests.cpp:10) | Extend structural compatibility coverage for all three widths. |

No schema, migration, or Campaign Operations change is required.

## 10. Regression Test Plan

1. Add a production-feature-vector parity test proving training and inference produce byte-identical 38-value rows for windows beginning before and after offsets 1, 4, 8, and 16.
2. Test return-feature startup boundaries and no-look-ahead behavior.
3. Test query-start/preload behavior explicitly: same source observation with identical prior history must match.
4. Extend `LSTMModelInputCompatibilityTests` to prove 36- and 38-wide persisted models still project correctly after introducing width 39.
5. For the deferred volume feature, add deterministic tests for first row, partial history, zero volume, lower/inside/upper ratio/clamp cases, and column placement.
6. Add Tensor integration coverage that verifies all old columns remain unchanged when volume is present.
7. Run gate-state validation mode 1 against non-production fixtures to confirm the shared packed input is accepted by CPU-reference and Metal paths.

The current tests cover Donchian causality and width 36/38 projection, but do not test training/inference feature-vector equality.

## 11. Build and Validation Plan

After implementation, run focused tests first:

```bash
Tests/LSTMModelInputCompatibilityTests.sh
Tests/LSTMFeatureVectorParityTests.sh
Tests/VolumeFeatureTests.sh
```

Then run the Release build without cleaning:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build
```

No build or test was run during this read-only review.

## 12. Risks / Blockers

- Blocking: appended-return training/inference mismatch.
- Precondition: feature histories reset at each query’s `fromDate`; a later inference start lacks EMA/ATR/Donchian/rolling prehistory.
- Compatibility metadata persists width, not a complete feature-layout version/hash.
- `vol` is returned by the SQL contract but currently ignored; parser behavior must be extended before it can be a feature.
- No production database, scheduler, Campaign Operations, H4 state, schema, or migrations were changed.

`git status --short`: clean
`git diff --stat`: no changes

## 13. Final Recommendation

Do not add a new feature yet.

Implement and test the narrow return-feature equivalence repair first. Once training and inference use the same global causal history, the next single feature increment should be `causal_log_relative_tick_volume`, with new models at width 39 and preserved 36/38 prefix compatibility.