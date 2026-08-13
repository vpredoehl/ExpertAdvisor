---
title: "LSTM Feature Pipeline Inventory and Minimal Increment Review"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeaturePipeline_Inventory_and_MinimalIncrement_Review_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.4-mini"
---

# LSTM Feature Pipeline Inventory and Minimal Increment Review

# LSTM Feature Pipeline Inventory and Minimal Increment Review

## 1. Executive Finding
The active LSTM pipeline does not read `input15m.txt` directly in repository code. The runtime path starts from a PostgreSQL `candlestick(...)` query, parses `dt/open/close/high/low` into a per-bar tensor, builds 32 base feature channels in `Tensor::Add`, then appends 4 causal lookback-return channels inside the LSTM input assembly, for a current per-timestep width of **36 channels** and a current sample shape of **64 timesteps × 36 channels**.

The smallest useful next feature to test is one Donchian-style breakout context family: **2 causal channels** capturing close vs recent rolling high and rolling low. It is new, strictly causal, derived from existing data only, and not already represented explicitly.

## 2. Active Data Path
1. `LSTM/main.cpp` opens a PostgreSQL cursor with `select * from candlestick(..., 15, 'minute', ..., ...) order by dt;` and iterates `db_cursor_stream<Feature>` into a `Tensor` named after the symbol. See [`LSTM/main.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp#L7134) around the candlestick query path and [`Common/db_cursor.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/Common/db_cursor.cpp#L86) for `Feature` parsing.
2. `Common/db_cursor.cpp::db_input_iterator<Feature>::ReadPP()` parses only `dt`, `open`, `close`, `high`, `low` from each row into `Feature{open, close, high, low, time}`. Volume and target are not parsed there.
3. `LSTM/Tensor.cpp::Tensor::Add(Feature)` converts each raw bar into a `MetaNN::Matrix<float, 1, 32>` row. The first row is zero-filled because there is no prior close yet; later rows are causal transforms of the current and prior bars. See [`LSTM/Tensor.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) around the feature builder.
4. `Tensor` stores the rows in `DataSet ds` and also caches raw arrays for `open/close/high/low/time` so label construction can use the same chronology. See [`Headers/Tensor.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp#L28).
5. `EA::LSTM::CalculateBatch()` and `EA::LSTM::PredictNextDirectionProbs()` append 4 additional return-lookback channels per timestep, widening each row from 32 to 36 features. See [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L3154) and [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1505).
6. Window assembly uses `window_size=64`, so one sample is a 64-step sequence. Training materializes packed step matrices of shape `B × 36` for each of the 64 steps; inference feeds one `1 × 36` row at a time through the same 64-step loop.
7. Labels are built separately by `BuildLookaheadClassInfo()` from the last bar in the 64-step window plus future bars up to `prediction_horizon`. See [`Headers/TargetLabel.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/TargetLabel.hpp#L28).

## 3. Current Model Input Tensor
Current default code-path dimensions:
- Base row tensor: `1 × 32`
- Appended causal return channels: `+4`
- Model input width: `36`
- Window length: `64`
- Per-sample shape: `64 × 36`

Training shape detail:
- `CalculateBatch()` prebuilds a contiguous `(batchRows, 36)` row tensor, then packs it into 64 step matrices of shape `(B, 36)` where `B` is the current mini-batch of windows. See [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L3262).

Important compatibility note:
- Saved model metadata persists `n_in`, and runtime validation rejects mismatched widths. See [`Headers/PgModelIO.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp#L187) and [`LSTM/main.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp#L4693).

## 4. Feature Inventory
Base tensor channels 0-31 are built in `Tensor::Add`. Appended channels 32-35 are added inside LSTM input assembly. All formulas below are causal.

| idx | feature | raw source columns | exact formula / transformation | scaling / normalization | causal at prediction time | train + infer | source |
|---|---|---|---|---|---|---|---|
| 0 | open vs prev close | `open`, prior `close` | `log(open / prev_close) * 1000` | fixed `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 1 | close vs prev close | `close`, prior `close` | `log(close / prev_close) * 1000` | `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 2 | high vs prev close | `high`, prior `close` | `log(high / prev_close) * 1000` | `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 3 | low vs prev close | `low`, prior `close` | `log(low / prev_close) * 1000` | `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 4 | candle body | open/close transforms | `c - o` where `o,c` are rows 0/1 | inherits `kFeatureScale` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 5 | candle range | high/low transforms | `h - l` where `h,l` are rows 2/3 | inherits `kFeatureScale` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 6 | realized volatility | `close` history | population stdev of log returns over up to 32 bars incl current, then `*1000` | `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 7 | rolling cumulative return | `close` history | sum of log returns over up to 32 bars incl current, then `*1000` | `kFeatureScale=1000` | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 8 | time-of-day sin | `dt` | `sin(2π * secInCycle / 900)` | none | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 9 | time-of-day cos | `dt` | `cos(2π * secInCycle / 900)` | none | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 10 | day-of-week sin | `dt` | `sin(2π * secOfWeek / 604800)` | none | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 11 | day-of-week cos | `dt` | `cos(2π * secOfWeek / 604800)` | none | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 12 | upper wick geometry | open/close/high/low transforms | `(h - max(o,c)) / max(range,1e-6)` | range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 13 | lower wick geometry | open/close/high/low transforms | `(min(o,c) - l) / max(range,1e-6)` | range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 14 | close vs EMA8 | `close`, EMA8 history | `(c - ema8_s) / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 15 | close vs EMA21 | `close`, EMA21 history | `(c - ema21_s) / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 16 | close vs EMA50 | `close`, EMA50 history | `(c - ema50_s) / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 17 | EMA8 vs EMA21 spread | EMA8, EMA21 history | `(ema8_s - ema21_s) / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 18 | EMA21 vs EMA50 spread | EMA21, EMA50 history | `(ema21_s - ema50_s) / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 19 | close vs EMA8 | `close`, EMA8 history | `(close - ema8) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 20 | close vs EMA21 | `close`, EMA21 history | `(close - ema21) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 21 | close vs EMA50 | `close`, EMA50 history | `(close - ema50) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 22 | EMA8 vs EMA21 spread | EMA8, EMA21 history | `(ema8 - ema21) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 23 | EMA21 vs EMA50 spread | EMA21, EMA50 history | `(ema21 - ema50) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 24 | EMA8 slope | EMA8 history | `log(max(ema8,1e-12)/max(ema8_prev,1e-12)) * 1000 / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 25 | EMA21 slope | EMA21 history | `log(max(ema21,1e-12)/max(ema21_prev,1e-12)) * 1000 / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 26 | EMA50 slope | EMA50 history | `log(max(ema50,1e-12)/max(ema50_prev,1e-12)) * 1000 / denom_range` | dynamic range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 27 | EMA8 raw slope | EMA8 history | `(ema8 - ema8_prev) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 28 | EMA21 raw slope | EMA21 history | `(ema21 - ema21_prev) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 29 | EMA50 raw slope | EMA50 history | `(ema50 - ema50_prev) / max(atr14,1e-12)` | ATR-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 30 | body strength | open/close/high/low transforms | `(c - o) / max(range,1e-6)` | range-normalized | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 31 | range expansion | raw high/low history | `(high - low) / rolling_mean(raw_range, 32)` | causal rolling mean | yes | yes | [`Tensor::Add`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77) |
| 32 | 1-bar log return | `close` history within current 64-step window | `log(close_t / close_{t-1}) * 1000` | `kFeatScale=1000` | yes | yes | [`AppendMultiHorizonReturnFeatures`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1934) |
| 33 | 4-bar log return | `close` history within current 64-step window | `log(close_t / close_{t-4}) * 1000` | `kFeatScale=1000` | yes | yes | [`AppendMultiHorizonReturnFeatures`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1934) |
| 34 | 8-bar log return | `close` history within current 64-step window | `log(close_t / close_{t-8}) * 1000` | `kFeatScale=1000` | yes | yes | [`AppendMultiHorizonReturnFeatures`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1934) |
| 35 | 16-bar log return | `close` history within current 64-step window | `log(close_t / close_{t-16}) * 1000` | `kFeatScale=1000` | yes | yes | [`AppendMultiHorizonReturnFeatures`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1934) |

Notes:
- Row 0 is zero-filled on the first bar because there is no prior close yet.
- Rows 32-35 are computed from the current 64-step window only and are zero for early positions that lack enough within-window history.
- `c`, `o`, `h`, `l` above are the scaled log-ratio values already created in `Tensor::Add`.
- `ema8_s`, `ema21_s`, `ema50_s`, and `denom_range` are the exact intermediate values used in `Tensor::Add`.

## 5. Raw Column Disposition

| raw field | disposition | evidence |
|---|---|---|
| `dt` | source for derived model inputs; metadata | Parsed in `Common/db_cursor.cpp`, then used to compute time-of-day and day-of-week sin/cos channels in `Tensor::Add` |
| `open` | source for derived model inputs | Used to derive row 0, body, and wick features in `Tensor::Add` |
| `close` | source for derived model inputs; also label source | Used for most row features and for lookahead labels via future close comparisons in `BuildLookaheadClassInfo()` |
| `high` | source for derived model inputs; also label source | Used for wick/range/EMA/ATR features and for lookahead label high-hit checks |
| `low` | source for derived model inputs; also label source | Used for wick/range/EMA/ATR features and for lookahead label low-hit checks |
| `vol` | unused in active path | Diagnostics explicitly report `raw_volume=0`; `db_input_iterator<Feature>::ReadPP()` never parses it |
| `target` | unused in active path | Diagnostics explicitly report `raw_target=0`; labels are reconstructed from future OHLC, not from file `target` |

## 6. Normalization, Causality, and Leakage Review
Observed facts:
- Input feature scaling is deterministic, not fit on training statistics.
- The only fixed scale factor on input features is `kFeatureScale=1000` for log-return-like features.
- Additional input normalization is local and causal: rolling mean, EMA, ATR, and per-bar ratios all use current and past data only.
- `Tensor::Add()` never references future bars.
- The appended return-lookback channels use only the current window and past bars inside that window.
- Labels intentionally use future bars, but only in `BuildLookaheadClassInfo()` and only for target construction.

Concern:
- Training and classification inference are not perfectly identical at the final preprocessing step. Training clamps every model input to `[-10, 10]` before the LSTM. The 3-class inference path checks finiteness but does not clamp. See [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L3335) and [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1505). This is a preprocessing mismatch, not feature leakage.

Open technical ambiguity:
- `ParseTimestampStrict()` uses `std::mktime()`, which is local-time sensitive, while day-of-week features later come from `gmtime_r()`. If the raw timestamps are not already in the expected local timezone, time features can shift. See [`Common/db_cursor.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/Common/db_cursor.cpp#L25).

## 7. Temporal and Label Semantics
A 64-step sample means 64 consecutive timestamps starting at `start` and ending at `start + 63`. The current bar is included as the last timestep in the input window.

Label construction:
- `lastIt = startIt + (window_size - 1)`
- `targetIt = lastIt + prediction_horizon`
- The label timestamp is the bar at `targetIt`.
- For `prediction_horizon=6` on 15-minute bars, the terminal label point is 90 minutes ahead of the last input bar.

Class rule:
- Scan future bars from `lastIt + 1` through `lastIt + prediction_horizon`.
- If the first qualifying future high exceeds `+threshold`, class becomes `up`.
- If the first qualifying future low falls below `-threshold`, class becomes `down`.
- If both happen, the earlier hit wins.
- If neither happens, class is `neutral`.

The label is not the file `target` column. It is reconstructed from future OHLC and the threshold rule in [`Headers/TargetLabel.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/TargetLabel.hpp#L28).

Gaps/weekends/session discontinuities:
- There is no explicit gap flag, weekend flag, or session-identity feature.
- Discontinuities are only implicit through the timestamp-derived cyclic features and the raw price path itself.

## 8. Information-Family Coverage Matrix

| category | coverage | why |
|---|---|---|
| absolute price level | indirectly / implicitly represented | The model sees price only through ratios to previous close and EMA/ATR-normalized distances, not raw absolute price. The sequence can infer relative scale, but the pipeline does not explicitly provide an absolute-level channel. |
| returns / price changes | directly represented | Rows 0-3, 7, and 32-35 are explicit return or change channels. |
| candle body | directly represented | Row 4 and row 30. |
| candle range | directly represented | Row 5 and row 31. |
| upper/lower wick geometry | directly represented | Rows 12-13. |
| volume / activity | not represented | `vol` is not parsed in the active path. |
| realized volatility | directly represented | Row 6, plus related ATR/range features. |
| trend | directly represented | EMA levels and slopes, plus cumulative return channels. |
| momentum | directly represented | Rows 7 and 32-35. |
| mean-reversion / distance-from-trend | directly represented | Rows 14-23. |
| time-of-day | directly represented | Rows 8-9. |
| day-of-week | directly represented | Rows 10-11. |
| session identity | not represented | No session flag or market-session encoding exists. |
| gap / discontinuity information | directly represented, but not explicitly sessionized | `open / prev_close` captures bar-to-bar gap; no separate gap/session-break marker exists. |
| multi-timeframe context | not represented | No higher/lower timeframe aggregation or cross-timeframe channels. |
| rolling extrema / breakout context | not represented | No prior-window high/low channels or breakout-distance channels. |
| RSI-type oscillator information | not represented | No explicit RSI/oscillator channel in the active pipeline. |
| EMA / moving-average information | directly represented | Rows 14-29. |
| Bollinger / volatility-band information | not represented | No explicit band-width or band-position feature. |
| Donchian information | not represented | No rolling high/low channel or Donchian breakout channel. |
| cross-symbol / market context | not represented | Single-symbol, single-series pipeline only. |

## 9. Smallest Useful Feature Increment
Recommend exactly one feature family: **Donchian breakout context**, implemented as **2 new scalar channels per timestep**.

Exact definition:
- Lookback window: `N = 20` bars
- Channel A: `donchian_up_20(t) = log(close_t / max(high_{t-20 .. t-1})) * 1000`
- Channel B: `donchian_down_20(t) = log(close_t / min(low_{t-20 .. t-1})) * 1000`

Why this is the best next test:
- It supplies explicit rolling-extrema / breakout context, which is currently absent.
- It is strictly causal.
- It uses only existing OHLC data.
- It is easy to compute identically in training and inference.
- It is interpretable: positive breakout vs support-break context.
- It is minimally invasive: 2 channels, no architecture redesign.

Why it is not redundant with current explicit inputs:
- Current features encode current-bar geometry, EMA distance, volatility, and short-horizon returns.
- They do not explicitly encode where the close sits relative to a prior resistance/support envelope.
- Donchian channels provide information about breakout position against recent extrema, which is not directly exposed elsewhere.

Implementation touchpoints for a future change:
- Add rolling high/low state and two new channels in `Tensor::Add()` in [`LSTM/Tensor.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77).
- Increase the base feature count constant in [`Headers/Params.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/Params.hpp#L73) from 32 to 34, or otherwise make the feature width explicit.
- Ensure model-width validation and persisted metadata reflect the new width in [`LSTM/main.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp#L4693) and [`Headers/PgModelIO.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp#L187).
- No change is needed to the label rule.

Normalization requirement:
- Use the same `*1000` log-ratio scale as existing return channels.
- No learned scaler should be introduced.

Warm-up handling:
- For bars with fewer than 20 prior observations, compute extrema over the available history; for the very first row, preserve the current zero-fill behavior.
- This matches the current causal, partial-history style used by rolling volatility and EMA state.

Leakage considerations:
- None, if the rolling extrema are built only from current and past bars.
- Do not include future bars in the extrema window.

Compatibility implications:
- Saved checkpoints become incompatible because input width changes.
- The current binary will reject mismatched `feature_count` / `modelInputWidth`.
- With hidden size 64, the LSTM gate input matrix would widen from `(100, 256)` to `(102, 256)`.

Minimal A/B experiment design:
- Baseline: current model, same symbol, same date range, same `window_size`, same `prediction_horizon`, same optimizer, same learning rate multipliers, same class weights, same epoch policy, same seed policy.
- Treatment: baseline plus the 2 Donchian channels only.
- Hold fixed: hidden size, number of layers, checkpoint cadence, evaluation metric, train/validation split.
- Compare: validation macro F1, balanced accuracy, per-class recall, and confusion matrix on the same held-out dates.
- Keep the feature only if the gain is consistent across matched seeds and not just a one-off validation bump, with no material degradation in neutral recall.

## 10. Minimal A/B Experiment Design
Controlled comparison only; do not run it.

- Same symbol.
- Same date range.
- Same `window_size`.
- Same `prediction_horizon`.
- Same label rule.
- Same class weights.
- Same optimizer family and learning-rate settings.
- Same hidden size and number of layers.
- Same epoch count and checkpoint policy.
- Same seed policy where supported.
- Same evaluation metric set.

Comparison pair:
- Baseline: current 36-channel model.
- Variant: current model plus the 2 Donchian channels, yielding 38 channels.

Acceptance evidence:
- Variant improves validation macro F1 and balanced accuracy on the identical holdout slice.
- Improvement is stable across repeated runs or seeds.
- No unacceptable degradation in neutral recall or calibration.

## 11. Implementation Touchpoints for a Future Change
Likely files and symbols that would need updates for the Donchian experiment:
- [`LSTM/Tensor.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp#L77): `Tensor::Add`
- [`Headers/Tensor.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp#L28): rolling state members
- [`Headers/Params.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/Params.hpp#L73): `feature_size`
- [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L3154): `CalculateBatch`
- [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L1505): `PredictNextDirectionProbs`
- [`LSTM/LSTM.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp#L4693): `RuntimeModelInputWidth`
- [`Headers/PgModelIO.hpp`]( /Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp#L187): `saveModelMeta`
- [`LSTM/main.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp#L7144): runtime model-width validation
- [`LSTM/main.cpp`]( /Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp#L436): diagnostics that print base feature counts

No source code was modified, and no build or tests were run for this review, per the read-only scope.

## 12. Open Questions / Uncertainties
- I could not verify the exact direct contents of `input15m.txt` from repository code because the active runtime path does not read it directly; it reads equivalent fields from PostgreSQL.
- The exact timezone semantics of timestamp parsing are not documented. `ParseTimestampStrict()` uses `mktime()`, which is local-time sensitive, while day-of-week derivation uses `gmtime_r()`.
- I could not verify the currently deployed checkpoint’s runtime `window_size` / `prediction_horizon` from repository state alone. The code default is 64 / 6, and runtime validation enforces consistency with loaded model metadata.
- I did not run `xcodebuild` or any tests because the task was explicitly read-only.