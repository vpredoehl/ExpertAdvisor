---
title: "LSTM Current Feature Inventory and Next Increment"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeaturePipeline_CurrentInventory_and_NextIncrement_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Current Feature Inventory and Next Increment

## Executive finding

The checked-out branch is `campaign-operations`, not `lstm-feature-development`. Its active LSTM feature contract is still **32 base + 4 appended returns = `n_in=36`**, with **64 × 36** samples.

The previously proposed Donchian-20 feature was implemented, tested, and parity-repaired on the separate `lstm-feature-development` branch (`4616f2a` onward), but it is **not merged into nor active on this branch**. This branch does contain a `donchian20_mode` experiment-provenance migration, but no LSTM runtime code reads that mode or produces Donchian channels. Thus the mode is presently inert relative to model inputs.

One material existing defect remains in the active branch: the four appended return channels are batch-relative in training but window-relative in inference. Base channels share semantics; appended channels do not for windows not starting at a batch boundary.

My exactly-one recommendation is therefore: **integrate causal Donchian-20 breakout context as two append-only base channels, but only after/alongside restoration of the already-developed return-feature parity correction as a baseline correctness prerequisite.** Result: **34 base + 4 returns = `n_in=38`**.

## Active data path

```text
PostgreSQL candlestick(symbol, 15, 'minute', from, to) ORDER BY dt
  → db_input_iterator<Feature>: dt/open/close/high/low only
  → Tensor::Add: 32 base columns
  → LSTM input assembly: append 1/4/8/16-bar close returns
  → finite check + clamp each input to [-10, 10]
  → LSTM
```

The production query is constructed at [LSTM/main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:7134>). The repository does not contain the deployed `candlestick(...)` function definition, so its exact live `vol` semantics cannot be certified from this checkout.

`Feature` contains only OHLC and time ([PricePoint.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/PricePoint.hpp:31>)); the cursor parses those five fields by name ([db_cursor.cpp](</Volumes/Developer SSD/ExpertAdvisor/Common/db_cursor.cpp:203>)). Timestamps are deterministically interpreted as New York civil time then converted to UTC, rejecting ambiguous/nonexistent DST times ([db_cursor.cpp](</Volumes/Developer SSD/ExpertAdvisor/Common/db_cursor.cpp:119>)).

## Dimensions and parity

| Item | Current active value |
|---|---:|
| Base width | 32 |
| Appended width | 4 |
| Model `n_in` | 36 |
| Window size | 64 |
| Per-sample shape | `64 × 36` |
| Training source-batch size | 256 |
| Input clamp | `[-10, 10]` |
| Active feature modes | One effective LSTM layout; `donchian20_mode` has no runtime effect |

`RuntimeModelInputWidth()` computes base width plus four returns ([main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:4693>)); model metadata persists `[schemaVersion, n_in, hidden_size]` ([PgModelIO.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp:187>)) and inference rejects a mismatched width ([main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:4945>)).

Base feature semantics are shared because both paths consume the same `Tensor`. Classification inference and training both clamp inputs. However, return features differ:

- Training calls `AppendMultiHorizonReturnFeatures(batch, r, ...)` over the enclosing batch ([LSTM.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:3296>)).
- Inference calls it over the 64-row inference window ([LSTM.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:1505>)).
- The helper returns zero when its local row index is less than the lookback ([LSTM.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:1917>)).

So, for example, a training window beginning at batch offset 16 has a valid return-16 at timestep zero, while inference assigns zero there. The separate branch commit `87c8b06` repairs this with global tensor positions, but that fix is not in the current branch.

A second parity caveat: EMA, ATR, rolling return/volatility state begin at query start. Inference using a later `fromDate` has different warm-up state for the same market bar unless prior history is preloaded.

## Current feature inventory

Let `r(x)=1000·log(x_t/close_{t-1})`, `range=h-l`, and `G=max(range, |log((prev_close+max(1e-4, .25·max(avgRange,ATR14)))/prev_close)|·1000)`. All base channels originate in [Tensor::Add](</Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp:77>); all appended channels originate in [AppendMultiHorizonReturnFeatures](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:1936>).

| Index | Channel / exact formula | Raw inputs, lookback, scaling/warm-up | Train / inference |
|---:|---|---|---|
| 0 | `r(open)` | O, prior C; ×1000; first row zero | Shared base |
| 1 | `r(close)` | C, prior C; ×1000; first row zero | Shared base |
| 2 | `r(high)` | H, prior C; ×1000; first row zero | Shared base |
| 3 | `r(low)` | L, prior C; ×1000; first row zero | Shared base |
| 4 | `r(close)-r(open)` | Current OHLC; inherited scaling | Shared base |
| 5 | `r(high)-r(low)` | Current H/L; inherited scaling | Shared base |
| 6 | population SD of up-to-32 close log returns, including current, ×1000 | C; partial history; zero until two returns | Shared base |
| 7 | sum of up-to-32 close log returns, including current, ×1000 | C; partial history | Shared base |
| 8 | `sin(2π·(epochSeconds mod 900)/900)` | `dt`; no scaling | Shared base |
| 9 | cosine counterpart | `dt`; no scaling | Shared base |
| 10 | `sin(2π·UTCSecondOfWeek/604800)` | `dt`; no scaling | Shared base |
| 11 | cosine counterpart | `dt`; no scaling | Shared base |
| 12 | `(h-max(o,c))/max(range,1e-6)` | Current OHLC; range-normalized | Shared base |
| 13 | `(min(o,c)-l)/max(range,1e-6)` | Current OHLC; range-normalized | Shared base |
| 14 | `(c-EMA8scaled)/G` | C; EMA-8 seeded first close | Shared base |
| 15 | `(c-EMA21scaled)/G` | C; EMA-21 seeded first close | Shared base |
| 16 | `(c-EMA50scaled)/G` | C; EMA-50 seeded first close | Shared base |
| 17 | `(EMA8scaled-EMA21scaled)/G` | C; EMA states | Shared base |
| 18 | `(EMA21scaled-EMA50scaled)/G` | C; EMA states | Shared base |
| 19 | `(C-EMA8)/max(ATR14,1e-12)` | OHLC; Wilder-style EMA ATR-14 | Shared base |
| 20 | `(C-EMA21)/max(ATR14,1e-12)` | OHLC; ATR-14 | Shared base |
| 21 | `(C-EMA50)/max(ATR14,1e-12)` | OHLC; ATR-14 | Shared base |
| 22 | `(EMA8-EMA21)/max(ATR14,1e-12)` | C; ATR-14 | Shared base |
| 23 | `(EMA21-EMA50)/max(ATR14,1e-12)` | C; ATR-14 | Shared base |
| 24 | `1000·log(EMA8/EMA8prev)/G` | C; EMA-8 state | Shared base |
| 25 | `1000·log(EMA21/EMA21prev)/G` | C; EMA-21 state | Shared base |
| 26 | `1000·log(EMA50/EMA50prev)/G` | C; EMA-50 state | Shared base |
| 27 | `(EMA8-EMA8prev)/max(ATR14,1e-12)` | C; ATR-14 | Shared base |
| 28 | `(EMA21-EMA21prev)/max(ATR14,1e-12)` | C; ATR-14 | Shared base |
| 29 | `(EMA50-EMA50prev)/max(ATR14,1e-12)` | C; ATR-14 | Shared base |
| 30 | `(c-o)/range`, or zero if `range==0` | Current OHLC; range-normalized | Shared base |
| 31 | `(H-L)/mean(last up-to-32 raw ranges, including current)` | H/L; partial rolling mean | Shared base |
| 32 | `1000·log(C_t/C_{t-1})` | C; zero for local row 0 | Appended; mismatched scope |
| 33 | `1000·log(C_t/C_{t-4})` | C; zero for local rows <4 | Appended; mismatched scope |
| 34 | `1000·log(C_t/C_{t-8})` | C; zero for local rows <8 | Appended; mismatched scope |
| 35 | `1000·log(C_t/C_{t-16})` | C; zero for local rows <16 | Appended; mismatched scope |

All values are causal for completed candles. The first tensor row is entirely zero. Invalid/nonfinite parsed fields abort; price sanity violations are diagnosed. There is no learned input z-score normalization; `normalization_version` is persisted metadata, while actual input handling is local scaling/guards and final clamping.

Notably, channels 8–9 are not genuine time-of-day features: with 15-minute candle timestamps, `epochSeconds mod 900` is normally zero, so they collapse to approximately `(0,1)`.

## `input15m.txt` disposition

The file is a PostgreSQL text-table dump, not runtime input: 369,855 rows from `2010-01-03 17:00:00` through `2024-12-31 16:45:00`, pipe-delimited, with header/separator/footer. OHLC rows are internally valid.

| Column | Disposition |
|---|---|
| `dt` | Used only to derive channels 8–11; also retained for diagnostics |
| `open` | Used only to derive features |
| `close` | Used only to derive features and runtime labels |
| `high` | Used only to derive features and look-ahead classification labels |
| `low` | Used only to derive features and look-ahead classification labels |
| `vol` | Completely unused |
| `target` | Completely unused; not the training target |

`vol` is nonzero in every data row (range 1–10,779; mean about 774). `target` is nonzero in every row and equals `(high+low+close)/3`, rounded to six decimals. It is neither consumed by `Feature` nor used as a label. Runtime classification labels instead use future high/low barrier hits, so `target` must not become an input.

## Changes since the 36-channel review

On this branch, `ef67967` is the most recent LSTM-pipeline closure. It made correctness changes, not information-bearing additions:

- deterministic New York timestamp parsing;
- classification inference clamp parity with training;
- review artifacts.

The most recent active information-bearing additions remain older: `dbd6055` added EMA spreads/slopes, range expansion, and body strength; `96f0974` introduced the multi-horizon returns.

A separate branch contains the intended later feature work:

- `4616f2a`: actual Donchian-20 channels, width 38, causality tests;
- `843c967`: Donchian zero-ablation/provenance and paired campaign arms;
- `2817bcf`: 36/38 prefix compatibility;
- `87c8b06`: global-history return parity repair.

None of those commits is reachable from current `HEAD`; only later provenance remnants were copied here in `1a42eb7`. Therefore calling the current `donchian20_mode=enabled` a live model feature would be incorrect.

## Candidate ranking

| Rank | Candidate | Information vs redundancy | Main risk |
|---:|---|---|---|
| 1 | Causal Donchian-20 upper/lower distances | Distinct rolling-extrema/breakout context; not represented by EMA, ATR, or current-candle range | Requires branch integration plus explicit 36/38 compatibility |
| 2 | True UTC session-phase sin/cos, append-only | Existing “time” columns 8–9 are degenerate; session regime is distinct and causal | Two channels; must preserve existing 8–9 semantics rather than reinterpret them |
| 3 | Relative tick-volume activity | Plausibly strong and distinct; file volume is populated | Production `candlestick` volume contract is not present in checkout; parser/Feature/raw path changes required |
| 4 | Volatility-regime ratio | Easy from existing prices | Redundant with channels 6, 31, ATR-normalized distances, and ranges |
| 5 | Momentum acceleration | Causal and cheap | Substantially redundant with 1/4/8/16 returns, rolling return, EMA slopes |

## Recommendation: causal Donchian-20 breakout context

Add exactly two base channels:

```text
donchian_up_20(t)   = 1000 · log(close_t / max(high_{t-20} … high_{t-1}))
donchian_down_20(t) = 1000 · log(close_t / min(low_{t-20} … low_{t-1}))
```

- Lookback: prior 20 completed bars, explicitly excluding current H/L.
- Warm-up: first source row remains all-zero; thereafter use available prior history; zero a direction only if no valid prior extrema.
- Scaling: existing `kFeatureScale=1000`; retain final `[-10,10]` clamp.
- Added channels: 2.
- New layout: 34 base + four returns = **`n_in=38`**.
- Causality: strict; only current completed close and prior extrema.
- Scientific control: `zero_ablation` must retain width 38 but zero only these two columns.

It wins over volume because it needs no new raw field, no unverified live database contract, and already has a self-contained implementation/test history on the local feature branch. It wins over session phase because the high/low breakout state is more directly tied to price structure and does not change calendar semantics.

Required future implementation surface:

- New/ported pure layout and calculation helpers, plus [Params.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/Params.hpp:73>), [Tensor.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/Tensor.hpp:28>), and [Tensor.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/Tensor.cpp:77>).
- [LSTM.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/LSTM.cpp:1917>) and [LSTM/main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:4693>) for the explicit 36/38 model-input contract and global-history return parity.
- [PgModelIO.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/PgModelIO.hpp:150>) for persisted feature-layout/mode metadata and validation.
- Scheduler launch wiring; current `donchian20_mode` only exists in recommendation identity/materialization, not the LSTM CLI/runtime.
- Retain [migration 060](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/060_donchian20_mode.sql:1>) and extend model provenance rather than silently relying on width.
- Add causality, zero-ablation, 36/38 projection, model-metadata mismatch, train/inference vector-parity, and old-column-regression tests. Update Xcode target membership.

For experiments, use matched train/infer ranges, seed/configuration, labels, horizon, and training budget; vary only `donchian20_mode=enabled` versus `zero_ablation`. Compare the predeclared evaluation metrics and class distributions. Do not compare against a 36-wide model as the primary causal estimate.

## Uncertainty / blocker

The only material unresolved evidence issue is the live `candlestick(...)` definition: it is absent from this source tree, so volume availability/meaning in production is unverified. That excludes volume from the recommended increment.

No files were changed, no database operations or experiments were run, and no build/tests were run for this read-only review.

`git status --short`: clean.
`git diff --stat`: no changes.