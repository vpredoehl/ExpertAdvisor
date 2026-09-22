# Phase Pocket 3 — prospective empirical evaluation protocol freeze

## Status, authority, and scope

**Status: frozen before historical Pocket outcomes are examined.** This is the
first empirical protocol for the Phase Pocket 2 causal observations. It does
not run an empirical study or select a lookback, strategy, threshold, horizon,
outcome, or model feature.

| Layer | Authority and role |
| --- | --- |
| Source semantics | Market Traders Institute, Inc., *Pockets* manual, `ResearchSources/Pockets/Pockets.pdf`. |
| Project operationalization | The immutable causal detector contract frozen in Phase Pocket 2. |
| Empirical protocol | This document's outcome, censoring, aggregation, and reporting choices. |
| Empirical results and model integration | Explicitly excluded from Phase Pocket 3. |

Baselines are Phase Pocket 1 commit `c1d1b77` (*Complete Phase Pocket 1 source
specification*) and Phase Pocket 2 commit `41800f4` (*Implement Phase Pocket 2
causal detector*). The manual is authoritative only for what it establishes.
It does not make the project-defined OHLC predicate, endpoints, contact rules,
horizons, or statistics into source facts.

The manual supports the 15-candle indicator default and directional
highest-high/lowest-low references (manual p. 48, PDF p. 53), range-orientation
labels (manual pp. 44–45, PDF pp. 49–50), any-timeframe applicability, and
separate reversion and breakout concepts. It does not provide complete OHLC
creation, confirmation, touch/fill, lifecycle, or empirical-reporting rules.
The ignored local PDF was read only for these bounded checks and is not changed,
moved, staged, or committed.

No historical Pocket outcome, frequency, return, touch/fill, continuation,
profitability, or lookback-comparison statistic was inspected to select this
protocol. No historical Pocket study or outcome artifact is created here.

During prospective review, before any historical Pocket outcome study or
outcome inspection, two corrections were identified and applied: continuation
MFE/MAE are nonnegative excursions, and 64-bar temporal thinning retains a
confirmation at the non-overlapping boundary. These corrections do not result
from observed historical performance. No other frozen methodology is changed.

## Frozen Phase Pocket 2 structural contract

For a lookback `L`, supplied completed OHLC bars are `B[j]`. For event
coordinate `e`, the predecessor-only reference is:

```text
W(e) = { B[e - L], ..., B[e - 1] }
H(e) = max(B[k].high for k in W(e))
M(e) = min(B[k].low  for k in W(e))
c = e + 1
```

The event is `B[e]`, confirmation is next completed bar `B[c]`, and information
cutoff is `B[c]`. The observation is available only after `B[c]` completes and
is supplied. Thus `eventBar < confirmationBar == informationCutoffBar`; no
later bar can revise an emitted observation.

| Direction | Frozen event/confirmation predicate | Immutable range and endpoint labels |
| --- | --- | --- |
| Bullish | `close > open`, `B[e].close > H(e)`, `B[c].low > H(e)` | `lower = H(e)`, `upper = B[c].low`; close=`lower`, touch=`upper`. |
| Bearish | `close < open`, `B[e].close < M(e)`, `B[c].high < M(e)` | `lower = B[c].high`, `upper = M(e)`; touch=`lower`, close=`upper`. |

Extrema ties in `W(e)` are permitted; all shown break/gap inequalities are
strict; range width is positive. Phase Pocket 2 emits independent immutable
observations for qualifying consecutive and alternating events. It has no
active Pocket collection, expiry, replacement, invalidation, later contact
state, entry, stop, target, position, P&L, trendline, divergence, pattern,
news, support/resistance, or ATR rule.

The source-default baseline is `L = 15`; the only sensitivity candidates are
`L = 10` and `L = 20`. **21 is not a candidate**: it is neither supplied by
the manual nor justified by prior recollection. Each variant retains the same
one-completed-bar confirmation and all other structural semantics.

The Phase Pocket 2 header currently exposes the frozen source-default
`L = 15` detector. Phase Pocket 4 must make sensitivity support explicit
without changing the formula: either parameterize that pure detector or use an
equivalent pure evaluator implementation with regression evidence that
`L = 15` reproduces the frozen detector. That implementation work is not a
semantic amendment to Phase Pocket 2.

A later evaluator can consume observations without future information: it
streams completed bars in increasing timestamp order and admits an observation
only when its confirmation bar has completed. Post-availability bars belong
only to the separate outcome evaluator and cannot mutate the observation.

## Study universe, timeframe, and access

Initial study identity: `pocket-prospective-preconfirmation-v1`.

| Display symbol | Repository table | Pip size |
| --- | --- | ---: |
| AUDCAD | `audcadrmp` | 0.0001 |
| AUDUSD | `audusdrmp` | 0.0001 |
| EURUSD | `eurusdrmp` | 0.0001 |
| GBPUSD | `gbpusdrmp` | 0.0001 |
| USDCAD | `usdcadrmp` | 0.0001 |
| USDJPY | `usdjpyrmp` | 0.01 |

`usdjpyrmp` is verified against the repository's canonical supported-symbol
list; prompt whitespace is not part of its table name. The six symbols match
TG4 for comparability, not because of Pocket results.

The only initial timeframe is completed **15-minute** bars (900-second
cadence). The manual permits other timeframes, but that does not authorize a
timeframe grid search. Other timeframes require a later predeclared study.

The future evaluator uses a read-only historical market-data repository under
PostgreSQL `REPEATABLE READ, READ ONLY`. It may query only these canonical
tables; it must not write data, refresh a view, or access scheduler/experiment
tables. The adapter must stream one strictly increasing completed-bar sequence
per symbol and record its timestamp convention.

## Partitions, warmup, and resolution tail

All timestamps are UTC and all scoring ranges are half-open:

| Partition | Confirmation timestamp `t` |
| --- | --- |
| Exploratory | `2010-01-01T00:00:00Z <= t < 2020-01-01T00:00:00Z` |
| Calibration | `2020-01-01T00:00:00Z <= t < 2023-01-01T00:00:00Z` |
| Validation | `2023-01-01T00:00:00Z <= t < 2025-01-01T00:00:00Z` |
| Confirmation | `2025-01-01T00:00:00Z <= t < 2026-01-01T00:00:00Z` |

Partition membership is exclusively by confirmation timestamp, not event time
or outcome. An event may precede a boundary only when its confirmation is in
the scoring range. This preserves availability-time causality.

For every symbol/lookback/partition, require 21 contiguous completed warmup
bars before the first possible scoring confirmation: 20 predecessors for the
largest lookback plus its event bar. Insufficient or discontinuous warmup is
counted and makes affected candidates structurally unevaluable; it is never
repaired or inferred.

The maximum outcome horizon is 64 bars. The nominal partition-resolution tail
is the 16 elapsed hours immediately after the exclusive partition end:

| Run | Authorized scoring partitions | Latest data that may be read for resolution |
| --- | --- | --- |
| Preconfirmation | exploratory, calibration, validation | strictly before `2025-01-01T00:00:00Z` |
| Confirmation, after a new freeze only | confirmation | strictly before `2026-01-01T16:00:00Z` |

Exploratory and calibration outcomes may use their fixed tails inside the
authorized preconfirmation range. The validation tail is clipped at the 2025
firewall: an event needing a 2025 bar is administratively censored, not
dropped and not resolved with confirmation data. The preconfirmation run may
not read, score, or materialize any 2025 Pocket event or outcome.

## Outcome clock, completeness, and censoring

Let `c` be the confirmation coordinate and `P0 = B[c].close`. Structural data
in `B[c]` are known at availability but are not outcomes. The outcome clock
starts at next completed bar `B[c+1]`. A horizon `h` includes exactly
`{B[c+1], ..., B[c+h]}` and resolves at the timestamp of `B[c+h]`.

| Horizon | Bars | 15-minute elapsed time | Question |
| --- | ---: | ---: | --- |
| Short | 4 | 1 hour | Immediate post-confirmation behavior. |
| Medium | 16 | 4 hours | Substantial intraday behavior. |
| Long | 64 | 16 hours | Bounded multi-session behavior. |

These are distinct temporal questions, selected without outcome inspection;
they are not targets or an optimization grid.

A complete window has every timestamp from `B[c]` through `B[c+h]` exactly
900 seconds apart and inside the authorized resolution range. Duplicate or
out-of-order timestamps, malformed/non-finite OHLC, or a cadence interval
other than 900 seconds is a data-quality failure. The outcome is right-censored
immediately before the first unusable bar. No interpolation occurs, and no
outcome bridges a weekend, closure, or other gap. A tail/boundary-limited window
is administratively censored at the number of valid future bars available.

Every horizon report states eligible observations, complete windows, hits,
unresolved/censored observations, censoring reason (gap, invalid input,
boundary, tail), and valid future-bar count. Horizon rates use complete-window
denominators; censors and right-censored time-to-event summaries are reported
alongside them, never silently removed.

## Reversion outcomes: project-defined evaluation semantics

The manual names directional touch and close levels but does not prescribe
executable contact comparisons. These rules are outcome semantics only and do
not retrofit lifecycle behavior into `CausalPocketDetector.hpp`. Let `T` be
`TouchPrice()`, `C` be `ClosePrice()`, and `j > c` be a valid outcome bar:

| Outcome | Bullish Pocket | Bearish Pocket | Rule |
| --- | --- | --- | --- |
| First touch | `B[j].low <= T` | `B[j].high >= T` | Inclusive wick/high-low contact; first qualifying `j`. |
| Close-boundary fill | `B[j].close <= C` | `B[j].close >= C` | Inclusive completed-close contact; first qualifying `j`. |

Equality counts. The `touch` label motivates high/low contact and the `close`
label motivates completed-close contact, but the manual leaves this ambiguous;
these are transparent project choices. Bid/ask, spread, tick ordering, and
intrabar path are not inferred from OHLC.

At 4, 16, and 64 bars, report first-touch and close-boundary-fill rates,
time-to-touch, time-to-fill, and unresolved/censored status. Event time is
`j - c` future completed bars. Medians/quantiles use resolved times with
censor count reported; if fewer than half reach an outcome by 64 bars, report
the median as not reached within 64 rather than extrapolating. No discretionary
filter or trade-management rule participates.

## Continuation outcomes: separate from reversion

For complete horizon `h`, extrema are over `{c+1, ..., c+h}`. Define:

| Metric | Bullish Pocket | Bearish Pocket |
| --- | --- | --- |
| MFE | `max(0, max(high) - P0)` | `max(0, P0 - min(low))` |
| MAE | `max(0, P0 - min(low))` | `max(0, max(high) - P0)` |
| Directional close return | `B[c+h].close - P0` | `P0 - B[c+h].close` |

Report each metric at every frozen horizon in pips and Pocket-width multiples.
With `w = upper - lower > 0`, a signed price movement's width multiple is the
movement divided by `w`. ATR normalization is excluded from the primary study.

Freeze one long-horizon descriptive race. A one-Pocket-width favorable
excursion occurs at first `j` where bullish `high >= P0 + w` or bearish
`low <= P0 - w`. Compare that bar with first touch within 64 bars. Report:
continuation first, revisit first, same-bar intrabar-order-indeterminate,
neither in a complete window, or censored. A same-bar result is never assigned
to either side. This is neither an entry/exit rule nor a profit target.

## Structural descriptives, dependence, and aggregation

Structural metrics are not win rates. By lookback, symbol, direction, and
partition, report:

- emitted/eligible counts; bullish/bearish counts and shares;
- counts per structurally evaluable confirmation bar and per 10,000 such bars;
- range width in price and pips; confirmation/event spacing in bars and time;
- simultaneous observations (same confirmation timestamp);
- temporal overlap of `[c+1, c+64]` windows and inclusive price-range overlap
  of `[lower, upper]` among temporally overlapping observations of a symbol;
- repeated same-direction observations within 64 future bars; and
- UTC-calendar-week event cluster counts and their maximum/quantiles.

Observations are not iid. Every report provides both:

1. **Event-weighted:** every eligible observation contributes once; pooled
   rates use all complete eligible observations.
2. **Equal-symbol:** calculate each symbol's statistic, then take an unweighted
   mean over symbols with a defined denominator and state contributor count.

Keep directions, lookbacks, and partitions separate. Neither population
replaces the other, and no outcome-dependent filter is permitted.

The independence-oriented sensitivity is fixed as symbol/lookback-specific
greedy temporal thinning: sort by confirmation coordinate then immutable
observation identity; retain the earliest and thereafter only observations with
confirmation coordinate greater than or equal to the last retained coordinate
plus 64 (`next_confirmation >= last_retained_confirmation + 64`). Equivalently,
reject only a confirmation coordinate less than that boundary. Direction does
not change the rule. This outcome-independent sensitivity never
replaces the primary event-weighted dataset.

## Lookback sensitivity, units, and statistical reporting

Report exactly these detector variants separately:

| Lookback | Role |
| ---: | --- |
| 15 | Source-default baseline. |
| 10 | Predeclared shorter sensitivity candidate. |
| 20 | Predeclared longer sensitivity candidate. |

All three have identical structural semantics, universe, partitions, outcomes,
horizons, tail, aggregation, and censoring. Do not create a composite score,
rank a production winner, or choose a strategy/threshold from preconfirmation.

Canonical pip conversion is `0.0001` for AUDCAD, AUDUSD, EURUSD, GBPUSD, and
USDCAD and `0.01` for USDJPY. Pip distance is absolute price distance divided
by pip size; continuation values retain directional sign before conversion.
Cross-symbol reporting uses pips and Pocket-width multiples.

Primary reporting is counts/denominators, rates, absolute differences,
meaningful ratios, resolved-time/excursion medians and 25th/75th quantiles,
censoring, and separate symbol/period results. No p-value or numerical
advancement threshold declares a winner.

Report 95% percentile uncertainty intervals using a fixed 2,000-replicate
hierarchical bootstrap. Within each symbol/lookback/partition cohort, resample
UTC-calendar-week confirmation blocks with replacement, retaining each selected
week's observations. Pool resampled symbols for event-weighted summaries; for
equal-symbol summaries, sample six symbols with replacement and resample weeks
inside each selected symbol. Seed the process deterministically from the
immutable configuration SHA-256, record its derived seed, and mark undefined
intervals with no valid denominator. This is dependence-aware descriptive
uncertainty, not proof of independence or causal effect.

## Data quality, causal invariants, and metadata

The evaluator reports per symbol/lookback/partition: duplicate and out-of-order
timestamps; malformed/non-finite OHLC; insufficient/discontinuous warmup;
cadence gaps (both timestamps and duration); observations outside scoring;
complete and unresolved outcomes; censoring reason; and tail bars
requested/used. It fails rather than repairs structural input.

Immutable metadata records Git commit, study identity, canonicalized
configuration and SHA-256, source and Phase Pocket 2 baselines, read-only and
isolation status, symbol/table, lookback, timeframe, scoring/resolution ranges,
timestamp convention, warmup/tail policy, data-quality counts, and
observation/outcome/censoring counts.

Required causal invariants:

- eligibility occurs only at completed confirmation/information cutoff;
- outcomes start only on the next bar and cannot modify structure;
- partition membership uses confirmation timestamp, never a later outcome;
- resolution uses only the frozen horizon and authorized tail; and
- selection, thinning, aggregation, and exclusion never depend on future
  outcome.

## Preconfirmation questions and confirmation firewall

The later run will answer, without presupposing success:

1. Does the source-default 15-bar detector identify structurally stable Pocket
   populations across symbols and exploratory/calibration/validation time?
2. How frequently and quickly do confirmed Pockets objectively touch or
   close-boundary-fill within frozen horizons?
3. What MFE, MAE, directional return, and revisit-versus-one-width continuation
   behavior follows confirmation when distinct from reversion?
4. Are structural and outcome descriptions materially sensitive to 10, 15, and
   20 bars?
5. Are descriptions consistent across symbols and validation time, or driven by
   one pair, cluster, or era?

The 2025 confirmation partition is fully unread, unscored, and unmaterialized
during preconfirmation, including as a validation outcome tail. Before it is
opened, a separately committed confirmation freeze must record Git commit,
configuration hash, output schema, results-free advancement decision, and
authorized tail. This protocol has no winner rule: absent a future justified
freeze, all three variants remain separately reported in confirmation.

Detector semantics, universe, timeframe, lookbacks, outcomes, horizons,
censoring, aggregation, and uncertainty cannot be changed after
preconfirmation inspection and then represented as confirmatory. A new
post-result hypothesis begins a new experiment with a new untouched period.

## Minimum future evaluator architecture

```text
read-only market-data repository
  -> completed-bar validation / causal stream
  -> pure CausalPocketDetector (symbol, lookback)
  -> pure post-availability outcome evaluator
  -> deterministic aggregation / uncertainty reporter
  -> machine-readable metadata and atomic output publication
```

Configuration is immutable after parsing/canonicalization. Publish only by
writing to a newly created temporary sibling directory and atomically renaming
it to a previously absent target; failed runs publish no partial study.
Suggested immutable outputs are metadata, one structural-observation row with
outcome/censor fields, cohort aggregates, and gap/data-quality rows. Observation
identity includes study/configuration, symbol, lookback, timeframe, event, and
confirmation coordinates/timestamps.

Phase Pocket 4 alone may implement this minimal read-only evaluator and focused
synthetic protocol tests. This phase changes no LSTM feature/layout/target,
training, inference, profitability, scheduler, experiment table, Forex data,
or TG4 component.
