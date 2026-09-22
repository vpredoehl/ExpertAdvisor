# Phase Pocket 2 — Pocket operational definition freeze

## Status, authority, and boundary

**Status: frozen before empirical evaluation.**  This is the LSTM Trader
operational contract for the small, causal Pocket structural detector.  It is
not a claim that every operational rule below is supplied by the manual.

* **Authoritative source:** Market Traders Institute, Inc., *Pockets* manual,
  `ResearchSources/Pockets/Pockets.pdf` (106 PDF pages; read locally and not
  changed).  Citations use printed page then PDF page.
* **Baseline:** Phase Pocket 1, commit `c1d1b77` (*Complete Phase Pocket 1
  source specification*), especially
  `docs/phases/pockets/PhasePocket1/PocketSpecificationAndCausalDetector.md`.
* **Evidence boundary:** no historical Pocket outcome, profitability,
  frequency, TG4 confirmation artifact, or parameter comparison was read to
  choose any rule in this document.

The manual says a Pocket is a price inefficiency associated with
aggressively bullish or bearish abnormal-looking candles (p. 43, PDF p. 48),
defaults its indicator to a 15-candle lookback and identifies a highest high
for bullish output / lowest low for bearish output (p. 48, PDF p. 53), and
draws a two-boundary bullish/bearish range (pp. 44–45, PDF pp. 49–50).  It
does not state a complete OHLC predicate, endpoint formula, equality policy,
or completed-bar confirmation rule.  Phase Pocket 1 correctly did not
implement a detector.  This document adds only the auditable project choices
needed for one.

## Terminology

`B[j]` is one supplied **completed** OHLC bar at zero-based series coordinate
`j`; `B[j].time` is the timestamp supplied for that bar.  A candidate event
at `e` is confirmed by `c = e + 1`.  `L = 15` is the baseline reference
lookback.  Its predecessor window is exactly

```text
W(e) = { B[e - L], ..., B[e - 1] }.
H(e) = max(B[k].high for k in W(e))
M(e) = min(B[k].low  for k in W(e))
```

`event` means the middle, directional completed bar `B[e]`; `confirmation`
means the next completed bar `B[c]`; an **information cutoff** is `B[c]`.
An observation has not happened to a consumer until that cutoff bar has
closed and has been supplied to the detector.

## Rule classification and resolution

| Structural item | Manual / Pocket 1 evidence | Frozen result | Class |
| --- | --- | --- | --- |
| Qualifying ordered predicate | “Aggressively” bullish/bearish abnormal candles, without a threshold (p. 43/PDF 48). | The exact three-part OHLC predicate below. | PROJECT_OPERATIONAL |
| Lookback size | Indicator default is 15 candles (p. 48/PDF 53). | `L = 15` baseline/default. | EXPLICIT; parameter is SOURCE_DEFAULT |
| Lookback membership | “look back” does not say whether the event belongs. | Exactly the 15 completed predecessors; the event and confirmation are excluded. | PROJECT_OPERATIONAL |
| Tie policy | No equality policy. | Extrema are numeric values; ties in `W(e)` are allowed and need no chosen anchor.  All break/gap inequalities are strict. | PROJECT_OPERATIONAL |
| Direction | Manual has bullish/bearish Pockets and maps bullish to buy / bearish to sell in breakout strategy (p. 52/PDF 57; p. 81/PDF 86). | A bullish event has `close > open`; bearish has `close < open`.  No size threshold is claimed. | PROJECT_OPERATIONAL |
| Price orientation | Diagrams label bullish upper=`touch`, lower=`close`; bearish upper=`close`, lower=`touch` (pp. 44–45/PDF 49–50). | Directional touch/close endpoint labels below. | EXPLICIT for orientation; PROJECT_OPERATIONAL for OHLC formulas |
| Event time | Creation is discussed, not timestamped. | Event coordinates are `e` and `B[e].time`. | PROJECT_OPERATIONAL |
| Confirmation | “immediate confirmation” is named for breakout entry (p. 52/PDF 57); larger timeframes take longer to confirm (p. 46/PDF 51), but no bar rule is given. | One later completed bar provides the strict range confirmation. | PROJECT_OPERATIONAL |
| Touch / close evaluation | Touch and close are labelled levels/strategy targets, not wick/bid/ask/intrabar rules (pp. 44–45/PDF 49–50; pp. 60–61/PDF 65–66). | Labels are retained on immutable endpoints only; no later contact, fill, or target evaluation is structural. | DISCRETIONARY / UNSPECIFIED; excluded |
| Lifecycle, replacement, invalidation | “Available” and consecutive Pockets appear in strategy prose, but no structural expiry/replacement rule is given (p. 60/PDF 65; p. 81/PDF 86). | Each qualifying event emits one immutable observation.  Later bars cannot replace, invalidate, close, or mutate it. | PROJECT_OPERATIONAL |
| Timeframe | Pockets and indicator work on any timeframe (pp. 46, 48/PDF 51, 53).  Cross-timeframe chart marking is guidance. | One detector instance consumes one strictly ordered completed-bar series with an opaque non-empty timeframe identity; it neither aggregates nor joins timeframes. | EXPLICIT for applicability; PROJECT_OPERATIONAL for interface |

“Aggressive” remains a source-supported qualitative concept, not a hidden
size/ATR/news/volume threshold.  Adding such a threshold would require a new
frozen project policy and is not part of this phase.

## Exact frozen structural creation predicate

The detector considers a candidate only after `B[c]` is supplied, so it
requires `e >= L` and `c = e + 1`.  All bars named in the expression must be
valid completed OHLC bars.

| Direction | Event predicate | Immutable range | Endpoint labels |
| --- | --- | --- | --- |
| Bullish | `B[e].close > B[e].open`, `B[e].close > H(e)`, and `B[c].low > H(e)` | `lower = H(e)`, `upper = B[c].low` | `close = lower`; `touch = upper` |
| Bearish | `B[e].close < B[e].open`, `B[e].close < M(e)`, and `B[c].high < M(e)` | `lower = B[c].high`, `upper = M(e)` | `touch = lower`; `close = upper` |

The strict final inequality creates a non-zero, ordered range.  It is a
conservative project operationalization of the manual’s drawn range and
directional prior-extreme references, not a manual-provided formula.  The
strict event-close condition prevents a wick-only excursion from being called
a directional completed-bar event.  It is selected for deterministic
completed-bar semantics, not because it produced any observed outcome.

The event timestamp is `B[e].time`.  Confirmation timestamp and information
cutoff timestamp are both `B[c].time`; their bar coordinates are both `c`.
Thus `eventBar < confirmationBar` for every emitted observation (and hence
the required `eventBar <= confirmationBar` invariant holds).

## Information timing and replay invariants

1. Input is supplied completed bars only, in strictly increasing timestamp
   order.  No partial bar, future bar, database read, scheduler state, or
   mutable global state is consulted.
2. A prefix ending at `B[e]` emits no observation for `e`; the earliest legal
   emission is while adding completed `B[e + 1]`.
3. At emission, all inputs are in `W(e)`, `B[e]`, or `B[e + 1]`.  The emitted
   event, confirmation, cutoff, range, direction, and timeframe are then
   immutable.
4. Replaying the same valid prefix in the same order emits the same sequence.
   Extending a prefix cannot revise an earlier observation.
5. Invalid/non-finite or physically inconsistent OHLC input, or a
   non-increasing timestamp, is rejected rather than repaired or reordered.

## Lifecycle and strategy boundary

The structural detector is append-only.  Qualifying consecutive and
alternating events each produce independent observations.  It has no active
Pocket collection and makes no judgement that a later Pocket supersedes an
earlier one.  This is necessary because the source does not define structural
replacement, expiry, fill, or invalidation.

For the same reason, the detector does **not** decide whether a later wick,
close, bid/ask quote, intrabar tick, order, or position “touches,” “closes,”
or fills a Pocket.  The endpoint label describes the source diagram’s range
orientation only.  Reversion and breakout are separate strategy uses:
reversion targets a Pocket/inefficiency while breakout trades in its direction
(pp. 52, 60–61/PDF 57, 65–66).  They are not creation or lifecycle rules.

The following manual-supported but discretionary/execution concepts are
intentionally excluded: trendline breaks, divergence, price patterns,
support/resistance, ATR distance/levels, entry count, grids, position size,
orders, stops, profit targets, fractal or ATR trailing stops, and P&L.

## Parameter policy

| Parameter | Frozen value / proposal | Classification | Reason |
| --- | --- | --- | --- |
| Reference lookback `L` | 15 bars | SOURCE_DEFAULT | Manual indicator default; baseline, not an optimality assertion. |
| Predecessor-only membership | 15 bars before event | PROJECT_FIXED | Resolves inclusion without mixing event/confirmation information into the reference. |
| Confirmation latency | 1 completed bar | PROJECT_FIXED | Makes the drawn-range operationalization causal and records its cost explicitly. |
| Endpoint/contact tolerance | 0 price units; strict inequalities | PROJECT_FIXED | Avoids an unprovided instrument/pip/tick tolerance. |
| Directional body threshold | strict sign only (`0`) | PROJECT_FIXED | Gives a deterministic direction without inventing an aggression magnitude. |
| Later lookback comparison | candidate set `{10, 15, 20}`; 15 remains baseline | EMPIRICAL_CANDIDATE | A small shorter/default/longer sensitivity set, frozen before evaluation; no comparison is run here. |

No source says 21.  It is neither the baseline nor privileged as a candidate
because of prior recollection.

## Later empirical study design — not executed

A later, separately authorized phase may compare the predeclared `{10,15,20}`
variants only after fixing data eligibility and outcome definitions.  It must
keep 15 as the manual-default baseline; distinguish structural event frequency
from predictive/trading utility; and retain each variant’s one-bar
event/confirmation timing.  Before looking at outcomes it must predeclare
bounded train, calibration, validation, and final confirmation periods,
evaluate stability across symbols, and cluster/weight overlapping observations
instead of treating them as independent.  Reversion outcomes and breakout
outcomes must be evaluated separately.  No winning lookback is asserted here.

## Implementation eligibility

All rules needed by the intentionally small structural detector are now either
source-grounded or clearly labelled PROJECT_OPERATIONAL.  Touch/fill and
lifecycle evaluation are excluded rather than guessed.  The accompanying
implementation may therefore be pure, supplied-bar, deterministic, and
strategy-free.
