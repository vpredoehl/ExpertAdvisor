---
title: "Phase Pocket 3 Prospective Empirical Evaluation Protocol Freeze"
document_type: "run report"
status: "final"
---

# Phase Pocket 3 — prospective empirical evaluation protocol freeze: output

## 1. Repository and baseline state

Work occurred at `/Volumes/Developer SSD/ExpertAdvisor` on branch
`lstm-feature-development`. The pre-edit tree was clean and HEAD was
`41800f4` (*Implement Phase Pocket 2 causal detector*), with Phase Pocket 1
baseline `c1d1b77` in its history. Required Phase Pocket 1/2 documents,
`Headers/CausalPocketDetector.hpp`, and focused detector tests existed.

`ResearchSources/Pockets/Pockets.pdf` existed and was confirmed ignored by
`/ResearchSources/`. It was read only at the relevant definition/indicator
pages. Repository symbol configuration confirmed the exact USDJPY table name
`usdjpyrmp`. No commit was made; unrelated work was preserved.

During prospective review, before any historical Pocket outcome study or
outcome inspection, two corrections were identified and applied. MFE and MAE
were frozen as nonnegative excursions, while directional close return remains
signed; greedy 64-bar temporal thinning now retains
`next_confirmation >= last_retained_confirmation + 64`, the boundary at which
future outcome windows no longer overlap. These corrections do not result from
observed historical performance. They are not a post-result amendment, and no
other frozen methodology was changed.

## 2. Protocol decisions frozen

The protocol freezes the six-symbol TG4-comparable universe and 15-minute
completed-bar timeframe; half-open exploratory (2010–2020), calibration
(2020–2023), validation (2023–2025), and confirmation (2025–2026) partitions;
and the Phase Pocket 2 detector contract without changing it.

It freezes exactly three separately reported lookbacks: 15 as the
source-default baseline, 10 and 20 as sensitivity candidates. 21 is explicitly
not a candidate. All variants share the same event/confirmation semantics,
universe, partitions, outcomes, horizons, censoring, aggregation, and data
quality policy; they cannot be combined into a score or production selection.

## 3. Source-derived and project-defined distinctions

The manual supports the 15-candle default, prior-high/prior-low directional
references, range-orientation labels, any-timeframe applicability, and the
separate concepts of reversion and breakout. It does not define executable
creation, confirmation, contact, fill, horizon, or statistical rules.

Consequently, Phase Pocket 2's predecessor window, strict predicate, one-bar
confirmation, and immutable range remain project operationalization. Phase
Pocket 3's touch/fill comparisons, outcome clock, horizons, continuation race,
censoring, dependence treatment, and uncertainty method are transparent
project-defined empirical choices, not manual claims.

## 4. Exact horizons and outcomes

Outcomes begin on the completed bar after confirmation. Horizons are 4, 16,
and 64 15-minute bars (1, 4, and 16 hours). Reversion measures first inclusive
high/low touch of the directional touch boundary and first inclusive
completed-close contact with the close boundary. It reports rates,
time-to-event, and explicit unresolved/censor status.

Continuation separately reports directional MFE and MAE as nonnegative
excursions: bullish MFE is `max(0, max(high) - P0)` and MAE is
`max(0, P0 - min(low))`; bearish MFE is `max(0, P0 - min(low))` and MAE is
`max(0, max(high) - P0)`. Directional close return remains signed: bullish
`B[c+h].close - P0`; bearish `P0 - B[c+h].close`. All are reported at the
three horizons in pips and Pocket-width multiples. At 64 bars, it also reports
the race between revisit and a one-Pocket-width favorable excursion; same-bar
OHLC ordering is indeterminate rather than guessed.

The nominal tail is 16 hours. Preconfirmation data are strictly clipped before
2025: validation observations needing a 2025 bar are administrative censors,
not dropped or resolved using confirmation data. Gaps, weekends, invalid input,
and other boundaries are right-censored and never bridged or interpolated.

## 5. Aggregation and dependence

Every eligible observation contributes to event-weighted reporting. Equal-symbol
reporting first calculates each symbol statistic and then averages symbols
equally. Reports diagnose simultaneous observations, 64-bar temporal and
price-range overlap, repeated same-direction observations, and UTC-week
clustering.

A greedy 64-bar temporal thinning is frozen only as an outcome-independent
independence sensitivity: after retaining the earliest confirmation, retain a
subsequent observation when `next_confirmation >=
last_retained_confirmation + 64` (reject only coordinates below that boundary).
It does not replace the primary population. Primary
evidence is counts, denominators, rates, differences/ratios, censoring,
quantiles, and cross-symbol/temporal consistency. The frozen uncertainty method
is a deterministic 2,000-replicate hierarchical UTC-week block bootstrap with
symbol resampling for equal-symbol summaries, not an iid bootstrap or a
winner-declaring p-value threshold.

## 6. Temporal and confirmation firewall

Preconfirmation may examine only exploratory, calibration, and validation
under the frozen protocol. It must not read, score, materialize, or use 2025
as a validation tail. Before confirmation is opened, a separately committed
freeze must record the exact commit, configuration hash, schema, results-free
advancement decision, and tail boundary.

Post-result changes to detector semantics, lookbacks, timeframe, outcomes,
horizons, censoring, or aggregation cannot be called confirmatory. New ideas
start a new experiment with a newly reserved confirmation period.

## 7. Files changed

- `docs/phases/pockets/PhasePocket3/PocketProspectiveEmpiricalEvaluationProtocolFreeze.md`
- `docs/phases/pockets/PhasePocket3/PhasePocket3_ProspectiveEmpiricalEvaluationProtocolFreeze_Output.md`

## 8. Validation performed

Before editing: repository root, branch, status, baseline history, required
documents/header/tests, PDF existence/ignore state, and USDJPY table spelling
were checked. The manual was consulted only for source semantics. After
editing, `git diff --check` passed; the untracked-file no-index diff was
reviewed; status/stat, PDF ignore state, and protected-path checks passed.

No Release build is required for this documentation-only phase; no C++ source,
build setting, or executable behavior changed.

## 9. Safety and non-execution confirmation

No historical Pocket outcome study was run. No Pocket outcome statistic was
inspected. No historical Pocket outcome artifact was created. No database
connection or state change occurred. No historical Forex data or materialized
view changed. No scheduler or experiment state changed.

No TG4 process, protected artifact, source, configuration, or confirmation
result was inspected or modified. No Phase 21C work was touched. The manual
remains ignored and unmodified. No commit was made.

## 10. Unresolved issues and recommended Phase Pocket 4 scope

There is no unresolved analytical choice needed to implement the frozen
preconfirmation evaluator. Phase Pocket 4 should implement the minimum
read-only streamed evaluator, pure outcome evaluator, immutable configuration,
machine-readable metadata, atomic publication, and focused synthetic tests for
contact, race, censoring, gap, partition, and aggregation math. It may then
run only the frozen preconfirmation measurement and must exclude 2025 entirely.
