# TG2 causal trend-line break, retest, and empirical behavior measurement

TG2 is a reusable diagnostic layer in
`Headers/CausalTrendLineBreakRetestBehavior.hpp`. It composes over TG1B, which
in turn composes over TG1A. TG2 does not feed `Tensor`, alter `FeatureLayout`,
change feature-ablation semantics, change semantic layout 7, or register or
publish a worker.

The supplied 90% statements are hypotheses. TG2 contains no 0.9 constant,
prior, threshold, expected rate, acceptance gate, or trading rule. A reported
rate is only `successes / (successes + failures)` for the observed resolved
population. Censored and structurally ineligible observations are reported
separately. Historical frequency is not an assumed future probability.

## Source audit and semantic disposition

The audit covered the complete TG1A/TG1B headers, tests, and documentation;
the PostgreSQL candlestick and fractal functions; indicator functions; the
causal historical-level proximity feature and tests; and the existing
diagnostic/evaluation conventions.

The authoritative findings are:

- TG1A gives a directionally signed observation distance, but explicitly
  leaves break interpretation to a later layer. No wick/body/close break
  meaning was authoritative before TG2.
- No existing retest or reversal contract was found.
- `Database/forex/candlestick.plpgsql` defines OHLC aggregation. It does not
  identify candlestick formations, formation confirmation, takeout, or
  reversal.
- `CausalHistoricalLevelProximity` causally derives a bounded scalar density
  near corroborated weekly swing levels. Its level candidates are private,
  and it defines neither an externally stable level identity nor a level
  break/reversal event.
- `Database/indicators/pocket.plpgsql` is a retrospective future-searching
  routine and is not a causal support/resistance or candlestick-formation
  contract.

Accordingly, TG2 implements trend-line behavior with named configurable
measurement policies. The S/R hypothesis is deferred until an authoritative
contract provides stable known-level identity and availability time, level
side/direction, break component and tolerance, reversal threshold/component,
finite horizon, and lifecycle rules. The candlestick hypothesis is deferred
until an authoritative contract provides stable formation identity/type,
causal confirmation time, takeout direction/component/threshold, reversal
threshold/component, horizon, and invalidation rules. The existing OHLC
function alone is not that contract.

## Preserved TG1A/TG1B contracts

TG2 calls `CausalFractalTrendLineAngleClassification::AddCompletedBar` and
observes its live classified candidates. It does not modify either earlier
header. Candidate identity remains the TG1B lifecycle key:

```text
(direction, anchor1 bar, anchor2 bar, creation bar)
```

The diagnostic identity also carries anchor and creation timestamps. Frozen
TG1B class, creation geometry, and event identity/time are copied into every
break event. Later bars can resolve a measurement but cannot rewrite those
fields. Candidate expiry or deterministic TG1A eviction removes only the
ability to start a new episode; it never creates a break and does not cancel an
already pending finite-horizon observation.

All TG1A/TG1B definitions remain unchanged: strict five-candle fractals,
confirmation at `i+2`, candidate construction and projection, ATR convention,
creation-time classification, inclusive LongTerm/Outer/Inner bands, and
Unclassified gaps.

## Break policy and episode state

The neutral default is named `CompletedCloseBeyondLine`. A break is visible
only after the completed candle has been supplied:

```text
UTL: close < projected line - breakPriceTolerance
DTL: close > projected line + breakPriceTolerance
```

The configurable alternative `CompletedWickBeyondLine` substitutes low for a
UTL and high for a DTL. Tolerances are finite nonnegative absolute price units;
zero is exact. Neither policy infers an intrabar event time.

First observation of a newly created/discovered candidate establishes state
and cannot itself emit a transition. An episode emits exactly on the first
valid-side to broken-side transition. Consecutive beyond-line bars do not emit
again. The named `CompletedBarReturnsToValidSide` policy re-arms only after a
later completed bar no longer satisfies the selected break predicate. A
subsequent transition can then begin a new independently identified episode.

Each event records sequence, stable candidate identity, direction, frozen
class, bar/time, named policy, projection, OHLC, selected candle component,
positive penetration, and TG1A-compatible signed distance.

## Retest policy

`WickReachesProjectedLineFromBrokenSide` measures candle-range (wick) contact
with the projected-line tolerance band on a completed bar strictly after the
break:

```text
high >= projected inner line - retestPriceTolerance
and
low  <= projected inner line + retestPriceTolerance
```

This range-overlap rule is symmetric for UTL and DTL and does not call a candle
that gaps wholly across the line a wick contact. The break candle is never its
own retest. Contact is not described as rejection, continuation, an entry, or
a signal. Those are separate absent semantics. The
finite window contains bars `break+1` through `break+retestHorizonBars`,
inclusive. Exact tolerance and exact horizon boundaries count as contact.
Completed windows without contact are failures; unfinished input is censored.
Latency is measured in sequential completed bars.

## Causal Inner-to-Outer pairing and outcomes

The named policy is
`NearestCoexistingOuterBeyondBreakCandle`. At an Inner break, eligible Outer
candidates must:

1. have the same UTL/DTL direction;
2. already coexist in the TG1B live set at the completed break bar;
3. have frozen class `Outer`;
4. project strictly onto the broken side of the Inner projection; and
5. remain beyond the entire break candle, including configured target
   tolerance, so the target was not already contacted during an intrabar path
   whose ordering cannot be recovered.

The eligible Outer with minimum projection distance from the Inner is paired.
Exact ties use the stable TG1B identity ordering. This uses no future bar or
future outcome. It adds no unsupported anchor-age preference. If none exists,
the Inner break is `structurally_ineligible`/unpaired, not a failure.

Target contact is symmetric and wick based on bars strictly after the break:

```text
UTL target: low  <= projected Outer + outerTargetPriceTolerance
DTL target: high >= projected Outer - outerTargetPriceTolerance
```

The target window is `break+1` through
`break+outerTargetHorizonBars`, inclusive. The implementation reports:

- all paired eligible Inner-to-Outer outcomes;
- Inner breaks with a subsequent measured retest;
- retested paired Inner breaks; and
- Outer contact strictly after the retest, within the original target horizon.

If retest and Outer contact occur in one candle, overall Inner-to-Outer may
resolve, but retest-then-Outer cannot resolve on that candle because intrabar
order is unknown. Its first eligible contact bar is the next completed bar.

## Censoring and denominator

Every outcome state is one of `pending`, `succeeded`, `failed`, `censored`, or
`structurally_ineligible`. Explicit finalization changes unfinished pending
windows to `censored/end_of_input`. A completed horizon with no contact is a
failure. No observation is treated as “eventually” successful.

`OutcomeCounts` reports eligible, resolved, successes, failures, censored,
pending, structurally ineligible, and an optional empirical rate. The rate is
present only when `successes + failures > 0`:

```text
empirical rate = successes / (successes + failures)
```

No censored, pending, or unpaired observation enters that denominator.

## Bounds, ordering, and complexity

TG1A/TG1B continue to bound live candidates. TG2 separately defaults to at
most 4,096 active break observations and 4,096 retained auditable observations.
Both are configurable positive limits. Capacity pressure deterministically
selects the oldest retained/active observation, converts pending outcomes to
`censored/capacity_eviction`, transfers its counts to a cumulative accumulator,
and removes the detailed record. It never silently converts an incomplete
window to failure. Event sequences and candidate-state ordering are
deterministic.

For `C` bounded live candidates, `O` bounded retained observations, and `P`
bounded Outer candidates, a normal bar costs `O(O + C)` plus `O(P)` pairing
for each new Inner break. Storage is `O(C + O)`, independent of total stream
length. The focused test processes 50,000 bars while checking both bounds and
deterministic cumulative counts.

## Diagnostics and scope

Per-observation diagnostics contain symbol/timeframe, direction, anchors and
creation identity, frozen class, break bar/time/policy/projection/OHLC,
component, penetration/distance, retest policy/tolerance/horizon/outcome and
latency, paired Outer identity/projection, target tolerance/horizon/outcome and
latency, conditioned outcome, and censor reason. Aggregates expose raw counts
and the exact empirical denominator inputs.

TG2 contains no Fibonacci/confluence, order entry, exit, stop, sizing,
profitability, label/target, training objective, model feature, input-width,
semantic-layout, feature-ablation, scheduler, worker selection, registry, or
publication behavior. Fibonacci and “Outer within up AB Fibonacci” remain TG3.
